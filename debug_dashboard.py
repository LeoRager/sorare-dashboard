import bcrypt
import requests
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
from graphql import build_schema
from typing import List, Optional
import pandas as pd
import numpy as np
import soccerdata as sd
from thefuzz import process

AUD = "choose_decisive_player"
GRAPHQL_ENDPOINT = "https://api.sorare.com/graphql"


def load_local_schema(path: str = "schema.graphql"):
    with open(path, "r", encoding="utf-8") as fh:
        sdl = fh.read()
    return build_schema(sdl)


def get_salt_for_email(email: str) -> str:
    url = f"https://api.sorare.com/api/v1/users/{email}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data.get("salt")


def hash_password(password: str, salt: str) -> str:
    salt_bytes = salt.encode("utf-8")
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt_bytes)
    return hashed.decode("utf-8")


def make_client(jwt_token: Optional[str] = None, aud: Optional[str] = None, local_schema=None):
    headers = {"Content-type": "application/json"}
    if jwt_token:
        headers["Authorization"] = f"Bearer {jwt_token}"
    if aud:
        headers["JWT-AUD"] = aud

    transport = RequestsHTTPTransport(
        url=GRAPHQL_ENDPOINT,
        use_json=True,
        headers=headers,
        verify=True,
    )

    if local_schema is not None:
        return Client(transport=transport, schema=local_schema, fetch_schema_from_transport=False)
    return Client(transport=transport, fetch_schema_from_transport=False)


def sign_in_with_password(email: str, hashed_password: str, aud: str, client: Client):
    mutation = gql(
        '''
        mutation SignIn($input: signInInput!) {
          signIn(input: $input) {
            currentUser { slug }
            jwtToken(aud: "%s") { token expiredAt }
            otpSessionChallenge
            errors { message }
          }
        }
        ''' % aud
    )
    variables = {"input": {"email": email, "password": hashed_password}}
    return client.execute(mutation, variable_values=variables)["signIn"]


def sign_in_with_otp(challenge: str, otp: str, aud: str, client: Client):
    mutation = gql(
        '''
        mutation SignInWithOtp($input: signInInput!) {
          signIn(input: $input) {
            currentUser { slug }
            jwtToken(aud: "%s") { token expiredAt }
            errors { message }
          }
        }
        ''' % aud
    )
    variables = {"input": {"otpSessionChallenge": challenge, "otpAttempt": otp}}
    return client.execute(mutation, variable_values=variables)["signIn"]


def fetch_owned_cards(jwt_token: str, aud: str, local_schema, page_size: int = 100):
    client = make_client(jwt_token=jwt_token, aud=aud, local_schema=local_schema)

    query = gql("""
    query MyCards($first: Int!, $after: String) {
      currentUser {
        cards(first: $first, after: $after) {
          nodes {
            slug
            ... on Card {
              rarity
              player {
                firstName
                lastName
                activeClub {
                  name
                }
                nextClassicFixturePlayingStatusOdds {
                  starterOddsBasisPoints
                  reliability
                }
                nextGame {
                  date
                  homeTeam { name }
                  awayTeam { name }
                }
              }
            }
          }
          pageInfo {
            endCursor
            hasNextPage
          }
        }
      }
    }
    """)

    cards = []
    after = None

    while True:
        variables = {"first": page_size, "after": after}
        result = client.execute(query, variable_values=variables)

        cards_conn = result.get("currentUser", {}).get("cards", {})
        nodes = cards_conn.get("nodes", [])

        for node in nodes:
            player = node.get("player") or {}
            club = player.get("activeClub") or {}
            odds = player.get("nextClassicFixturePlayingStatusOdds") or {}
            next_game = player.get("nextGame") or {}
            home_team = (next_game.get("homeTeam") or {}).get("name")
            away_team = (next_game.get("awayTeam") or {}).get("name")

            cards.append({
                "first_name": player.get("firstName"),
                "last_name": player.get("lastName"),
                "team": club.get("name"),
                "rarity": node.get("rarity"),
                "starter_odds_bp": odds.get("starterOddsBasisPoints"),
                "odds_reliability": odds.get("reliability"),
                "next_game_date": next_game.get("date"),
                "next_game_home": home_team,
                "next_game_away": away_team,
            })

        page_info = cards_conn.get("pageInfo") or {}
        if not page_info.get("hasNextPage"):
            break
        after = page_info.get("endCursor")

    return cards


def clean_data(df):
    df = df.dropna(subset=[col for col in df.columns if col not in ["starter_odds_bp", "odds_reliability"]])
    df = df.drop_duplicates()
    df["next_game"] = np.where(
        df["team"] == df["next_game_home"],
        df["next_game_away"],
        df["next_game_home"]
    )
    df = df.drop(columns=["next_game_home", "next_game_away"])
    return df


def calculate_avg_xG_conceded(df):
    df["avg_xG_conceded_league"] = (
        df.groupby("league")["npxG_against_p90"].transform("mean")
    )
    return df


def get_fbref_stats(df: pd.DataFrame, season="2025-26"):
    fbref = sd.FBref(leagues=["Big 5 European Leagues Combined"], seasons=[season])
    player_stats = fbref.read_player_season_stats(stat_type="standard")
    fbref_teams = player_stats.index.get_level_values("team").unique().tolist()

    def build_team_map(sorare_teams, fbref_teams, threshold: int = 80) -> dict:
        hardcoded = {
            "RCD Espanyol de Barcelona": "Espanyol",
            "Stade Rennais F.C.": "Rennes",
            "Real Valladolid CF": "Valladolid",
            "Paris Saint-Germain": "Paris S-G",
        }

        team_map = {}
        for sorare_team in sorare_teams:
            if sorare_team in hardcoded:
                team_map[sorare_team] = hardcoded[sorare_team]
                continue
            match_name = sorare_team
            for token in ["FC", "F.C.", "RC", "CF", "RCD"]:
                match_name = match_name.replace(token, "")
            match_name = match_name.strip()
            best_match, score = process.extractOne(match_name, fbref_teams)
            team_map[sorare_team] = best_match if score >= threshold else None
        return team_map

    all_sorare_teams = pd.concat([df["team"], df["next_game"]]).unique().tolist()
    team_map = build_team_map(all_sorare_teams, fbref_teams)
    df["fbref_team"] = df["team"].map(team_map)

    fbref_index = player_stats.index.to_frame(index=False)[["player", "team"]]

    def match_player(row, threshold: int = 85):
        if pd.isna(row["fbref_team"]):
            return None
        candidates = fbref_index.loc[fbref_index["team"] == row["fbref_team"], "player"].tolist()
        if not candidates:
            return None
        target_name = f"{row['first_name']} {row['last_name']}"
        best_match, score = process.extractOne(target_name, candidates)
        return best_match if score >= threshold else None

    df["fbref_name"] = df.apply(match_player, axis=1)
    df = df.dropna(subset=["fbref_name"])

    def extract_data(player_stats):
        cols_to_keep = [
            ('Per 90 Minutes', 'npxG+xAG'),
            ('Expected', 'npxG+xAG'),
            ('Playing Time', '90s'),
            ('Playing Time', 'Starts')
        ]
        stats_subset = player_stats.loc[:, cols_to_keep].reset_index()
        return stats_subset

    stats_subset = extract_data(player_stats)
    stats_subset.columns = ['league', 'season', 'team', 'player', 'npxG+xAG_p90', 'npxG+xAG', '90s', 'Starts']

    df = df.merge(
        stats_subset,
        left_on=["fbref_name", "fbref_team"],
        right_on=["player", "team"],
        how="left"
    ).drop(columns=["player", "team_y"])

    df.rename(columns={"team_x": "team"}, inplace=True)

    team_stats = fbref.read_team_season_stats(stat_type="standard", opponent_stats=True)
    team_stats_subset = team_stats.loc[:, [('Per 90 Minutes', 'npxG'), ('Playing Time', '90s')]].reset_index()
    team_stats_subset.columns = ['league', 'season', 'team', 'npxG_against_p90', 'team_90s']
    team_stats_subset['team'] = team_stats_subset['team'].str[3:]

    df["fbref_next_game_team"] = df["next_game"].map(team_map)
    team_stats_subset = calculate_avg_xG_conceded(team_stats_subset)

    df = df.merge(
        team_stats_subset,
        left_on="fbref_next_game_team",
        right_on="team",
        how="left"
    ).drop(columns=["team_y", "league_y", "season_y", "league_x", "season_x"])

    return df


def analyse_players(df):
    df["next_game_date"] = pd.to_datetime(df["next_game_date"], utc=True)
    now = pd.Timestamp.now(tz="UTC")
    today = now.normalize()
    tomorrow_cutoff = today + pd.Timedelta(days=1) + pd.Timedelta(hours=7)

    mask_games = (df["next_game_date"].dt.normalize() == today) | (
        (df["next_game_date"] > today) & (df["next_game_date"] <= tomorrow_cutoff)
    )

    mask_odds = (df["odds_reliability"].isna()) | (df["starter_odds_bp"] >= 6000)
    mask_playing_time = (df["90s"] >= 0.65 * df["team_90s"]) | (df["Starts"] >= 8)
    filtered = df[mask_games & mask_odds & mask_playing_time].copy()

    filtered["adjusted_xG"] = np.where(
        filtered["avg_xG_conceded_league"].notna() & (filtered["avg_xG_conceded_league"] != 0),
        filtered["npxG+xAG_p90"] * (filtered["npxG_against_p90"] / filtered["avg_xG_conceded_league"]),
        filtered["npxG+xAG_p90"]
    )

    filtered["prob_scoring"] = 1 - np.exp(-filtered["adjusted_xG"])
    filtered = filtered.drop(columns=["fbref_team", 'rarity', 'next_game_date', 'fbref_name', 'fbref_next_game_team'])
    filtered = filtered.sort_values(by="prob_scoring", ascending=False)

    new_names = ['First Name', 'Last Name', 'Team', 'Starting Likelihood (%)', 'Odds Reliability', 'Next Opponent']
    filtered.columns = new_names + list(filtered.columns[len(new_names):])
    return filtered


def main():
    print("=== Sorare Data Dashboard ===")
    EMAIL = input("Enter your Sorare email: ").strip()
    PASSWORD = input("Enter your Sorare password: ").strip()

    print("Signing in...")
    local_schema = load_local_schema("schema.graphql")
    salt = get_salt_for_email(EMAIL)
    hashed = hash_password(PASSWORD, salt)
    sign_in_client = make_client(local_schema=local_schema)
    sign_in_result = sign_in_with_password(EMAIL, hashed, AUD, sign_in_client)

    if sign_in_result.get("otpSessionChallenge"):
        print("Two factor authentication required.")
        otp = input("Enter your 2FA code: ").strip()
        sign_in_result = sign_in_with_otp(
            sign_in_result["otpSessionChallenge"], otp, AUD,
            make_client(local_schema=local_schema)
        )

    if sign_in_result.get("errors"):
        print(f"Sign in error: {sign_in_result['errors']}")
        return

    token = sign_in_result["jwtToken"]["token"]
    print("Fetching your cards and analysing data...")

    cards = fetch_owned_cards(token, AUD, local_schema, page_size=50)
    df = pd.DataFrame(cards)
    df = clean_data(df)
    df = get_fbref_stats(df)
    filtered_df = analyse_players(df)

    print("\n=== Top Players Analysis ===")
    print(filtered_df.to_string(index=False))


if __name__ == "__main__":
    main()
