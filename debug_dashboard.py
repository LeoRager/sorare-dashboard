import pandas as pd
import numpy as np
import soccerdata as sd
from thefuzz import process
from login import *
from sorare_backend import fetch_owned_cards, load_local_schema
from price_history import get_price_history, run_lambda_analysis


def clean_data(df):
    # Only keep the following column: first_name, last_name, team, rarity, starter_odds_bp, odds_reliability,
    # next_game_date, next_game_home, next_game_away
    df = df[["first_name", "last_name", "team", "rarity", "starter_odds_bp", "odds_reliability", "next_game_date",
             "next_game_home", "next_game_away"]]
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


def build_team_map(sorare_teams, fbref_teams, data_provider, threshold: int = 80) -> dict:
    hardcoded_fbref = {
        "RCD Espanyol de Barcelona": "Espanyol",
        "Stade Rennais F.C.": "Rennes",
        "Real Valladolid CF": "Valladolid",
        "Paris Saint-Germain": "Paris S-G",
        "Romania": np.nan
    }

    hardcoded_understat = {
        "1. FC Köln": "FC Cologne",
        "D. Alavés": "Alaves",
        "Stade Rennais F.C.": "Rennes",
    }

    # Choose the correct hardcoded dictionary based on the provider
    if data_provider == "fbref":
        hardcoded = hardcoded_fbref
    elif data_provider == "understat":
        hardcoded = hardcoded_understat
    else:
        hardcoded = {}

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


def match_player(row, data_index, threshold: int = 85):
    if pd.isna(row["data_team"]):
        return None
    candidates = data_index.loc[data_index["team"] == row["data_team"], "player"].tolist()
    if not candidates:
        return None
    target_name = f"{row['first_name']} {row['last_name']}"
    best_match, score = process.extractOne(target_name, candidates)
    return best_match if score >= threshold else None


def get_fbref_stats(df: pd.DataFrame, season="2025-26"):
    fbref = sd.FBref(leagues=["Big 5 European Leagues Combined"], seasons=[season])
    player_stats = fbref.read_player_season_stats(stat_type="standard")
    fbref_teams = player_stats.index.get_level_values("team").unique().tolist()

    all_sorare_teams = pd.concat([df["team"], df["next_game"]]).unique().tolist()
    team_map = build_team_map(all_sorare_teams, fbref_teams, data_provider="fbref")
    df["data_team"] = df["team"].map(team_map)

    fbref_index = player_stats.index.to_frame(index=False)[["player", "team"]]

    df["data_name"] = df.apply(match_player, axis=1, data_index=fbref_index)
    df = df.dropna(subset=["data_name"])

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
        left_on=["data_name", "data_team"],
        right_on=["player", "team"],
        how="left"
    ).drop(columns=["player", "team_y"])

    df.rename(columns={"team_x": "team"}, inplace=True)

    vs_team_stats = fbref.read_team_season_stats(stat_type="standard", opponent_stats=True)
    vs_team_stats_subset = vs_team_stats.loc[:, [('Per 90 Minutes', 'npxG')]].reset_index()
    vs_team_stats_subset.columns = ['league', 'season', 'team', 'npxG_against_p90']
    vs_team_stats_subset['team'] = vs_team_stats_subset['team'].str[3:]

    df["data_next_game_team"] = df["next_game"].map(team_map)
    vs_team_stats_subset = calculate_avg_xG_conceded(vs_team_stats_subset)

    # Get team_90s
    team_stats = fbref.read_team_season_stats(stat_type="standard")
    team_stats_subset = team_stats.loc[:, [('Playing Time', '90s')]].reset_index()
    team_stats_subset.columns = ['league', 'season', 'team', 'team_90s']

    df = df.merge(
        vs_team_stats_subset,
        left_on="data_next_game_team",
        right_on="team",
        how="left"
    ).drop(columns=["team_y", "league_y", "season_y", "league_x", "season_x"])

    df = df.merge(
        team_stats_subset,
        left_on="data_team",
        right_on="team",
        how="left"
    ).drop(columns=["team", "league", "season"])

    df.rename(columns={"team_x": "team"}, inplace=True)

    return df


def get_understat_stats(
    df: pd.DataFrame,
    season="2025-26",
    leagues=None,
):
    """
    Mirror of get_fbref_stats but powered by Understat through soccerdata.
    Returns a dataframe that keeps the same columns expected by analyse_players.
    """
    # Defaults to the big five leagues if none were provided
    if leagues is None:
        leagues = [
            "ENG-Premier League",
            "ESP-La Liga",
            "GER-Bundesliga",
            "ITA-Serie A",
            "FRA-Ligue 1",
        ]

    understat = sd.Understat(leagues=leagues, seasons=[season])
    # Player season level stats
    player_stats = understat.read_player_season_stats().reset_index()

    # Build a mapping between Sorare team names and Understat team names
    understat_teams = player_stats["team"].dropna().unique().tolist()
    all_sorare_teams = pd.concat([df["team"], df["next_game"]]).dropna().unique().tolist()
    team_map = build_team_map(all_sorare_teams, understat_teams, data_provider="understat")
    df["data_team"] = df["team"].map(team_map)

    us_index = player_stats[['team', 'player']]
    df["data_name"] = df.apply(match_player, axis=1, data_index=us_index)
    df = df.dropna(subset=["data_name"])

    stats_subset = player_stats[
        ["league", "season", "team", "player", "np_xg", "xa", "minutes", "matches"]
    ].copy()

    stats_subset["npxG+xAG"] = stats_subset["np_xg"].fillna(0) + stats_subset["xa"].fillna(0)
    stats_subset["90s"] = stats_subset["minutes"].fillna(0) / 90.0
    stats_subset["npxG+xAG_p90"] = np.where(
        stats_subset["minutes"].fillna(0) > 0,
        stats_subset["npxG+xAG"] / (stats_subset["minutes"] / 90.0),
        0.0,
    )
    stats_subset["Starts"] = 0

    # Merge player season stats on name and team
    df = df.merge(
        stats_subset,
        left_on=["data_name", "data_team"],
        right_on=["player", "team"],
        how="left",
    ).drop(columns=["player", "team_y"])
    df.rename(columns={"team_x": "team"}, inplace=True)

    team_stats = understat.read_team_match_stats().reset_index()

    # Aggregate directly by team to avoid building a full long-form table
    home_conceded = (
        team_stats.groupby(["league", "season", "home_team"], as_index=False)
        .agg(conceded_np_xg=("away_np_xg", "median"), team_90s=("away_np_xg", "size"))
        .rename(columns={"home_team": "team"})
    )

    away_conceded = (
        team_stats.groupby(["league", "season", "away_team"], as_index=False)
        .agg(conceded_np_xg=("home_np_xg", "median"), team_90s=("home_np_xg", "size"))
        .rename(columns={"away_team": "team"})
    )

    team_conceded = pd.concat([home_conceded, away_conceded], ignore_index=True)

    team_conceded = (
        team_conceded.groupby(["league", "season", "team"], as_index=False)
        .agg(
            npxG_against_p90=("conceded_np_xg", "median"),
            team_90s=("team_90s", "sum")
        )
    )

    # Add league average conceded for scaling
    conceded_for_merge = calculate_avg_xG_conceded(team_conceded[["league", "season", "team", "npxG_against_p90"]])

    # Map the next opponent to Understat team names and merge opponent strength
    df["data_next_game_team"] = df["next_game"].map(team_map)

    df = df.merge(
        conceded_for_merge,
        left_on="data_next_game_team",
        right_on="team",
        how="left",
    ).drop(columns=["team_y", "league_y", "season_y", "league_x", "season_x"])

    # Merge team 90s for playing time filter
    df = df.merge(
        team_conceded[["league", "season", "team", "team_90s"]],
        left_on="data_team",
        right_on="team",
        how="left",
    ).drop(columns=["team", "league", "season"])

    df.rename(columns={"team_x": "team"}, inplace=True)

    # Drop the following columns: np_xg, xa, minutes, matches
    df = df.drop(columns=["np_xg", "xa", "minutes", "matches"])

    # move npxg+xag_p90 to the 11th column
    cols = df.columns.tolist()

    # Remove and reinsert at position 10 (11th place)
    cols.insert(10, cols.pop(cols.index("npxG+xAG_p90")))

    df = df[cols]

    return df


def analyse_players(
    df,
    min_nineties_ratio: float = 0.65,
    min_starts: int = 8,
    min_starter_odds: int = 60,
    rarities: list = ["common", "limited", "rare", "superRare", "unique"],
    date_threshold: pd.Timestamp = pd.Timestamp.now(tz="UTC")
):
    df = df[df["rarity"].isin(rarities)]
    df["next_game_date"] = pd.to_datetime(df["next_game_date"], utc=True)
    date_threshold = date_threshold.normalize()
    tomorrow_cutoff = date_threshold + pd.Timedelta(days=1) + pd.Timedelta(hours=7)

    # Select games played date_threshold or before tomorrow morning (7am)
    mask_games = (df["next_game_date"].dt.normalize() == date_threshold) | (
            (df["next_game_date"] > date_threshold) & (df["next_game_date"] <= tomorrow_cutoff)
    )

    # Starter odds filter
    mask_odds = (df["odds_reliability"].isna()) | (df["starter_odds_bp"] >= min_starter_odds*100)

    # Playing time filter
    mask_playing_time = (df["90s"] >= min_nineties_ratio * df["team_90s"]) | (df["Starts"] >= min_starts)

    # Apply combined filters
    filtered = df[mask_games & mask_odds & mask_playing_time].copy()

    # Adjust xG using opponent defensive strength
    filtered["adjusted_xG"] = np.where(
        filtered["avg_xG_conceded_league"].notna() & (filtered["avg_xG_conceded_league"] != 0),
        filtered["npxG+xAG_p90"] * (filtered["npxG_against_p90"] / filtered["avg_xG_conceded_league"]),
        filtered["npxG+xAG_p90"]
    )

    # Poisson probability of scoring at least once
    filtered["prob_scoring"] = 1 - np.exp(-filtered["adjusted_xG"])

    # Drop unnecessary columns
    drop_cols = ["data_team", "rarity", "next_game_date", "data_name", "data_next_game_team",
                 "avg_xG_conceded_league", "team_90s", "adjusted_xG", "odds_reliability"]
    filtered = filtered.drop(columns=[c for c in drop_cols if c in filtered.columns])

    # Sort by scoring probability
    filtered = filtered.sort_values(by="prob_scoring", ascending=False)

    # Rename visible columns
    new_names = [
        "First Name", "Last Name", "Team", "Starting Likelihood (%)", "Next Opponent",
        "npxG+xAG p90", "npxG+xAG", "90s", "Starts", "npxG p90 Conceded", "Scoring Probability (%)"
    ]
    filtered.columns = new_names

    # Combine First and Last name column to "Name"
    filtered["Name"] = filtered["First Name"] + " " + filtered["Last Name"]

    # Drop First and Last name columns
    filtered = filtered.drop(columns=["First Name", "Last Name"])

    # Divide Starting Likelihood by 100
    filtered["Starting Likelihood (%)"] = filtered["Starting Likelihood (%)"] / 100

    # Reorder the columns: "Name", "Team", "Next Opponent", "Starting Likelihood", "Scoring Probability", "90s", "Starts"
    # "npxG+xAG p90", "npxG+xAG", "npxG p90 Conceded"
    filtered = filtered[
        ["Name", "Team", "Next Opponent", "Starting Likelihood (%)", "Scoring Probability (%)", "90s", "Starts",
         "npxG+xAG p90", "npxG+xAG", "npxG p90 Conceded"]
    ]


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

    cards = fetch_owned_cards(token, AUD, page_size=50)
    df = pd.DataFrame(cards)
    df = clean_data(df)
    df = get_understat_stats(df)
    filtered_df = analyse_players(df)
    #
    # print("\n=== Top Players Analysis ===")
    # print(filtered_df.to_string(index=False))

    # cards = run_lambda_analysis(token, AUD, total_limit=300)
    # print(cards.head)
    # return cards

if __name__ == "__main__":
    main()
