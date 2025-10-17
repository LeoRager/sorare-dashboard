from gql import gql
from graphql import build_schema
from login import make_client

def load_local_schema(path: str = "schema.graphql"):
    with open(path, "r", encoding="utf-8") as fh:
        sdl = fh.read()
    return build_schema(sdl)


def fetch_owned_cards(jwt_token: str, aud: str, page_size: int = 50):
    local_schema = load_local_schema()
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