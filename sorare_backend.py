from gql import gql
from graphql import build_schema
from login import make_client
import pandas as pd

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
              inSeasonEligible
              ownershipHistory {
                amounts {
                  eurCents
                }
              }
              player {
                slug
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

            ownership_history = node.get("ownershipHistory")
            if ownership_history and isinstance(ownership_history, list):
                amounts = ownership_history[-1].get("amounts", [])
                # If amounts is None, the price paid is 0
                if amounts is not None:
                    card_price = amounts.get("eurCents")
                    card_price = card_price / 100.0
                else:
                    card_price = 0
            else:
                card_price = None

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
                "player_slug": player.get("slug"),
                "in_season_eligible": node.get("inSeasonEligible"),
                "card_price": card_price
            })

        page_info = cards_conn.get("pageInfo") or {}
        if not page_info.get("hasNextPage"):
            break
        after = page_info.get("endCursor")

    return cards


def fetch_recent_sales(jwt_token: str, aud: str, df: pd.DataFrame, batch_size: int = 50, total_limit: int = 150):
    local_schema = load_local_schema()
    client = make_client(jwt_token=jwt_token, aud=aud, local_schema=local_schema)

    query = gql("""
    query PlayerRecentTokenPrices(
      $slug: String!,
      $rarity: Rarity!,
      $before: String,  
      $last: Int!,
      $seasonEligibility: SeasonEligibility!
    ) {
      anyPlayer(slug: $slug) {
        slug
        tokenPrices(
          last: $last,
          before: $before,
          rarity: $rarity,
          includePrivateSales: true,
          seasonEligibility: $seasonEligibility
        ) {
          nodes {
            id
            amounts { eurCents }
            date
            card { slug inSeasonEligible }
            deal {
              ... on TokenOffer { userBuyer { slug } userSeller { slug } }
              ... on TokenPrimaryOffer { buyer { slug } userSeller { slug } }
              ... on TokenAuction { userBuyer { slug } userSeller { slug } }
            }
          }
          pageInfo {
            startCursor 
            hasPreviousPage
          }
        }
      }
    }
    """)

    last_sales = []

    for _, row in df.iterrows():
        slug = row["player_slug"]
        rarity = row["rarity"].lower().replace(" ", "_")
        season_eligibility = "IN_SEASON" if row.get("in_season_eligible") else "CLASSIC"

        before = None
        fetched = 0

        try:
            while fetched < total_limit:
                result = client.execute(
                    query,
                    variable_values={
                        "slug": slug,
                        "rarity": rarity,
                        "before": before,
                        "last": batch_size,
                        "seasonEligibility": season_eligibility,
                    },
                )

                player = result.get("anyPlayer")
                if not player:
                    break

                token_prices = player.get("tokenPrices", {})
                nodes = token_prices.get("nodes", []) or []

                for node in nodes:
                    amounts = node.get("amounts") or {}
                    eur_cents = amounts.get("eurCents")
                    eur = eur_cents / 100.0 if eur_cents else None

                    last_sales.append({
                        "player_slug": player.get("slug"),
                        "rarity": rarity,
                        "card_slug": (node.get("card") or {}).get("slug"),
                        "in_season_eligible": (node.get("card") or {}).get("inSeasonEligible"),
                        "price_eur": eur,
                        "buyer_slug": (
                                (node.get("deal") or {}).get("userBuyer")
                                or (node.get("deal") or {}).get("buyer")
                                or {}
                        ).get("slug"),
                        "seller_slug": (
                            ((node.get("deal") or {}).get("userSeller") or {}).get("slug")
                            if (node.get("deal") or {}).get("userSeller")
                            else None
                        ),
                        "date": node.get("date"),
                    })

                fetched += len(nodes)
                if fetched >= total_limit:
                    break

                page_info = token_prices.get("pageInfo") or {}

                if not page_info.get("hasPreviousPage"):
                    break

                before = page_info.get("startCursor")

        except Exception as e:
            print(f"Error fetching sales for {slug}: {e}")

    return last_sales
