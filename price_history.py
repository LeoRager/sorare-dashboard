import pandas as pd
from sorare_backend import fetch_owned_cards

def get_price_history(jwt_token: str, aud: str, page_size: int = 50):
    cards = fetch_owned_cards(jwt_token, aud, page_size)
    df = pd.DataFrame(cards)

    # Drop any cards that are "common" or NaN rarity
    df = df[df["rarity"].notna()]
    df = df[df["rarity"] != "common"]
    return df
