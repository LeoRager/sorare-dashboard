import pandas as pd
import numpy as np
from sorare_backend import fetch_owned_cards, fetch_recent_sales


def get_price_history(jwt_token: str, aud: str, page_size: int = 50, total_limit: int = 150):
    cards = fetch_owned_cards(jwt_token, aud, page_size)
    df = pd.DataFrame(cards)
    df = df[df["rarity"].notna()]
    df = df[df["rarity"] != "common"]

    recent_sales = fetch_recent_sales(jwt_token, aud, df, batch_size=50, total_limit=total_limit)
    recent_sales_df = pd.DataFrame(recent_sales)

    if "rarity" in recent_sales_df.columns:
        recent_sales_df = recent_sales_df.drop(columns=["rarity"])
    if "seller_slug" in recent_sales_df.columns:
        recent_sales_df = recent_sales_df[recent_sales_df["seller_slug"].notna()]

    recent_sales_df = estimate_current_value(recent_sales_df)
    df = df.merge(recent_sales_df, on="player_slug", how="left")

    df["price_diff"] = df["estimated_value_eur"] - df["card_price"]
    df["Name"] = df["first_name"] + " " + df["last_name"]

    df = df[["Name", "rarity", "card_price", "estimated_value_eur", "price_diff"]]
    df = df.rename(columns={
        "Name": "Player",
        "rarity": "Rarity",
        "card_price": "Purchase Price (EUR)",
        "estimated_value_eur": "Current Value (EUR)",
        "price_diff": "P/L (EUR)"
    })
    df = df[[c for c in df.columns if c != "Rarity"] + ["Rarity"]]
    return df



def _get_price_history(jwt_token: str, aud: str, page_size: int = 50, total_limit: int = 150):
    cards = fetch_owned_cards(jwt_token, aud, page_size)
    df = pd.DataFrame(cards)

    # Drop any cards that are "common" or NaN rarity
    df = df[df["rarity"].notna()]
    df = df[df["rarity"] != "common"]

    # Fetch recent sales
    recent_sales = fetch_recent_sales(jwt_token, aud, df, total_limit=total_limit)
    recent_sales_df = pd.DataFrame(recent_sales)

    # Drop the rarity column
    recent_sales_df = recent_sales_df.drop(columns=["rarity"])

    # Drop any row where seller_slug is None
    recent_sales_df = recent_sales_df[recent_sales_df["seller_slug"].notna()]

    recent_sales_df = estimate_current_value(recent_sales_df)

    df = df.merge(recent_sales_df, on="player_slug", how="left")

    # Calculate the price difference
    df["price_diff"] = df["estimated_value_eur"] - df["card_price"]

    df["Name"] = df["first_name"] + " " + df["last_name"]

    # Keep only the following columns: Name, team, rarity, card_price, price_eur, date
    df = df[["Name", "rarity", "card_price", "estimated_value_eur", "price_diff"]]

    # Rename the columns for clarity
    new_names = {
        'Name': 'Player',
        'rarity': 'Rarity',
        'card_price': 'Purchase Price (EUR)',
        'estimated_value_eur': 'Current Value (EUR)',
        'price_diff': 'P/L (EUR)'
    }

    # Apply the renaming
    df = df.rename(columns=new_names)

    # Move rarity column to the end
    df = df[[c for c in df.columns if c != "Rarity"] + ["Rarity"]]

    return df


def estimate_current_value(df, lambda_=0.1):
    """
    Estimate the current value of each player's card based on sale history
    using a time-decayed weighted median.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: 'Name', 'price_eur', 'date'
    lambda_ : float, optional
        Decay rate for time weighting (default 0.1). Higher = faster decay.

    Returns
    -------
    pandas.DataFrame
        One row per player with their estimated current value.
    """

    # Ensure correct date type
    df['date'] = pd.to_datetime(df['date'])

    def weighted_median(data, weights):
        data, weights = np.array(data), np.array(weights)
        sorter = np.argsort(data)
        data, weights = data[sorter], weights[sorter]
        cumsum = np.cumsum(weights)
        cutoff = weights.sum() / 2
        return data[np.searchsorted(cumsum, cutoff)]

    results = []

    for name, group in df.groupby('player_slug', as_index=False):
        group = group.sort_values('date')
        latest_date = group['date'].max()
        group['days_ago'] = (latest_date - group['date']).dt.days
        group['weight'] = np.exp(-lambda_ * group['days_ago'])

        est_value = weighted_median(group['price_eur'], group['weight'])

        # Add the estimated value to the results list
        results.append({
            'player_slug': name,
            'estimated_value_eur': est_value,
        })

    result_df = pd.DataFrame(results)

    return result_df


def test_lambdas(sales_df: pd.DataFrame, lambdas_to_test: list) -> pd.DataFrame:
    """
    Runs estimate_current_value for multiple lambdas and combines results.

    Parameters
    ----------
    sales_df : pandas.DataFrame
        The cleaned DataFrame of recent sales.
        Must contain 'player_slug', 'price_eur', and 'date'.
    lambdas_to_test : list
        A list of float values to use as the lambda_ parameter.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per player_slug, and columns for
        each lambda's estimated value (e.g., 'est_value_lambda_0.1').
    """

    all_results = []

    for lam in lambdas_to_test:
        # We pass a .copy() because estimate_current_value modifies
        # the 'date' column, and we want to start fresh for each loop.
        est_df = estimate_current_value(sales_df.copy(), lambda_=lam)

        # Rename the result column to be unique for this lambda
        est_df = est_df.rename(columns={
            'estimated_value_eur': f'est_value_lambda_{lam}'
        })
        all_results.append(est_df)

    if not all_results:
        return pd.DataFrame()

    # Start with the first DataFrame
    merged_df = all_results[0]

    # Iteratively merge the results from all other lambdas
    for next_df in all_results[1:]:
        merged_df = merged_df.merge(next_df, on='player_slug', how='outer')

    return merged_df


# --- NEW EXAMPLE IMPLEMENTATION FUNCTION ---
def run_lambda_analysis(jwt_token: str, aud: str, page_size: int = 50, total_limit: int = 150,
                        lambdas_to_test: list = None):
    """
    Fetches all data and runs the lambda comparison test.

    This function mirrors your 'get_price_history' flow but calls
    'test_lambdas' instead of 'estimate_current_value'.
    """

    if lambdas_to_test is None:
        # Set some sensible defaults if none are provided
        lambdas_to_test = [0.01, 0.05, 0.1, 0.2, 0.5]

    print(f"Fetching owned cards...")
    cards = fetch_owned_cards(jwt_token, aud, page_size)
    df = pd.DataFrame(cards)

    # Drop any cards that are "common" or NaN rarity
    df = df[df["rarity"].notna()]
    df = df[df["rarity"] != "common"]

    if df.empty:
        print("No eligible owned cards found.")
        return pd.DataFrame()

    print(f"Fetching recent sales for {len(df)} cards...")
    recent_sales = fetch_recent_sales(jwt_token, aud, df, total_limit=total_limit)
    recent_sales_df = pd.DataFrame(recent_sales)

    if recent_sales_df.empty:
        print("No recent sales found.")
        return pd.DataFrame()

    # --- Prepare sales data for analysis ---
    # Drop the rarity column (not needed for price estimation)
    recent_sales_df = recent_sales_df.drop(columns=["rarity"], errors='ignore')
    # Drop any row where seller_slug is None (as in your original)
    recent_sales_df = recent_sales_df[recent_sales_df["seller_slug"].notna()]
    # CRITICAL: Drop rows where price is missing, as they can't be used
    recent_sales_df = recent_sales_df.dropna(subset=['price_eur'])

    if recent_sales_df.empty:
        print("No valid recent sales data after cleaning.")
        return pd.DataFrame()

    print(f"Running analysis for lambdas: {lambdas_to_test}...")
    # --- Run the lambda test ---
    lambda_results_df = test_lambdas(recent_sales_df, lambdas_to_test)

    # --- Merge results back onto your card list ---
    df["Name"] = df["first_name"] + " " + df["last_name"]

    # Merge the analysis results back into the main card DataFrame
    analysis_df = df.merge(lambda_results_df, on="player_slug", how="left")

    # --- Format the final output ---
    result_cols = [col for col in analysis_df.columns if col.startswith('est_value_lambda_')]

    # Keep the same columns as your original, but with all the new estimate cols
    final_cols = ["Name", "team", "rarity", "card_price"] + result_cols

    # Ensure we only return columns that actually exist
    final_cols = [col for col in final_cols if col in analysis_df.columns]

    return analysis_df[final_cols]