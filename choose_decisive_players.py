import streamlit as st

from debug_dashboard import *
from login import *
from price_history import get_price_history

AUD = "choose_decisive_player"

st.title("Sorare Data Dashboard")

# Initialise session state
if "step" not in st.session_state:
    st.session_state.step = 1
if "sign_in_result" not in st.session_state:
    st.session_state.sign_in_result = None
if "token" not in st.session_state:
    st.session_state.token = None
if "EMAIL" not in st.session_state:
    st.session_state.EMAIL = ""
if "PASSWORD" not in st.session_state:
    st.session_state.PASSWORD = ""


# Callback for sign in
def handle_sign_in():
    EMAIL = st.session_state.EMAIL
    PASSWORD = st.session_state.PASSWORD

    with st.spinner("Signing in..."):
        local_schema = load_local_schema("schema.graphql")
        salt = get_salt_for_email(EMAIL)
        hashed = hash_password(PASSWORD, salt)
        sign_in_client = make_client(local_schema=local_schema)
        sign_in_result = sign_in_with_password(EMAIL, hashed, AUD, sign_in_client)
        st.session_state.sign_in_result = sign_in_result

    if sign_in_result.get("otpSessionChallenge"):
        st.session_state.step = 2
    elif sign_in_result.get("errors"):
        st.error(f"Sign in errors: {sign_in_result['errors']}")
    else:
        st.session_state.token = sign_in_result["jwtToken"]["token"]
        st.session_state.step = 3


# Step 1: Email and password form
if st.session_state.step == 1:
    with st.form("login_form"):
        st.text_input("Enter your email", key="EMAIL")
        st.text_input("Enter your password", type="password", key="PASSWORD")
        st.form_submit_button("Submit", on_click=handle_sign_in)

# Step 2: OTP input
elif st.session_state.step == 2:
    st.info("Two factor authentication is enabled for this account.")
    with st.form("otp_form"):
        otp = st.text_input("Enter your 2FA code", type="password")
        otp_submitted = st.form_submit_button("Submit OTP")

    if otp_submitted:
        with st.spinner("Verifying OTP..."):
            challenge = st.session_state.sign_in_result["otpSessionChallenge"]
            sign_in_result = sign_in_with_otp(
                challenge,
                otp,
                AUD,
                make_client(local_schema=load_local_schema("schema.graphql"))
            )
            st.session_state.sign_in_result = sign_in_result

        if sign_in_result.get("errors"):
            st.error(f"OTP errors: {sign_in_result['errors']}")
        else:
            st.session_state.token = sign_in_result["jwtToken"]["token"]
            st.session_state.step = 3

# Step 3: Tools
if st.session_state.step == 3 and st.session_state.token:
    st.success("Signed in successfully.")

    tab_decisive, tab_value = st.tabs(["Choose decisive players", "Estimate club value"])

    with tab_decisive:
        st.subheader("Adjust player filters")

        # Choose data provider
        provider = st.radio(
            "Data source",
            options=["FBref", "Understat"],
            horizontal=True
        )

        min_nineties_ratio = st.slider(
            "Available Minutes Played percent",
            0.0, 100.0, 65.0, 1.0
        )
        min_starts = st.number_input(
            "Minimum number of starts required",
            min_value=0, max_value=50, value=8, step=1
        )
        min_starter_odds = st.slider(
            "Likelihood of starting percent",
            min_value=0, max_value=100, value=60, step=5
        )

        rarity_options = ["Common", "Limited", "Rare", "Super Rare", "Unique"]
        selected_rarities = st.multiselect(
            "Select card rarities",
            options=rarity_options,
            default=rarity_options
        )

        rarities = [
            "superRare" if r == "Super Rare" else r.lower()
            for r in selected_rarities
        ]

        selected_date = st.date_input(
            "Select game date",
            value=pd.Timestamp.now().date(),
            min_value=pd.Timestamp.now().date(),
            max_value=(pd.Timestamp.now() + pd.Timedelta(days=5)).date()
        )
        date_threshold = pd.Timestamp(selected_date).tz_localize("UTC").normalize()

        if st.button("Fetch and analyse my cards"):
            with st.spinner("Fetching your cards and analysing data..."):
                token = st.session_state.token
                cards = fetch_owned_cards(token, AUD, page_size=50)

                df = pd.DataFrame(cards)
                df = clean_data(df)

                if provider == "Understat":
                    df = get_understat_stats(df)
                else:
                    df = get_fbref_stats(df)

                filtered_df = analyse_players(
                    df,
                    min_nineties_ratio=min_nineties_ratio / 100,
                    min_starts=min_starts,
                    min_starter_odds=min_starter_odds,
                    rarities=rarities,
                    date_threshold=date_threshold
                )

            st.success("Data loaded successfully.")
            st.dataframe(filtered_df, width="stretch")

    with tab_value:
        st.subheader("Club value estimator")

        # The only input for get_price_history
        total_limit = st.slider(
            "Total cards fetched per player",
            min_value=50,
            max_value=1000,
            value=300,
            step=50
        )

        def colour_pl(val):
            try:
                x = float(val)
            except Exception:
                return ""
            if x > 0:
                return "color: green;"
            if x < 0:
                return "color: red;"
            return "color: grey;"


        if st.button("Estimate my club value"):
            with st.spinner("Fetching price history and estimating value..."):
                token = st.session_state.token

                # This returns a DataFrame with columns:
                # Player, Team, Rarity, Purchase Price (EUR), Current Value (EUR), P/L (EUR)
                df_prices = get_price_history(
                    jwt_token=token,
                    aud=AUD,
                    page_size=50,
                    total_limit=total_limit,
                )

                # Ensure numeric for formatting and styling
                for col in ["Purchase Price (EUR)", "Current Value (EUR)", "P/L (EUR)"]:
                    if col in df_prices.columns:
                        df_prices[col] = pd.to_numeric(df_prices[col], errors="coerce")

                total_spent = float(np.nansum(df_prices.get("Purchase Price (EUR)", pd.Series(dtype=float)).values))
                total_value = float(np.nansum(df_prices.get("Current Value (EUR)", pd.Series(dtype=float)).values))
                total_pl = float(np.nansum(df_prices.get("P/L (EUR)", pd.Series(dtype=float)).values))

            # Create three columns for metrics
            col1, col2, col3 = st.columns(3)

            col1.metric(
                label="Total Spent",
                value=f"€{total_spent:,.2f}"
            )
            col2.metric(
                label="Estimated Club Value",
                value=f"€{total_value:,.2f}"
            )
            col3.metric(
                label="Total P/L",
                value=f"€{total_pl:+,.2f}",
                delta=None
            )

            styled = (
                df_prices.style
                .format({
                    "Purchase Price (EUR)": "€{:,.2f}",
                    "Current Value (EUR)": "€{:,.2f}",
                    "P/L (EUR)": "€{:+,.2f}",
                })
                .map(colour_pl, subset=["P/L (EUR)"])
            )

            st.dataframe(styled, width='stretch')

