import streamlit as st
from debug_dashboard import *

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

# Step 3: Fetch and display data
if st.session_state.step == 3 and st.session_state.token:
    st.success("Signed in successfully.")

    st.subheader("Adjust Player Filters")

    # Interactive sliders / inputs for analyse_players parameters
    min_nineties_ratio = st.slider(
        "Available Minutes Played (%)",
        0.0, 100.0, 65.0, 1.0
    )
    min_starts = st.number_input(
        "Minimum number of starts required",
        min_value=0, max_value=50, value=8, step=1
    )
    min_starter_odds = st.slider(
        "Likelihood of Starting (%)",
        min_value=0, max_value=100, value=60, step=5
    )

    # Dropdown menu for rarities
    rarity_options = ["Common", "Limited", "Rare", "Super Rare", "Unique"]
    selected_rarities = st.multiselect(
        "Select Card Rarities",
        options=rarity_options,
        default=rarity_options  # Select all by default
    )

    # Convert selections to lowercase and adjust 'Super Rare' to 'superRare'
    rarities = [
        "superRare" if r == "Super Rare" else r.lower()
        for r in selected_rarities
    ]

    if st.button("Fetch and Analyse My Cards"):
        with st.spinner("Fetching your cards and analysing data..."):
            token = st.session_state.token
            local_schema = load_local_schema("schema.graphql")
            cards = fetch_owned_cards(token, AUD, local_schema, page_size=50)

            df = pd.DataFrame(cards)
            df = clean_data(df)
            df = get_fbref_stats(df)
            filtered_df = analyse_players(
                df,
                min_nineties_ratio=min_nineties_ratio,
                min_starts=min_starts,
                min_starter_odds=min_starter_odds,
                rarities=rarities
            )

        st.success("Data loaded successfully!")
        st.dataframe(filtered_df)
