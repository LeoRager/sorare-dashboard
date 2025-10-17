import bcrypt
import requests
from gql.transport.requests import RequestsHTTPTransport
from typing import Optional
from gql import gql, Client

AUD = "choose_decisive_player"
GRAPHQL_ENDPOINT = "https://api.sorare.com/graphql"

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