import requests
import logging
import json

# --- Configuration ---

# Setup basic logging to print messages to the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# API endpoint from your curl command
API_URL = 'https://open.bigmodel.cn/api/paas/v4/chat/completions'

# Headers from your curl command (Authorization will be set dynamically)
BASE_HEADERS = {
    'Content-Type': 'application/json',
}

# Data payload from your curl command
PAYLOAD = {
    "model": "glm-4.1v-thinking-flash",  # Using a common model like glm-4v is sufficient for auth checks
    "messages": [
        {
            "role": "user",
            "content": "你好"  # A simple test prompt
        }
    ],
    "max_tokens": 10,
    "stream": False
}

# List of API tokens you provided for validation
API_TOKENS = [
    "23171b320b1a4a659c7e173413491e65.yuUlKqwuXQgSfORG",  # The one from your curl example
    "04acb67e3adc4b2db59649dc84bff13a.2dgL99awi8U8fdee",
    "0ebdd87e48bd4d10a5e6e2c83774a139.KU7lVwzzNdLsmGGW",
    "37d82140cca74282b18946db411bdf0c.qcqckWPWv7IqPvya",
    "3e8fe55bdccd4463a2587f36bd0d3088.gFyaidDn7r6lSYEa",
    "3efda81a698e43d9ab9c04778e4fba29.6ZrBX7cPleBfXRDH",
    "59c026d8a5e94ec7a9e2cb6761671716.BEJVrmMPzTQ5Rm7q",
    "5ce0f49fa47a4a8ba8bff5102e397baf.FlrpwHU5HCgQ44TY",
    "5ef99394c40844a180eacebac2b2d2ab.bHOSJoPE6Kc2UX2c",
    "78687c47e94e40ef8c0458db2f8c1e56.TB3HGPJ1B8mqBWL6",
    "906f83b37d774d5a9b34dcba628f453d.z9mwC55jACpRdJoR",
    "95c3aa5104f8412ca82b2116cd886f6f.yqz34xuReKpptP98",
    "a185a59c292d13a822ba24c21998ae15.Mc1QsSDVhL6q4dlS",
    "a72a743ec100470f8e0e8e39e5557b01.li8PBRBxXX64C45x",
    "a95e0b504a124e7a9a6176e0b6ed538e.9iveIF82cV9PXO93",
    "c978708e4f92475ebefe02ecfdf0d031.yARoEC9rphOvfYhX",
    "cbc6557dc1984c24b05d53ffe1632c96.5QbcuYMfwR6DlGkN",
    "ccd5c0f0124f4702b6712fe07f601175.rawG2E4OtCDCKkqH",
    "f1a6717168e3443188752f8435a13555.wD4jlEKS0oQHd9hH",
    "f6989bfa53f04c78bfc3d25bdd7617b6.QbBds6GVEnTkEI7K",
    "fe6ad8627c7347d383a621369e8738ca.RRnt8TFZmc6YiNxw"
]


def validate_token(token: str) -> bool:
    """
    Validates a single API token by making a request to the BigModel API.
    Returns True if the token is valid, False otherwise.
    """
    # To avoid showing the full token in logs, let's create a masked version
    masked_token = f"{token[:4]}...{token[-4:]}"
    logging.info(f"--- Checking token: {masked_token} ---")

    # Prepare headers for this specific request
    headers = BASE_HEADERS.copy()
    headers['Authorization'] = f'Bearer {token}'

    try:
        # Make the POST request
        response = requests.post(API_URL, headers=headers, data=json.dumps(PAYLOAD), timeout=15)

        # Check the response status code
        if response.status_code == 200:
            logging.info(f"✅ SUCCESS: Token {masked_token} is VALID.")
            return True
        else:
            # Log an error if the status code is not 200
            logging.error(
                f"❌ FAILED: Token {masked_token} is INVALID or the API returned an error."
            )
            logging.error(f"Status Code: {response.status_code}")
            # Try to print the error message from the API response
            try:
                logging.error(f"Response Body: {response.json()}")
            except json.JSONDecodeError:
                logging.error(f"Response Body: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        # Handle network-related errors (e.g., timeout, connection error)
        logging.critical(f"CRITICAL: A network error occurred while checking token {masked_token}. Error: {e}")
        return False

    finally:
        logging.info("-" * (len(masked_token) + 24))


if __name__ == '__main__':
    successful_tokens = []
    failed_tokens = []

    logging.info(f"Starting validation for {len(API_TOKENS)} tokens.")
    for api_token in API_TOKENS:
        if validate_token(api_token):
            successful_tokens.append(api_token)
        else:
            failed_tokens.append(api_token)

    logging.info("All tokens have been checked.")

    # --- Summary Section ---
    print("\n" + "=" * 60)
    print(" " * 20 + "VALIDATION SUMMARY")
    print("=" * 60)

    print(f"\n✅ Successful Tokens ({len(successful_tokens)}):")
    if successful_tokens:
        for token in successful_tokens:
            print(token)
    else:
        print("None")

    print("\n" + "-" * 60)

    print(f"\n❌ Failed Tokens ({len(failed_tokens)}):")
    if failed_tokens:
        for token in failed_tokens:
            print(token)
    else:
        print("None")

    print("\n" + "=" * 60)
