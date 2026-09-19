"""
Shared test configuration.
"""

import os

# api_football validates the API key at import time, so tests need a
# placeholder key present even though every API call is mocked.
os.environ.setdefault("API_FOOTBALL_KEY", "test_api_key")