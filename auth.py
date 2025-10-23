from fastapi import Header, HTTPException
from starlette.status import HTTP_401_UNAUTHORIZED
import os

# Set an API key via environment variable for production
API_KEYS = set()
key = os.environ.get("API_KEY")
if key:
    API_KEYS.add(key)
# For convenience during local dev, add a default dev key
API_KEYS.add("dev-key-123")


def get_api_key_header(x_api_key: str = Header(None)):
    if x_api_key is None:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Missing X-API-KEY header")
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    return x_api_key


def verify_api_key(key: str) -> bool:
    return key in API_KEYS
