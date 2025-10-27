import time
from collections import defaultdict
from fastapi import HTTPException
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

# Simple in-memory rate limiter
# Format: {api_key: [(timestamp, timestamp, ...)]}
_rate_limit_store = defaultdict(list)

# Rate limit: 10 requests per minute per API key
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW = 60  # seconds


async def rate_limit(api_key: str):
    """
    Check if the API key has exceeded the rate limit.
    Raises HTTPException if limit exceeded.
    """
    now = time.time()
    
    # Get request timestamps for this API key
    timestamps = _rate_limit_store[api_key]
    
    # Remove timestamps older than the window
    timestamps = [ts for ts in timestamps if now - ts < RATE_LIMIT_WINDOW]
    
    # Check if limit exceeded
    if len(timestamps) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds."
        )
    
    # Add current timestamp
    timestamps.append(now)
    _rate_limit_store[api_key] = timestamps