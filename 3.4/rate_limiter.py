import time
from collections import defaultdict, deque

# naive sliding-window limiter: allow N requests per WINDOW seconds per key
WINDOW = 60
MAX_REQUESTS = 30

_store = defaultdict(lambda: deque())

async def rate_limit(api_key: str):
    now = time.time()
    q = _store[api_keys]
    # pop old
    while q and q[0] <= now - WINDOW:
        q.popleft()
    if len(q) >= MAX_REQUESTS:
        raise Exception("Rate limit exceeded")
    q.append(now)
