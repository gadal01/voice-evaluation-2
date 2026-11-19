"""Checks the WebSocket JWT auth against a RUNNING server:   python test_auth.py [base_url]

    python -m uvicorn main:app --port 8000      # in another terminal
    python test_auth.py

Needs JWT_SECRET and AUTH_API_KEY in .env (it forges bad tokens with the secret to prove they are refused).
Good-token cases open a real Nova Sonic session for a moment, so they need AWS credentials too.
"""
import asyncio
import base64
import json
import sys
import time

import httpx
import jwt
import websockets
from websockets.exceptions import InvalidStatus

import auth

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
WS = BASE.replace("http", "ws", 1) + "/ws"
results: list[tuple[bool, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    results.append((ok, name))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))


def forge(secret=None, alg="HS256", **overrides) -> str:
    now = int(time.time())
    claims = {"iss": auth.ISSUER, "aud": auth.AUDIENCE, "sub": "attacker", "iat": now, "exp": now + 300, **overrides}
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, secret or auth._secret(), algorithm=alg)


def unsigned_token() -> str:  # alg "none": the classic JWT bypass
    enc = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    now = int(time.time())
    return f"{enc({'alg': 'none', 'typ': 'JWT'})}.{enc({'iss': auth.ISSUER, 'aud': auth.AUDIENCE, 'sub': 'x', 'iat': now, 'exp': now + 300})}."


async def refused(label: str, **kw) -> None:
    try:
        async with websockets.connect(WS, open_timeout=10, **kw) as ws:
            await asyncio.wait_for(ws.recv(), 3)
        check(False, label, "connection was accepted")
    except InvalidStatus as exc:
        check(exc.response.status_code == 403, label, f"HTTP {exc.response.status_code}")
    except Exception as exc:
        check(False, label, f"{type(exc).__name__}: {exc}")


async def accepted(label: str, **kw) -> None:
    try:
        async with websockets.connect(WS, open_timeout=10, **kw) as ws:
            msg = json.loads(await asyncio.wait_for(ws.recv(), 25))
            check(msg.get("type") == "ready", label, f"first message: {msg.get('type')}; subprotocol={ws.subprotocol}")
    except Exception as exc:
        check(False, label, f"{type(exc).__name__}: {exc}")


async def main() -> int:
    bearer = lambda t: {"additional_headers": {"Authorization": f"Bearer {t}"}}

    # --- token endpoint
    async with httpx.AsyncClient() as http:
        bad = await http.post(f"{BASE}/auth/token", headers={"X-API-Key": "wrong"})
        none = await http.post(f"{BASE}/auth/token")
        good = await http.post(f"{BASE}/auth/token", headers={"X-API-Key": auth.get_settings().auth_api_key})
    check(bad.status_code == 401, "/auth/token refuses a wrong API key", f"HTTP {bad.status_code}")
    check(none.status_code == 401, "/auth/token refuses a missing API key", f"HTTP {none.status_code}")
    check(good.status_code == 200 and good.json().get("token_type") == "bearer", "/auth/token issues a token for the right key")
    token = good.json()["access_token"]

    # --- sockets that must be refused
    await refused("no token")
    await refused("garbage token", **bearer("not.a.jwt"))
    await refused("wrong signing key", **bearer(forge(secret="x" * 40)))
    await refused("expired token", **bearer(forge(iat=int(time.time()) - 600, exp=int(time.time()) - 300)))
    await refused("wrong audience", **bearer(forge(aud="somewhere-else")))
    await refused("wrong issuer", **bearer(forge(iss="someone-else")))
    await refused("missing exp claim", **bearer(forge(exp=None)))
    await refused("alg=none (unsigned) token", **bearer(unsigned_token()))
    await refused("token in the wrong header scheme", additional_headers={"Authorization": f"Basic {token}"})
    await refused("subprotocol without a token", subprotocols=["bearer"])

    # --- sockets that must be accepted
    await accepted("valid token in Authorization header", **bearer(token))
    await accepted("valid token as 'bearer' subprotocol (browser style)", subprotocols=["bearer", token])

    failed = [n for ok, n in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
