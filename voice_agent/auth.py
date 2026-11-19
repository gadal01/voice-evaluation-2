"""JWT authentication for the voice WebSocket.

Flow:
  1. A client calls  POST /auth/token  with the header  X-API-Key: <AUTH_API_KEY>
     and gets back a short-lived signed JWT.
  2. The client opens  /ws  presenting that JWT in one of two ways:
       - header     Authorization: Bearer <jwt>                    (scripts, servers)
       - subprotocol  Sec-WebSocket-Protocol: bearer, <jwt>        (browsers: they cannot set headers,
                                                                    and a query string would end up in logs)
  3. The server verifies the token BEFORE accepting the socket; otherwise the handshake is refused (HTTP 403).

Config (.env):
  JWT_SECRET     signing key, at least 32 characters. The server refuses every connection without it.
  AUTH_API_KEY   the secret a client must present to obtain a token.
  JWT_TTL_SECONDS  token lifetime, default 300.

A token is only checked when the socket opens; a session that outlives the token keeps running.
"""
import hmac
import time
import uuid

import jwt
from starlette.websockets import WebSocket

from agent_settings import get_settings


ISSUER = "voice-agent"
AUDIENCE = "voice-ws"
ALGORITHM = "HS256"  # pinned: never let the token choose its own algorithm
SUBPROTOCOL = "bearer"
MIN_SECRET_LEN = 32


class AuthError(Exception):
    """Raised for any reason a client is not allowed in. The message is for server logs only."""


def _secret() -> str:
    secret = get_settings().jwt_secret
    if len(secret) < MIN_SECRET_LEN:
        raise AuthError(f"JWT_SECRET is missing or shorter than {MIN_SECRET_LEN} characters")
    return secret


def ttl_seconds() -> int:
    return get_settings().jwt_ttl_seconds


def check_api_key(candidate: str | None) -> bool:
    expected = get_settings().auth_api_key
    return bool(expected) and bool(candidate) and hmac.compare_digest(candidate.encode(), expected.encode())


def issue_token(subject: str = "voice-client") -> tuple[str, int]:
    """Returns (jwt, seconds until it expires)."""
    now = int(time.time())
    ttl = ttl_seconds()
    claims = {"iss": ISSUER, "aud": AUDIENCE, "sub": subject, "iat": now, "exp": now + ttl, "jti": uuid.uuid4().hex}
    return jwt.encode(claims, _secret(), algorithm=ALGORITHM), ttl


def verify_token(token: str) -> dict:
    try:
        return jwt.decode(
            token, _secret(), algorithms=[ALGORITHM], issuer=ISSUER, audience=AUDIENCE,
            options={"require": ["exp", "iat", "iss", "aud", "sub"]},
        )
    except jwt.InvalidTokenError as exc:
        raise AuthError(f"invalid token: {type(exc).__name__}: {exc}") from exc


def token_from_websocket(ws: WebSocket) -> tuple[str | None, bool]:
    """Finds the token on a handshake. Returns (token, offered_via_subprotocol)."""
    header = ws.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() == "bearer" and value.strip():
        return value.strip(), False
    offered = ws.scope.get("subprotocols") or []
    if len(offered) >= 2 and offered[0] == SUBPROTOCOL:
        return offered[1], True
    return None, False


def authenticate(ws: WebSocket) -> tuple[dict, str | None]:
    """Verifies a WebSocket handshake. Returns (claims, subprotocol to echo back, if any)."""
    token, via_subprotocol = token_from_websocket(ws)
    if not token:
        raise AuthError("no token presented")
    return verify_token(token), (SUBPROTOCOL if via_subprotocol else None)
