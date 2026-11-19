"""The voice agent's configuration, read once and typed.

Real environment variables win over the nearest `.env`. Unknown variables in `.env` are ignored (it is shared with the
evaluation harness). A bad value (for example a non-numeric PORT) stops the server at startup with a clear message.

    JWT_SECRET, AUTH_API_KEY, JWT_TTL_SECONDS      WebSocket authentication (see auth.py)
    AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN      credentials for Bedrock
    NOVA_SONIC_MODEL_ID, NOVA_SONIC_VOICE, NOVA_SONIC_PROMPT                      the speech model
    MCP_SERVER_URL, PORT                                                           tools and the listening port
"""
from functools import lru_cache

from dotenv import find_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    jwt_secret: str = ""                                   # checked where it is used (auth.py), so the server can still start and say why
    auth_api_key: str = ""
    jwt_ttl_seconds: int = Field(300, ge=30, le=3600)

    aws_region: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None

    nova_sonic_model_id: str = "amazon.nova-2-sonic-v1:0"
    nova_sonic_voice: str = "matthew"                     # matthew, tiffany, amy, ...
    nova_sonic_prompt: str | None = None                  # overrides the built-in system prompt

    mcp_server_url: str | None = None                     # default: the sample service mounted in this app
    port: int = Field(8000, ge=1, le=65535)


@lru_cache
def get_settings() -> AgentSettings:
    """Loaded once per process. In tests, call get_settings.cache_clear() after changing the environment."""
    return AgentSettings(_env_file=find_dotenv() or None)
