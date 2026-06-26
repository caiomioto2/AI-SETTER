from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    platform_supabase_url: str
    platform_supabase_key: str  # service role key

    model_config = SettingsConfigDict(env_file=Path(__file__).parent.parent.parent / ".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
