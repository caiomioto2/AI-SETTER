from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        platform_supabase_url: The URL of the platform Supabase instance.
        platform_supabase_key: The service role key for platform Supabase access.
    """
    platform_supabase_url: str
    platform_supabase_key: str  # service role key

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
