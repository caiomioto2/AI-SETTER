from supabase import create_client, Client
from app.core.config import settings
import asyncio
from functools import partial

def get_platform_supabase() -> Client:
    """Returns a Supabase client connected to the platform database using the service role key."""
    return create_client(settings.platform_supabase_url, settings.platform_supabase_key)

def get_client_supabase(url: str, key: str) -> Client:
    """Returns a Supabase client connected to a specific client's database."""
    return create_client(url, key)

async def _run_async(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))

async def get_client_config(ghl_account_id: str):
    """Fetch client configuration from platform supabase based on GHL_Account_ID"""
    supabase = get_platform_supabase()
    def _get():
        return supabase.table("clients").select(
            "id, openrouter_api_key, llm_model, supabase_url, supabase_service_key, supabase_table_name, ghl_location_id"
        ).eq("ghl_location_id", ghl_account_id).execute()

    response = await _run_async(_get)

    if not response.data:
        raise ValueError(f"No client found for GHL_Account_ID: {ghl_account_id}")
    if len(response.data) > 1:
        raise ValueError(f"Multiple clients found for GHL_Account_ID: {ghl_account_id}")

    return response.data[0]

async def get_client_prompts(supabase_url: str, supabase_key: str):
    """Fetch text prompts from client's supabase."""
    client_supabase = get_client_supabase(supabase_url, supabase_key)
    def _get():
        return client_supabase.table("Text_Prompts").select("*").execute()

    response = await _run_async(_get)

    if not response.data:
        return {}

    prompts = {item.get("Prompt_Name"): item.get("Content") for item in response.data}
    return prompts

async def get_chat_history(supabase_url: str, supabase_key: str, session_id: str, limit: int = 20):
    """Fetch chat history for a lead from client's supabase Chat_History table."""
    client_supabase = get_client_supabase(supabase_url, supabase_key)

    try:
        def _get():
            return client_supabase.table("Chat_History").select("*").eq("sessionId", session_id).order("id", desc=True).limit(limit).execute()

        response = await _run_async(_get)
        return response.data
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception("Error fetching chat history")
        raise

async def save_chat_message(supabase_url: str, supabase_key: str, session_id: str, message: dict):
    """Save a message to the client's chat history"""
    client_supabase = get_client_supabase(supabase_url, supabase_key)
    try:
        def _insert():
            return client_supabase.table("Chat_History").insert({
                "sessionId": session_id,
                "message": message
            }).execute()
        response = await _run_async(_insert)
        return response.data
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception("Error saving chat message")
        raise
