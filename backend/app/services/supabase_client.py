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
    """Run a synchronous function in a thread pool executor.

    This helper converts synchronous Supabase calls to async by running
    them in a background thread.

    Args:
        func: The synchronous function to run.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the synchronous function.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))

async def get_client_config(ghl_account_id: str) -> dict:
    """Fetch client configuration from platform supabase based on GHL_Account_ID.

    Queries the platform database for client configuration including
    API keys, LLM model settings, and Supabase credentials.

    Args:
        ghl_account_id: The GoHighLevel location/account identifier.

    Returns:
        Dictionary containing client configuration (id, openrouter_api_key, llm_model, etc.).

    Raises:
        ValueError: If no client is found for the given GHL_Account_ID.
    """
    supabase = get_platform_supabase()
    def _get():
        return supabase.table("clients").select(
            "id, openrouter_api_key, llm_model, supabase_url, supabase_service_key, supabase_table_name, ghl_location_id"
        ).eq("ghl_location_id", ghl_account_id).execute()

    response = await _run_async(_get)

    if not response.data:
        raise ValueError(f"No client found for GHL_Account_ID: {ghl_account_id}")

    return response.data[0]

async def get_client_prompts(supabase_url: str, supabase_key: str) -> dict:
    """Fetch text prompts from client's Supabase database.

    Retrieves all prompts from the Text_Prompts table and returns
    them as a dictionary keyed by Prompt_Name.

    Args:
        supabase_url: The client's Supabase URL.
        supabase_key: The client's Supabase service role key.

    Returns:
        Dictionary mapping Prompt_Name to Content. Returns empty dict if no prompts found.
    """
    client_supabase = get_client_supabase(supabase_url, supabase_key)
    def _get():
        return client_supabase.table("Text_Prompts").select("*").execute()

    response = await _run_async(_get)

    if not response.data:
        return {}

    prompts = {item.get("Prompt_Name"): item.get("Content") for item in response.data}
    return prompts

async def get_chat_history(supabase_url: str, supabase_key: str, session_id: str, limit: int = 20) -> list:
    """Fetch chat history for a lead from client's Supabase Chat_History table.

    Retrieves the conversation history for a specific lead session,
    ordered chronologically.

    Args:
        supabase_url: The client's Supabase URL.
        supabase_key: The client's Supabase service role key.
        session_id: The lead's session identifier.
        limit: Maximum number of messages to retrieve (default: 20).

    Returns:
        List of chat history records. Returns empty list on error.
    """
    client_supabase = get_client_supabase(supabase_url, supabase_key)

    try:
        def _get():
            return client_supabase.table("Chat_History").select("*").eq("sessionId", session_id).order("id", desc=False).limit(limit).execute()

        response = await _run_async(_get)
        return response.data
    except Exception as e:
        print(f"Error fetching chat history: {e}")
        return []

async def save_chat_message(supabase_url: str, supabase_key: str, session_id: str, message: dict) -> list:
    """Save a message to the client's chat history.

    Inserts a new chat message into the Chat_History table for
    the specified lead session.

    Args:
        supabase_url: The client's Supabase URL.
        supabase_key: The client's Supabase service role key.
        session_id: The lead's session identifier.
        message: Dictionary with 'role' and 'content' keys representing the message.

    Returns:
        The inserted data on success, None on error.
    """
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
        print(f"Error saving chat message: {e}")
        return None
