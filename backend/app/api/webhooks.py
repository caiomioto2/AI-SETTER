from fastapi import APIRouter, Depends, HTTPException, Query
from app.models.webhook import DMWebhookPayload, AgentResponse
from app.services import get_client_config, get_client_prompts, get_chat_history, save_chat_message
from app.agents import (
    create_text_engine_agent, Deps,
    create_voice_sales_rep_agent, VoiceDeps,
    create_ghl_booking_agent, BookingDeps,
    create_knowledgebase_agent, KBDeps,
    create_database_reactivation_agent, ReactivationDeps,
    create_lead_scoring_agent, ScoringDeps
)
import json

router = APIRouter()


async def get_common_context(GHL_Account_ID: str, Lead_ID: str, Message_Body: str = None) -> tuple:
    """Retrieve common context needed for webhook processing.

    Fetches client configuration, prompts, and chat history from Supabase.
    Optionally saves incoming user message to chat history.

    Args:
        GHL_Account_ID: The GoHighLevel account/location identifier.
        Lead_ID: The lead identifier for chat history.
        Message_Body: Optional incoming message body to save to history.

    Returns:
        A tuple of (client_config, prompts, history_str).

    Raises:
        ValueError: If no client is found for the given GHL_Account_ID.
    """
    client_config = await get_client_config(GHL_Account_ID)
    prompts = await get_client_prompts(
        client_config["supabase_url"],
        client_config["supabase_service_key"]
    )
    history = await get_chat_history(
        client_config["supabase_url"],
        client_config["supabase_service_key"],
        Lead_ID
    )
    if Message_Body:
        user_message_entry = {"role": "user", "content": Message_Body}
        await save_chat_message(
            client_config["supabase_url"],
            client_config["supabase_service_key"],
            Lead_ID,
            user_message_entry
        )
        history_str = json.dumps([h.get("message", {}) for h in history]) + "\n" + json.dumps(user_message_entry)
    else:
        history_str = json.dumps([h.get("message", {}) for h in history])

    return client_config, prompts, history_str

@router.post("/text-engine", response_model=AgentResponse)
async def text_engine_webhook(
    Message_Body: str = Query(...),
    Lead_ID: str = Query(...),
    GHL_Account_ID: str = Query(...),
    Name: str = Query(default=""),
    Email: str = Query(default=""),
    Phone: str = Query(default=""),
    Setter_Number: str = Query(default="1")
) -> AgentResponse:
    """Process incoming text messages through the AI text engine.

    This webhook handles lead conversations by:
    1. Loading client prompts and chat history from Supabase
    2. Running lead scoring to qualify the lead
    3. Generating AI responses using the text engine agent
    4. Saving all responses to chat history

    Args:
        Message_Body: The incoming message from the lead.
        Lead_ID: The lead identifier in GHL.
        GHL_Account_ID: The GoHighLevel location/account ID.
        Name: Optional lead name.
        Email: Optional lead email.
        Phone: Optional lead phone.
        Setter_Number: Which setter configuration to use (default: "1").

    Returns:
        AgentResponse containing 1-5 message chunks for the lead.

    Raises:
        HTTPException: 500 if any error occurs during processing.
    """
    try:
        client_config, prompts, history_str = await get_common_context(GHL_Account_ID, Lead_ID, Message_Body)
        model_name = client_config.get("llm_model", "google/gemini-2.5-flash")
        api_key = client_config["openrouter_api_key"]

        # Lead Scoring Agent execution
        scoring_agent = create_lead_scoring_agent(model_name, api_key)
        scoring_deps = ScoringDeps(chat_history_str=history_str)
        scoring_result = await scoring_agent.run("Calculate lead score", deps=scoring_deps)
        lead_score = scoring_result.data.score

        # Save score in memory/DB if needed (optional based on your GHL workflow)
        # For this prototype we will just print it to logs
        print(f"Lead Score for {Lead_ID}: {lead_score}")

        # Main text engine execution
        agent = create_text_engine_agent(model_name, api_key)
        deps = Deps(
            user_details={"Name": Name, "Email": Email, "Phone": Phone, "Score": lead_score},
            chat_history=history_str,
            prompts=prompts
        )

        result = await agent.run(Message_Body, deps=deps)
        response_data = result.data

        # Save ALL assistant messages to history
        messages_to_save = [
            msg for msg in [
                response_data.Message_1,
                response_data.Message_2,
                response_data.Message_3,
                response_data.Message_4,
                response_data.Message_5
            ] if msg
        ]

        for msg in messages_to_save:
            await save_chat_message(
                client_config["supabase_url"],
                client_config["supabase_service_key"],
                Lead_ID,
                {"role": "assistant", "content": msg}
            )

        return AgentResponse(
            Message_1=response_data.Message_1,
            Message_2=response_data.Message_2,
            Message_3=response_data.Message_3,
            Message_4=response_data.Message_4,
            Message_5=response_data.Message_5
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/voice-sales-rep")
async def voice_sales_rep_webhook(
    Message_Body: str = Query(...),
    Lead_ID: str = Query(...),
    GHL_Account_ID: str = Query(...)
) -> dict:
    """Process voice sales rep requests through the AI agent.

    Handles voice-based sales conversations by loading client prompts
    and generating conversational responses.

    Args:
        Message_Body: The incoming message from the lead.
        Lead_ID: The lead identifier in GHL.
        GHL_Account_ID: The GoHighLevel location/account ID.

    Returns:
        dict with 'response' key containing the AI-generated response.

    Raises:
        HTTPException: 500 if any error occurs during processing.
    """
    try:
        client_config, prompts, history_str = await get_common_context(GHL_Account_ID, Lead_ID, Message_Body)
        model_name = client_config.get("llm_model", "google/gemini-2.5-flash")
        agent = create_voice_sales_rep_agent(model_name, client_config["openrouter_api_key"])
        deps = VoiceDeps(user_details={"Lead_ID": Lead_ID}, prompts=prompts)
        result = await agent.run(Message_Body, deps=deps)
        return {"response": result.data.response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ghl-booking")
async def ghl_booking_webhook(
    Message_Body: str = Query(...),
    Lead_ID: str = Query(...),
    GHL_Account_ID: str = Query(...)
) -> dict:
    """Process GHL booking requests through the AI agent.

    Handles appointment booking workflows by loading booking prompts
    and generating booking actions.

    Args:
        Message_Body: The incoming message from the lead.
        Lead_ID: The lead identifier in GHL.
        GHL_Account_ID: The GoHighLevel location/account ID.

    Returns:
        dict with 'action' and 'message' keys.

    Raises:
        HTTPException: 500 if any error occurs during processing.
    """
    try:
        client_config, prompts, history_str = await get_common_context(GHL_Account_ID, Lead_ID, Message_Body)
        model_name = client_config.get("llm_model", "google/gemini-2.5-flash")
        agent = create_ghl_booking_agent(model_name, client_config["openrouter_api_key"])
        deps = BookingDeps(user_details={"Lead_ID": Lead_ID}, prompts=prompts)
        result = await agent.run(Message_Body, deps=deps)
        return {"action": result.data.action, "message": result.data.message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/knowledgebase-automation")
async def knowledgebase_webhook(
    Message_Body: str = Query(...),
    Lead_ID: str = Query(...),
    GHL_Account_ID: str = Query(...)
) -> dict:
    """Process knowledge base queries through the AI agent.

    Answers user questions based on knowledge base rules loaded
    from the client's prompts.

    Args:
        Message_Body: The user's question.
        Lead_ID: The lead identifier in GHL.
        GHL_Account_ID: The GoHighLevel location/account ID.

    Returns:
        dict with 'answer' key containing the retrieved answer.

    Raises:
        HTTPException: 500 if any error occurs during processing.
    """
    try:
        client_config, prompts, history_str = await get_common_context(GHL_Account_ID, Lead_ID, Message_Body)
        model_name = client_config.get("llm_model", "google/gemini-2.5-flash")
        agent = create_knowledgebase_agent(model_name, client_config["openrouter_api_key"])
        deps = KBDeps(user_details={"Lead_ID": Lead_ID}, prompts=prompts)
        result = await agent.run(Message_Body, deps=deps)
        return {"answer": result.data.answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/database-reactivation")
async def database_reactivation_webhook(
    Message_Body: str = Query(default=""),
    Lead_ID: str = Query(...),
    GHL_Account_ID: str = Query(...)
) -> dict:
    """Process database reactivation requests through the AI agent.

    Generates reactivation messages for cold leads based on
    persona and reactivation prompts from the database.

    Args:
        Message_Body: Optional message context for reactivation.
        Lead_ID: The lead identifier in GHL.
        GHL_Account_ID: The GoHighLevel location/account ID.

    Returns:
        dict with 'message' key containing the reactivation message.

    Raises:
        HTTPException: 500 if any error occurs during processing.
    """
    try:
        client_config, prompts, history_str = await get_common_context(GHL_Account_ID, Lead_ID)
        model_name = client_config.get("llm_model", "google/gemini-2.5-flash")
        agent = create_database_reactivation_agent(model_name, client_config["openrouter_api_key"])
        deps = ReactivationDeps(user_details={"Lead_ID": Lead_ID}, prompts=prompts)
        result = await agent.run(f"Reactivate lead {Lead_ID}", deps=deps)
        return {"message": result.data.message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
