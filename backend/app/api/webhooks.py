from fastapi import APIRouter, Depends, HTTPException, Query, Header, Request
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
from app.core.config import settings
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

def verify_webhook_secret(x_webhook_secret: str = Header(None)):
    if x_webhook_secret != settings.webhook_secret:
        logger.error("Unauthorized webhook attempt")
        raise HTTPException(status_code=401, detail="Unauthorized")
    return x_webhook_secret

async def get_common_context(GHL_Account_ID: str, Lead_ID: str, Message_Body: str = None, Execution_ID: str = None):
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

    is_duplicate = False
    if Message_Body and history and Execution_ID:
        for h in reversed(history):
            msg = h.get("message", {})
            if msg.get("type") == "human" and msg.get("execution_id") == Execution_ID:
                is_duplicate = True
                break

    if Message_Body and not is_duplicate:
        user_message_entry = {"type": "human", "content": Message_Body}
        if Execution_ID:
            user_message_entry["execution_id"] = Execution_ID

        await save_chat_message(
            client_config["supabase_url"],
            client_config["supabase_service_key"],
            Lead_ID,
            user_message_entry
        )
        history_str = json.dumps([h.get("message", {}) for h in history]) + "\n" + json.dumps(user_message_entry)
    else:
        history_str = json.dumps([h.get("message", {}) for h in history])

    return client_config, prompts, history_str, history

@router.post("/text-engine", response_model=AgentResponse)
async def text_engine_webhook(
    request: Request,
    _=Depends(verify_webhook_secret),
    Message_Body: str = Query(...),
    Lead_ID: str = Query(...),
    GHL_Account_ID: str = Query(...),
    Execution_ID: str = Query(default=None),
    Name: str = Query(default=""),
    Email: str = Query(default=""),
    Phone: str = Query(default=""),
    Setter_Number: str = Query(default="1")
):
    try:
        client_config, prompts, history_str, history = await get_common_context(GHL_Account_ID, Lead_ID, Message_Body, Execution_ID)
        model_name = client_config.get("llm_model", "google/gemini-2.5-flash")
        api_key = client_config["openrouter_api_key"]

        scoring_agent = create_lead_scoring_agent(model_name, api_key)
        scoring_deps = ScoringDeps(chat_history_str=history_str)
        scoring_prompt = f"Calculate lead score based on this chat history:\n{history_str}"
        scoring_result = await scoring_agent.run(scoring_prompt, deps=scoring_deps)
        lead_score = scoring_result.data.score

        print(f"Lead Score for {Lead_ID}: {lead_score}")

        agent = create_text_engine_agent(model_name, api_key)
        deps = Deps(
            user_details={"Name": Name, "Email": Email, "Phone": Phone, "Score": lead_score},
            chat_history=history_str,
            prompts=prompts
        )

        user_prompt = f"User Contact Details: {deps.user_details}\n\nChat History:\n{deps.chat_history}\n\nUser Request: {Message_Body}"
        result = await agent.run(user_prompt, deps=deps)
        response_data = result.data

        messages_to_save = [
            msg for msg in [
                response_data.Message_1,
                response_data.Message_2,
                response_data.Message_3,
                response_data.Message_4,
                response_data.Message_5
            ] if msg
        ]

        existing_ai_messages = [
            h.get("message", {}).get("content")
            for h in history
            if h.get("message", {}).get("type") == "ai"
        ]

        for msg in messages_to_save:
            if msg not in existing_ai_messages:
                await save_chat_message(
                    client_config["supabase_url"],
                    client_config["supabase_service_key"],
                    Lead_ID,
                    {"type": "ai", "content": msg}
                )
                existing_ai_messages.append(msg)

        return AgentResponse(
            Message_1=response_data.Message_1,
            Message_2=response_data.Message_2,
            Message_3=response_data.Message_3,
            Message_4=response_data.Message_4,
            Message_5=response_data.Message_5
        )
    except Exception as e:
        logger.exception("Webhook failed")
        raise HTTPException(status_code=500, detail="Internal server error") from e

@router.post("/voice-sales-rep")
async def voice_sales_rep_webhook(
    request: Request,
    _=Depends(verify_webhook_secret),
    Message_Body: str = Query(...),
    Lead_ID: str = Query(...),
    GHL_Account_ID: str = Query(...)
):
    try:
        client_config, prompts, history_str, history = await get_common_context(GHL_Account_ID, Lead_ID, Message_Body, Execution_ID)
        model_name = client_config.get("llm_model", "google/gemini-2.5-flash")
        agent = create_voice_sales_rep_agent(model_name, client_config["openrouter_api_key"])
        deps = VoiceDeps(user_details={"Lead_ID": Lead_ID}, prompts=prompts)
        result = await agent.run(Message_Body, deps=deps)
        return {"response": result.data.response}
    except Exception as e:
        logger.exception("Webhook failed")
        raise HTTPException(status_code=500, detail="Internal server error") from e

@router.post("/ghl-booking")
async def ghl_booking_webhook(
    request: Request,
    _=Depends(verify_webhook_secret),
    Message_Body: str = Query(...),
    Lead_ID: str = Query(...),
    GHL_Account_ID: str = Query(...)
):
    try:
        client_config, prompts, history_str, history = await get_common_context(GHL_Account_ID, Lead_ID, Message_Body, Execution_ID)
        model_name = client_config.get("llm_model", "google/gemini-2.5-flash")
        agent = create_ghl_booking_agent(model_name, client_config["openrouter_api_key"])
        deps = BookingDeps(user_details={"Lead_ID": Lead_ID}, prompts=prompts)
        result = await agent.run(Message_Body, deps=deps)
        return {"action": result.data.action, "message": result.data.message}
    except Exception as e:
        logger.exception("Webhook failed")
        raise HTTPException(status_code=500, detail="Internal server error") from e

@router.post("/knowledgebase-automation")
async def knowledgebase_webhook(
    request: Request,
    _=Depends(verify_webhook_secret),
    Message_Body: str = Query(...),
    Lead_ID: str = Query(...),
    GHL_Account_ID: str = Query(...)
):
    try:
        client_config, prompts, history_str, history = await get_common_context(GHL_Account_ID, Lead_ID, Message_Body, Execution_ID)
        model_name = client_config.get("llm_model", "google/gemini-2.5-flash")
        agent = create_knowledgebase_agent(model_name, client_config["openrouter_api_key"])
        deps = KBDeps(user_details={"Lead_ID": Lead_ID}, prompts=prompts)
        result = await agent.run(Message_Body, deps=deps)
        return {"answer": result.data.answer}
    except Exception as e:
        logger.exception("Webhook failed")
        raise HTTPException(status_code=500, detail="Internal server error") from e

@router.post("/database-reactivation")
async def database_reactivation_webhook(
    request: Request,
    _=Depends(verify_webhook_secret),
    Message_Body: str = Query(default=""),
    Lead_ID: str = Query(...),
    GHL_Account_ID: str = Query(...)
):
    try:
        client_config, prompts, history_str, history = await get_common_context(GHL_Account_ID, Lead_ID, Message_Body)
        model_name = client_config.get("llm_model", "google/gemini-2.5-flash")
        agent = create_database_reactivation_agent(model_name, client_config["openrouter_api_key"])
        deps = ReactivationDeps(user_details={"Lead_ID": Lead_ID}, prompts=prompts)
        result = await agent.run(f"Reactivate lead {Lead_ID}", deps=deps)
        return {"message": result.data.message}
    except Exception as e:
        logger.exception("Webhook failed")
        raise HTTPException(status_code=500, detail="Internal server error") from e
