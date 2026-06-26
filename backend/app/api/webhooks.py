from fastapi import APIRouter, Depends, HTTPException, Query, Header, Request
from typing import Optional
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
processed_messages = set()


from app.core.config import settings
import os
import logging

logger = logging.getLogger(__name__)

async def verify_webhook_secret(request: Request, x_webhook_secret: str = Header(None)):
    expected_secret = os.getenv("WEBHOOK_SECRET")
    if expected_secret and x_webhook_secret != expected_secret:
        raise HTTPException(status_code=401, detail="Unauthorized")


async def get_common_context(GHL_Account_ID: str, Lead_ID: str, Message_Body: Optional[str] = None):
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
        msg_id = f"{Lead_ID}:{hash(Message_Body)}"
        if msg_id in processed_messages:
            history_str = json.dumps([h.get("message", {}) for h in history])
            return client_config, prompts, history_str, False, True
        processed_messages.add(msg_id)

        user_message_entry = {"type": "human", "data": {"content": Message_Body}}

        await save_chat_message(
            client_config["supabase_url"],
            client_config["supabase_service_key"],
            Lead_ID,
            user_message_entry
        )
        history_str = json.dumps([h.get("message", {}) for h in history]) + "\n" + json.dumps(user_message_entry)
    else:
        history_str = json.dumps([h.get("message", {}) for h in history])

    return client_config, prompts, history_str, False

@router.post("/text-engine", response_model=AgentResponse, dependencies=[Depends(verify_webhook_secret)])
async def text_engine_webhook(
    Message_Body: str = Query(...),
    Lead_ID: str = Query(...),
    GHL_Account_ID: str = Query(...),
    Name: str = Query(default=""),
    Email: str = Query(default=""),
    Phone: str = Query(default=""),
    Setter_Number: str = Query(default="1")
):
    try:
        client_config, prompts, history_str, is_duplicate = await get_common_context(GHL_Account_ID, Lead_ID, Message_Body)
        if is_duplicate:
            return AgentResponse(Message_1="Duplicate message, skipped")

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
                {"type": "ai", "data": {"content": msg}}
            )

        return AgentResponse(
            Message_1=response_data.Message_1,
            Message_2=response_data.Message_2,
            Message_3=response_data.Message_3,
            Message_4=response_data.Message_4,
            Message_5=response_data.Message_5
        )
    except Exception as e:
        logger.exception("Internal server error in webhook")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/voice-sales-rep", dependencies=[Depends(verify_webhook_secret)])
async def voice_sales_rep_webhook(
    Message_Body: str = Query(...),
    Lead_ID: str = Query(...),
    GHL_Account_ID: str = Query(...)
):
    try:
        client_config, prompts, history_str, is_duplicate = await get_common_context(GHL_Account_ID, Lead_ID, Message_Body)
        if is_duplicate:
            return AgentResponse(Message_1="Duplicate message, skipped")

        model_name = client_config.get("llm_model", "google/gemini-2.5-flash")
        agent = create_voice_sales_rep_agent(model_name, client_config["openrouter_api_key"])
        deps = VoiceDeps(user_details={"Lead_ID": Lead_ID}, prompts=prompts)
        result = await agent.run(Message_Body, deps=deps)
        return {"response": result.data.response}
    except Exception as e:
        logger.exception("Internal server error in webhook")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/ghl-booking", dependencies=[Depends(verify_webhook_secret)])
async def ghl_booking_webhook(
    Message_Body: str = Query(...),
    Lead_ID: str = Query(...),
    GHL_Account_ID: str = Query(...)
):
    try:
        client_config, prompts, history_str, is_duplicate = await get_common_context(GHL_Account_ID, Lead_ID, Message_Body)
        if is_duplicate:
            return AgentResponse(Message_1="Duplicate message, skipped")

        model_name = client_config.get("llm_model", "google/gemini-2.5-flash")
        agent = create_ghl_booking_agent(model_name, client_config["openrouter_api_key"])
        deps = BookingDeps(user_details={"Lead_ID": Lead_ID}, prompts=prompts)
        result = await agent.run(Message_Body, deps=deps)
        return {"action": result.data.action, "message": result.data.message}
    except Exception as e:
        logger.exception("Internal server error in webhook")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/knowledgebase-automation", dependencies=[Depends(verify_webhook_secret)])
async def knowledgebase_webhook(
    Message_Body: str = Query(...),
    Lead_ID: str = Query(...),
    GHL_Account_ID: str = Query(...)
):
    try:
        client_config, prompts, history_str, is_duplicate = await get_common_context(GHL_Account_ID, Lead_ID, Message_Body)
        if is_duplicate:
            return AgentResponse(Message_1="Duplicate message, skipped")

        model_name = client_config.get("llm_model", "google/gemini-2.5-flash")
        agent = create_knowledgebase_agent(model_name, client_config["openrouter_api_key"])
        deps = KBDeps(user_details={"Lead_ID": Lead_ID}, prompts=prompts)
        result = await agent.run(Message_Body, deps=deps)
        return {"answer": result.data.answer}
    except Exception as e:
        logger.exception("Internal server error in webhook")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/database-reactivation", dependencies=[Depends(verify_webhook_secret)])
async def database_reactivation_webhook(
    Message_Body: str = Query(default=""),
    Lead_ID: str = Query(...),
    GHL_Account_ID: str = Query(...)
):
    try:
        client_config, prompts, history_str, _ = await get_common_context(GHL_Account_ID, Lead_ID)
        model_name = client_config.get("llm_model", "google/gemini-2.5-flash")
        agent = create_database_reactivation_agent(model_name, client_config["openrouter_api_key"])
        deps = ReactivationDeps(user_details={"Lead_ID": Lead_ID}, prompts=prompts)
        result = await agent.run(f"Reactivate lead {Lead_ID}", deps=deps)
        return {"message": result.data.message}
    except Exception as e:
        logger.exception("Internal server error in webhook")
        raise HTTPException(status_code=500, detail="Internal server error")
