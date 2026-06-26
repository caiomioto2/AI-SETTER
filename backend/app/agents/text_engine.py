from pydantic_ai import Agent, RunContext, Tool
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from app.models.webhook import AgentResponse
from pydantic_ai.models.openai import OpenAIChatModel

# We will define a dynamic agent that will be instantiated per request
# Since prompts are dynamic from DB, we inject them into the run context or during agent initialization.

class Deps(BaseModel):
    user_details: dict
    chat_history: str
    prompts: dict

# Result model as required by Trigger.dev
class TextEngineResult(BaseModel):
    Message_1: str = Field(description="The first response message chunk")
    Message_2: Optional[str] = Field(default=None, description="The second response message chunk if split is needed")
    Message_3: Optional[str] = Field(default=None, description="The third response message chunk if split is needed")
    Message_4: Optional[str] = Field(default=None, description="The fourth response message chunk if split is needed")
    Message_5: Optional[str] = Field(default=None, description="The fifth response message chunk if split is needed")


# Creating a factory function so we can bind dynamic models (e.g., from OpenRouter)
def create_text_engine_agent(model_name: str, api_key: str):
    """
    Creates an Agent instance using an OpenAI-compatible model (OpenRouter).
    """
    model = OpenAIChatModel(
        model_name,
        api_key=api_key,
        base_url='https://openrouter.ai/api/v1',
    )

    agent = Agent(
        model,
        result_type=TextEngineResult,
        deps_type=Deps,
        retries=3,
    )

    @agent.system_prompt
    def system_prompt(ctx: RunContext[Deps]) -> str:
        bot_persona = ctx.deps.prompts.get("Prompt_0", "You are an AI assistant.")
        system_rules = ctx.deps.prompts.get("Prompt_1", "")
        booking_prompt = ctx.deps.prompts.get("Prompt_7", "")

        prompt = (
            f"# Bot Persona Prompt:\n{bot_persona}\n\n"
            f"# System Prompt:\n{system_rules}\n\n"
            f"# Booking Function Prompt:\n{booking_prompt}\n\n"
            f"# User Contact Details:\n{ctx.deps.user_details}\n\n"
            f"# Chat History:\n{ctx.deps.chat_history}\n\n"
            f"Respond appropriately following the bot persona and rules. Output your final response broken down into 1-5 messages."
        )
        return prompt

    return agent
