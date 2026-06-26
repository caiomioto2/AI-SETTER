from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from typing import Optional
from pydantic_ai.models.openai import OpenAIChatModel

class VoiceDeps(BaseModel):
    """Dependency injection model for the voice sales rep agent.

    Attributes:
        user_details: Dictionary containing user information.
        prompts: Dictionary of prompts indexed by Prompt_Name.
    """
    user_details: dict
    prompts: dict


class VoiceRepResult(BaseModel):
    """Result model for voice sales rep responses.

    Attributes:
        response: The conversational response for the voice assistant.
    """
    response: str = Field(description="The response for the voice assistant")


def create_voice_sales_rep_agent(model_name: str, api_key: str) -> Agent:
    """Create a voice sales rep agent using an OpenAI-compatible model via OpenRouter.

    This agent handles voice-based sales conversations with leads,
    using dynamic prompts loaded from the client's database.

    Args:
        model_name: The model identifier (e.g., 'google/gemini-2.5-flash').
        api_key: The OpenRouter API key for authentication.

    Returns:
        A configured Pydantic AI Agent instance for voice sales.
    """
    model = OpenAIChatModel(
        model_name,
        api_key=api_key,
        base_url='https://openrouter.ai/api/v1',
    )

    agent = Agent(
        model,
        result_type=VoiceRepResult,
        deps_type=VoiceDeps,
        retries=3,
    )

    @agent.system_prompt
    def system_prompt(ctx: RunContext[VoiceDeps]) -> str:
        prompt = (
            f"# Bot Persona:\n{ctx.deps.prompts.get('Prompt_0', '')}\n\n"
            f"# Voice Instructions:\n{ctx.deps.prompts.get('Prompt_1', '')}\n\n"
            f"# User Details:\n{ctx.deps.user_details}\n\n"
            f"Please respond conversationally as a voice sales rep."
        )
        return prompt

    return agent
