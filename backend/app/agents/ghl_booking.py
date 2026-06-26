from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from pydantic_ai.models.openai import OpenAIChatModel

class BookingDeps(BaseModel):
    """Dependency injection model for the GHL booking agent.

    Attributes:
        user_details: Dictionary containing user information.
        prompts: Dictionary of prompts indexed by Prompt_Name.
    """
    user_details: dict
    prompts: dict


class BookingResult(BaseModel):
    """Result model for GHL booking actions.

    Attributes:
        action: The action taken (e.g., booked, unavailable, etc.).
        message: Response message to the user.
    """
    action: str = Field(description="The action taken (e.g., booked, unavailable, etc.)")
    message: str = Field(description="Response message to the user")


def create_ghl_booking_agent(model_name: str, api_key: str) -> Agent:
    """Create a GHL booking agent using an OpenAI-compatible model via OpenRouter.

    This agent handles appointment booking workflows integrated with GoHighLevel,
    using dynamic prompts for booking instructions.

    Args:
        model_name: The model identifier (e.g., 'google/gemini-2.5-flash').
        api_key: The OpenRouter API key for authentication.

    Returns:
        A configured Pydantic AI Agent instance for booking workflows.
    """
    model = OpenAIChatModel(
        model_name,
        api_key=api_key,
        base_url='https://openrouter.ai/api/v1',
    )

    agent = Agent(
        model,
        result_type=BookingResult,
        deps_type=BookingDeps,
        retries=3,
    )

    @agent.system_prompt
    def system_prompt(ctx: RunContext[BookingDeps]) -> str:
        prompt = (
            f"# Booking Instructions:\n{ctx.deps.prompts.get('Prompt_7', 'Help the user book an appointment.')}\n\n"
            f"# User Details:\n{ctx.deps.user_details}\n\n"
        )
        return prompt

    return agent
