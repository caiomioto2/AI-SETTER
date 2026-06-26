from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from pydantic_ai.models.openai import OpenAIChatModel

class BookingDeps(BaseModel):
    user_details: dict
    prompts: dict

class BookingResult(BaseModel):
    action: str = Field(description="The action taken (e.g., booked, unavailable, etc.)")
    message: str = Field(description="Response message to the user")

def create_ghl_booking_agent(model_name: str, api_key: str):
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
