from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from typing import Optional
from pydantic_ai.models.openai import OpenAIChatModel

class VoiceDeps(BaseModel):
    user_details: dict
    prompts: dict

class VoiceRepResult(BaseModel):
    response: str = Field(description="The response for the voice assistant")

def create_voice_sales_rep_agent(model_name: str, api_key: str):
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
