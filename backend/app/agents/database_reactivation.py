from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from pydantic_ai.models.openai import OpenAIChatModel

class ReactivationDeps(BaseModel):
    user_details: dict
    prompts: dict

class ReactivationResult(BaseModel):
    message: str = Field(description="Reactivation message to the lead")

def create_database_reactivation_agent(model_name: str, api_key: str):
    model = OpenAIChatModel(
        model_name,
        api_key=api_key,
        base_url='https://openrouter.ai/api/v1',
    )

    agent = Agent(
        model,
        result_type=ReactivationResult,
        deps_type=ReactivationDeps,
        retries=3,
    )

    @agent.system_prompt
    def system_prompt(ctx: RunContext[ReactivationDeps]) -> str:
        prompt = (
            f"# Persona:\n{ctx.deps.prompts.get('Prompt_0', 'You are an assistant.')}\n\n"
            f"# Reactivation Instructions:\n{ctx.deps.prompts.get('Prompt_1', 'Reactivate cold leads.')}\n\n"
            f"# User Details:\n{ctx.deps.user_details}\n\n"
        )
        return prompt

    return agent
