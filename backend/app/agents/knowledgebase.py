from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from pydantic_ai.models.openai import OpenAIChatModel

class KBDeps(BaseModel):
    user_details: dict
    prompts: dict

class KBResult(BaseModel):
    answer: str = Field(description="Answer from the knowledge base")

def create_knowledgebase_agent(model_name: str, api_key: str):
    model = OpenAIChatModel(
        model_name,
        api_key=api_key,
        base_url='https://openrouter.ai/api/v1',
    )

    agent = Agent(
        model,
        result_type=KBResult,
        deps_type=KBDeps,
        retries=3,
    )

    @agent.system_prompt
    def system_prompt(ctx: RunContext[KBDeps]) -> str:
        return f"You are a knowledge base assistant. Answer the user's question based on these rules: {ctx.deps.prompts.get('Prompt_1', '')}"

    return agent
