from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from pydantic_ai.models.openai import OpenAIChatModel

class KBDeps(BaseModel):
    """Dependency injection model for the knowledge base agent.

    Attributes:
        user_details: Dictionary containing user information.
        prompts: Dictionary of prompts with knowledge base rules (Prompt_1).
    """
    user_details: dict
    prompts: dict


class KBResult(BaseModel):
    """Result model for knowledge base queries.

    Attributes:
        answer: Answer retrieved from the knowledge base.
    """
    answer: str = Field(description="Answer from the knowledge base")


def create_knowledgebase_agent(model_name: str, api_key: str) -> Agent:
    """Create a knowledge base agent using an OpenAI-compatible model via OpenRouter.

    This agent answers user questions based on knowledge base rules
    loaded from the client's database.

    Args:
        model_name: The model identifier (e.g., 'google/gemini-2.5-flash').
        api_key: The OpenRouter API key for authentication.

    Returns:
        A configured Pydantic AI Agent instance for knowledge base queries.
    """
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
