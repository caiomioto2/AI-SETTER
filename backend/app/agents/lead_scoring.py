from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from pydantic_ai.models.openai import OpenAIChatModel

class ScoringDeps(BaseModel):
    chat_history_str: str

class LeadScoreResult(BaseModel):
    score: int = Field(ge=1, le=500, description="Lead score based on conversation history")

def create_lead_scoring_agent(model_name: str, api_key: str):
    model = OpenAIChatModel(
        model_name,
        api_key=api_key,
        base_url='https://openrouter.ai/api/v1',
    )

    agent = Agent(
        model,
        result_type=LeadScoreResult,
        deps_type=ScoringDeps,
        retries=3,
    )

    @agent.system_prompt
    def system_prompt(ctx: RunContext[ScoringDeps]) -> str:
        return f"""
# Prospect Scoring AI - LEARN FROM CONVERSATION HISTORY

You are a scoring engine that analyzes the user's final message AND the full conversation history to assign a numerical score to the prospect.

The minimum score is 1 (default for prospects with no meaningful interaction).
The maximum score is 500 based on cumulative point additions.

## SCORING LOGIC (APPLY ALL THAT APPLY)

- +5 points: The user sends any reply or interaction.
- +5 points: The user answered at least one question from the AI.
- +10 points: The user answered at least three questions.
- +10 points: The user asked at least one question.
- +20 points: The user asked at least three questions.
- +30 points: The user mentioned that they previously tried running webinars.
- +20 points: The user mentioned they are currently using ANY AI tools.
- +60 points: The user's comments imply they are making money, not struggling, and run a functioning business.
- +25 points: The user briefly explains what they do or what they sell.
- +50 points: The user explains their business in more depth.
- +100 points: The user explicitly states they are looking for a solution like ours.
- +80 points: The user explicitly states they are actively running webinars right now.

## Chat History
{ctx.deps.chat_history_str}
"""
    return agent
