from textwrap import dedent
from typing import Any, cast, Literal
import json
import os

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.ag_ui import StateDeps
from ag_ui.core import EventType, StateSnapshotEvent
from pydantic_ai.models.openai import OpenAIResponsesModel
from tavily import TavilyClient

# load environment variables
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# =====
# State
# =====
class ProverbsState(BaseModel):
    """List of the proverbs being written."""

    proverbs: list[str] = Field(
        default_factory=list,
        description="The list of already written proverbs",
    )


# =====
# Agent
# =====
agent = Agent(
    model=OpenAIResponsesModel("gpt-4.1-mini"),
    deps_type=StateDeps[ProverbsState],
    system_prompt=dedent("""
    You are a helpful assistant that helps manage and discuss proverbs.
    
    The user has a list of proverbs that you can help them manage.
    You have tools available to add, set, or retrieve proverbs from the list.
    
    When discussing proverbs, ALWAYS use the get_proverbs tool to see the current list before
    mentioning, updating, or discussing proverbs with the user.
  """).strip(),
)


# =====
# Tools
# =====
@agent.tool
def get_proverbs(ctx: RunContext[StateDeps[ProverbsState]]) -> list[str]:
    """Get the current list of proverbs."""
    print(f"📖 Getting proverbs: {ctx.deps.state.proverbs}")
    return ctx.deps.state.proverbs


@agent.tool
async def add_proverbs(
    ctx: RunContext[StateDeps[ProverbsState]], proverbs: list[str]
) -> StateSnapshotEvent:
    ctx.deps.state.proverbs.extend(proverbs)
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state,
    )


@agent.tool
async def set_proverbs(
    ctx: RunContext[StateDeps[ProverbsState]], proverbs: list[str]
) -> StateSnapshotEvent:
    ctx.deps.state.proverbs = proverbs
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state,
    )


@agent.tool
def get_weather(_: RunContext[StateDeps[ProverbsState]], location: str) -> str:
    """Get the weather for a given location. Ensure location is fully spelled out."""
    return f"The weather in {location} is sunny."


@agent.tool
def search_web(
    _: RunContext[StateDeps[ProverbsState]],
    query: str,
    search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "basic",
    topic: Literal["general", "news", "finance"] = "general",
    time_range: Literal["day", "week", "month", "year"] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    days: int | None = None,
    max_results: int = 10,
) -> dict:
    """
    Web search tool using Tavily API.

    Args:
        query: The search query string
        search_depth: "basic" or "advanced"
        topic: "general", "news", or "finance"
        time_range: "day", "week", "month" or "year" (optional)
        start_date: begining of date range yyy-mm-dd
        end_date: end of date range yyy-mm-dd
        days: Number of days back to search (optional)
        max_results: Maximum number of results
    """
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

    kwargs = {
        "search_depth": search_depth,
        "topic": topic,
        "time_range": time_range,
        "start_date": start_date,
        "end_date": end_date,
        "days": days,
        "max_results": max_results,
    }
    # Filter out None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    response = tavily_client.search(query, **cast(dict[str, Any], kwargs))

    with open("search_results.json", "w", encoding="utf-8") as f:
        json.dump(response, f, indent=2)
    return response
