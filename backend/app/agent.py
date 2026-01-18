from datetime import datetime
from textwrap import dedent
from typing import Any, Literal, cast

from ag_ui.core import EventType, StateSnapshotEvent
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.ag_ui import StateDeps
from pydantic_ai.common_tools.tavily import TavilySearchResult
from pydantic_ai.models.openai import OpenAIResponsesModel
from tavily import TavilyClient

from app.config import settings
from app.schemas import (
    CompanyHeadcount,
    LinkedinInmailBalance,
    LinkedinSearchItem,
    LinkedinSearchParameter,
    LinkedinSearchRequest,
    LinkedinSearchResponse,
    SearchFilterInclude,
    TenureFilter,
    UnipileAccountResponse,
)
from app.services import Unipile

_tavily_client = TavilyClient(api_key=settings.tavily_api_key.get_secret_value())


# =====
# State
# =====
class ChatState(BaseModel):
    """State for the chat agent."""

    proverbs: list[str] = Field(
        default_factory=list,
        description="The list of already written proverbs",
    )
    web_search_results: list[TavilySearchResult] = Field(
        default_factory=list,
        description="The list of web search results",
    )
    linkedin_search_parameters: list[LinkedinSearchParameter] = Field(
        default_factory=list,
        description="The list of LinkedIn search parameters (locations, companies, schools, etc.)",
    )
    linkedin_search_results: list[LinkedinSearchItem] = Field(
        default_factory=list,
        description="The list of LinkedIn search results (people, companies, etc.)",
    )


# =====
# Agent
# =====
agent = Agent(
    model=OpenAIResponsesModel("gpt-4.1-mini"),
    deps_type=StateDeps[ChatState],
    system_prompt=dedent("""
    You are a helpful assistant that helps manage proverbs and search LinkedIn.

    ## Proverbs
    The user has a list of proverbs that you can help them manage.
    You have tools available to add, set, or retrieve proverbs from the list.
    When discussing proverbs, ALWAYS use the get_proverbs tool to see the current list before
    mentioning, updating, or discussing proverbs with the user.

    ## Web Search
    Use the search_web tool to search the web.
    Always use the get_web_search_results tool to return the web search results to the user or to list previous web search results.

    ## LinkedIn Search
    You can search LinkedIn for people, companies, jobs, and posts using the following workflow:

    1. Use get_search_parameters to look up IDs for locations, companies, schools, industries, skills, etc.
       - Specify the type of parameter (e.g., LOCATION, COMPANY, SCHOOL, INDUSTRY, SKILL)
       - Specify the service (CLASSIC, SALES_NAVIGATOR, or RECRUITER) - defaults to CLASSIC
       - Use keywords to filter results
       - Results accumulate in state, so you can call this multiple times to gather all needed IDs

    2. Use search_linkedin to search for people, companies, jobs, or posts
       - Use the IDs from get_search_parameters for location, company, school, industry filters
       - Results accumulate in state - use cursor parameter to paginate through results
       - Use clear_linkedin_search_results before starting a new search if needed
       - The limit is automatically set to maximum based on API type (50 for classic, 100 for sales_navigator/recruiter)

    3. Use get_linkedin_search_results to retrieve and present the search results to the user

    4. Use clear_linkedin_search_results when the user wants to start a fresh search

    ## Other Tools
    If you require the current date or time use get_datetime.
    Use get_account_info to get the user's Unipile account information.
    Use get_inmail_balance to check the user's LinkedIn InMail balance.
  """).strip(),
)


# =====
# Tools
# =====
@agent.tool
def get_proverbs(ctx: RunContext[StateDeps[ChatState]]) -> list[str]:
    """Get the current list of proverbs."""
    return ctx.deps.state.proverbs


@agent.tool
async def add_proverbs(
    ctx: RunContext[StateDeps[ChatState]], proverbs: list[str]
) -> StateSnapshotEvent:
    ctx.deps.state.proverbs.extend(proverbs)
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state,
    )


@agent.tool
async def set_proverbs(
    ctx: RunContext[StateDeps[ChatState]], proverbs: list[str]
) -> StateSnapshotEvent:
    ctx.deps.state.proverbs = proverbs
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state,
    )


@agent.tool
def get_weather(_: RunContext[StateDeps[ChatState]], location: str) -> str:
    """Get the weather for a given location. Ensure location is fully spelled out."""
    return f"The weather in {location} is sunny."


@agent.tool
async def search_web(
    ctx: RunContext[StateDeps[ChatState]],
    query: str,
    search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "basic",
    topic: Literal["general", "news", "finance"] = "general",
    time_range: Literal["day", "week", "month", "year"] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    days: int | None = None,
    max_results: int = 10,
) -> StateSnapshotEvent:
    """
    Web search tool using Tavily API.

    Args:
        query: The search query string
        search_depth: "basic" or "advanced"
        topic: "general", "news", or "finance"
        time_range: "day", "week", "month" or "year" (optional)
        start_date: Beginning of date range yyyy-mm-dd
        end_date: End of date range yyyy-mm-dd
        days: Number of days back to search (optional)
        max_results: Maximum number of results
    """
    kwargs = {
        "search_depth": search_depth,
        "topic": topic,
        "time_range": time_range,
        "start_date": start_date,
        "end_date": end_date,
        "days": days,
        "max_results": max_results,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    response = _tavily_client.search(query, **cast(dict[str, Any], kwargs))

    ctx.deps.state.web_search_results.extend(response["results"])

    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state,
    )


@agent.tool
def get_web_search_results(
    ctx: RunContext[StateDeps[ChatState]],
) -> list[TavilySearchResult]:
    """Get the current list of web search results."""
    return ctx.deps.state.web_search_results


@agent.tool
async def get_account_info(
    _: RunContext[StateDeps[ChatState]],
) -> UnipileAccountResponse:
    """Get the user's Unipile account information."""
    return await Unipile.get_account()


@agent.tool
async def get_inmail_balance(
    _: RunContext[StateDeps[ChatState]],
) -> LinkedinInmailBalance:
    """Get the user's LinkedIn InMail balance."""
    return await Unipile.get_inmail_balance()


@agent.tool
async def get_search_parameters(
    ctx: RunContext[StateDeps[ChatState]],
    type: Literal[
        # CLASSIC types
        "LOCATION", "PEOPLE", "CONNECTIONS", "COMPANY", "SCHOOL",
        "INDUSTRY", "SERVICE", "JOB_FUNCTION", "JOB_TITLE", "EMPLOYMENT_TYPE", "SKILL",
        # SALES_NAVIGATOR types
        "GROUPS", "SALES_INDUSTRY", "DEPARTMENT", "PERSONA", "ACCOUNT_LISTS",
        "LEAD_LISTS", "TECHNOLOGIES", "SAVED_ACCOUNTS", "SAVED_SEARCHES",
        "RECENT_SEARCHES", "REGION", "POSTAL_CODE",
        # RECRUITER types
        "HIRING_PROJECTS", "SAVED_FILTERS", "DEGREE",
    ],
    service: Literal["CLASSIC", "SALES_NAVIGATOR", "RECRUITER"] = "CLASSIC",
    keywords: str | None = None,
    limit: int | None = None,
) -> StateSnapshotEvent:
    """
    Get LinkedIn search parameters (IDs) for use in search_linkedin.

    Use this to look up IDs for locations, companies, schools, industries, skills, etc.
    Results accumulate in state - call multiple times to gather all needed parameter IDs.

    Args:
        type: The type of parameter to search for. Available types depend on service:
            CLASSIC: LOCATION, PEOPLE, CONNECTIONS, COMPANY, SCHOOL, INDUSTRY, SERVICE,
                     JOB_FUNCTION, JOB_TITLE, EMPLOYMENT_TYPE, SKILL
            SALES_NAVIGATOR: GROUPS, SALES_INDUSTRY, DEPARTMENT, PERSONA, ACCOUNT_LISTS,
                             LEAD_LISTS, TECHNOLOGIES, SAVED_ACCOUNTS, SAVED_SEARCHES,
                             RECENT_SEARCHES, REGION, POSTAL_CODE
            RECRUITER: GROUPS, DEPARTMENT, HIRING_PROJECTS, SAVED_SEARCHES, SAVED_FILTERS, DEGREE
        service: The LinkedIn service to use (CLASSIC, SALES_NAVIGATOR, or RECRUITER)
        keywords: Keywords to search for within the parameter type
        limit: Maximum number of results (default 10, max 50 for CLASSIC, max 100 for SALES_NAVIGATOR/RECRUITER)
    """
    response = await Unipile.get_search_parameters(
        type=type,
        service=service,
        keywords=keywords,
        limit=limit,
    )
    ctx.deps.state.linkedin_search_parameters.extend(response.items)
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state,
    )


@agent.tool
async def search_linkedin(
    ctx: RunContext[StateDeps[ChatState]],
    api: Literal["classic", "sales_navigator", "recruiter"] = "sales_navigator",
    category: Literal["people", "companies", "jobs", "posts"] = "people",
    url: str | None = None,
    keywords: str | None = None,
    location_include: list[str] | None = None,
    school_include: list[str] | None = None,
    school_exclude: list[str] | None = None,
    company_include: list[str] | None = None,
    company_exclude: list[str] | None = None,
    industry_include: list[str] | None = None,
    industry_exclude: list[str] | None = None,
    company_headcount_min: int | None = None,
    company_headcount_max: int | None = None,
    tenure_min_years: int | None = None,
    profile_language: list[str] | None = None,
    has_job_offers: bool | None = None,
    network_distance: list[int] | None = None,
    role_include: list[str] | None = None,
    role_exclude: list[str] | None = None,
    cursor: str | None = None,
) -> StateSnapshotEvent:
    """
    Search LinkedIn for people, companies, jobs, or posts.

    Results accumulate in state. Use cursor parameter to paginate through results.
    Use get_search_parameters first to obtain IDs for location, company, school, etc.

    Args:
        api: LinkedIn API to use (classic, sales_navigator, recruiter). Defaults to sales_navigator.
        category: Type of search (people, companies, jobs, posts). Defaults to people.
        url: Alternative: provide a full LinkedIn search URL instead of parameters
        keywords: Search keywords
        location_include: List of location IDs (from get_search_parameters)
        school_include: List of school IDs to include
        school_exclude: List of school IDs to exclude
        company_include: List of company IDs to include
        company_exclude: List of company IDs to exclude
        industry_include: List of industry IDs to include
        industry_exclude: List of industry IDs to exclude
        company_headcount_min: Minimum company size
        company_headcount_max: Maximum company size
        tenure_min_years: Minimum years of tenure
        profile_language: List of profile languages (e.g., ["en", "fr"])
        has_job_offers: Filter for profiles with job offers
        network_distance: List of network distances (1, 2, 3)
        role_include: List of role keywords to include
        role_exclude: List of role keywords to exclude
        cursor: Cursor for pagination (from previous search response)
    """
    # Automatically set limit to maximum based on API type
    limit = 50 if api == "classic" else 100

    # Build the request object from individual parameters
    request_data: dict[str, Any] = {
        "api": api,
        "category": category,
        "limit": limit,
    }

    if url is not None:
        request_data["url"] = url
    if keywords is not None:
        request_data["keywords"] = keywords
    if cursor is not None:
        request_data["cursor"] = cursor
    if profile_language is not None:
        request_data["profile_language"] = profile_language
    if has_job_offers is not None:
        request_data["has_job_offers"] = has_job_offers
    if network_distance is not None:
        request_data["network_distance"] = network_distance

    # Build filter objects
    if location_include is not None:
        request_data["location"] = SearchFilterInclude(include=location_include)

    if school_include is not None or school_exclude is not None:
        request_data["school"] = SearchFilterInclude(
            include=school_include,
            exclude=school_exclude,
        )

    if company_include is not None or company_exclude is not None:
        request_data["company"] = SearchFilterInclude(
            include=company_include,
            exclude=company_exclude,
        )

    if industry_include is not None or industry_exclude is not None:
        request_data["industry"] = SearchFilterInclude(
            include=industry_include,
            exclude=industry_exclude,
        )

    if role_include is not None or role_exclude is not None:
        request_data["role"] = SearchFilterInclude(
            include=role_include,
            exclude=role_exclude,
        )

    if company_headcount_min is not None or company_headcount_max is not None:
        request_data["company_headcount"] = [
            CompanyHeadcount(min=company_headcount_min, max=company_headcount_max)
        ]

    if tenure_min_years is not None:
        request_data["tenure"] = [TenureFilter(min=tenure_min_years)]

    request = LinkedinSearchRequest(**request_data)
    response = await Unipile.search_linkedin(request)
    ctx.deps.state.linkedin_search_results.extend(response.items)

    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state,
    )


@agent.tool
def get_linkedin_search_results(
    ctx: RunContext[StateDeps[ChatState]],
) -> list[LinkedinSearchItem]:
    """Get the current list of LinkedIn search results."""
    return ctx.deps.state.linkedin_search_results


@agent.tool
async def clear_linkedin_search_results(
    ctx: RunContext[StateDeps[ChatState]],
) -> StateSnapshotEvent:
    """Clear the LinkedIn search results to start a new search."""
    ctx.deps.state.linkedin_search_results = []
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state,
    )


@agent.tool_plain
async def get_datetime() -> str:
    """Get the current time and date."""
    return datetime.now().astimezone().isoformat()
