from textwrap import dedent
from typing import Any
import os
import urllib.parse

from ag_ui.core import EventType, StateSnapshotEvent
from httpx import AsyncClient
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.ag_ui import StateDeps
from pydantic_ai.models.openai import OpenAIResponsesModel
import logfire

# load environment variables
from dotenv import load_dotenv

load_dotenv()

logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()


# =====
# State
# =====
class Deps(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    proverbs: list[str] = Field(
        default_factory=list,
        description="The list of already written proverbs",
    )
    client: AsyncClient = AsyncClient()
    weather_api_key: str | None = os.getenv("WEATHER_API_KEY")
    geo_api_key: str | None = os.getenv("GEO_API_KEY")


# =====
# Agent
# =====
agent = Agent(
    model=OpenAIResponsesModel("gpt-4.1-mini"),
    deps_type=StateDeps[Deps],
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
def get_proverbs(ctx: RunContext[StateDeps[Deps]]) -> list[str]:
    """Get the current list of proverbs."""
    print(f"📖 Getting proverbs: {ctx.deps.state.proverbs}")
    return ctx.deps.state.proverbs


@agent.tool
async def add_proverbs(
    ctx: RunContext[StateDeps[Deps]], proverbs: list[str]
) -> StateSnapshotEvent:
    ctx.deps.state.proverbs.extend(proverbs)
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state,
    )


@agent.tool
async def set_proverbs(
    ctx: RunContext[StateDeps[Deps]], proverbs: list[str]
) -> StateSnapshotEvent:
    ctx.deps.state.proverbs = proverbs
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state,
    )


@agent.tool
async def get_lat_lng(
    ctx: RunContext[StateDeps[Deps]], location_description: str
) -> dict[str, float]:
    """Get the latitude and longitude of a location.

    Args:
        ctx: The context.
        location_description: A description of a location.
    """
    if ctx.deps.state.geo_api_key is None:
        # if no API key is provided, return a dummy response (London)
        return {"lat": 51.1, "lng": -0.1}

    params = {"access_token": ctx.deps.state.geo_api_key}
    loc = urllib.parse.quote(location_description)
    r = await ctx.deps.state.client.get(
        f"https://api.mapbox.com/geocoding/v5/mapbox.places/{loc}.json", params=params
    )
    r.raise_for_status()
    data = r.json()

    if features := data["features"]:
        lng, lat = features[0]["center"]
        return {"lat": lat, "lng": lng}
    else:
        raise ModelRetry("Could not find the location")


@agent.tool
async def get_weather(
    ctx: RunContext[StateDeps[Deps]], lat: float, lng: float
) -> dict[str, Any]:
    """Get the weather at a location.

    Args:
        ctx: The context.
        lat: Latitude of the location.
        lng: Longitude of the location.
    """
    if ctx.deps.state.weather_api_key is None:
        # if no API key is provided, return a dummy response
        return {"temperature": "21 °C", "description": "Sunny"}

    params = {
        "apikey": ctx.deps.state.weather_api_key,
        "location": f"{lat},{lng}",
        "units": "metric",
    }
    with logfire.span("calling weather API", params=params) as span:
        r = await ctx.deps.state.client.get(
            "https://api.tomorrow.io/v4/weather/realtime", params=params
        )
        r.raise_for_status()
        data = r.json()
        span.set_attribute("response", data)

    values = data["data"]["values"]
    # https://docs.tomorrow.io/reference/data-layers-weather-codes
    code_lookup = {
        1000: "Clear, Sunny",
        1100: "Mostly Clear",
        1101: "Partly Cloudy",
        1102: "Mostly Cloudy",
        1001: "Cloudy",
        2000: "Fog",
        2100: "Light Fog",
        4000: "Drizzle",
        4001: "Rain",
        4200: "Light Rain",
        4201: "Heavy Rain",
        5000: "Snow",
        5001: "Flurries",
        5100: "Light Snow",
        5101: "Heavy Snow",
        6000: "Freezing Drizzle",
        6001: "Freezing Rain",
        6200: "Light Freezing Rain",
        6201: "Heavy Freezing Rain",
        7000: "Ice Pellets",
        7101: "Heavy Ice Pellets",
        7102: "Light Ice Pellets",
        8000: "Thunderstorm",
    }
    return {
        "temperature": f"{values['temperatureApparent']:0.0f}°C",
        "description": code_lookup.get(values["weatherCode"], "Unknown"),
    }
