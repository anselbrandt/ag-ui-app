from fastapi import FastAPI
from pydantic_ai.ui.ag_ui import AGUIAdapter
from starlette.requests import Request
from starlette.responses import Response
import logfire

from .agent import ProverbsState, StateDeps, agent

app = FastAPI()

logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_fastapi(app)
logfire.instrument_httpx()
logfire.instrument_pydantic_ai()


@app.post("/")
async def run_agent(request: Request) -> Response:
    return await AGUIAdapter.dispatch_request(
        request, agent=agent, deps=StateDeps(ProverbsState())
    )
