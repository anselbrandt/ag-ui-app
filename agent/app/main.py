from dataclasses import replace

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic_ai.ui.ag_ui import AGUIAdapter

from .agent import Deps, StateDeps, agent


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/")
async def run_agent(request: Request):
    """Handle AG-UI requests by running the agent."""
    # Create a fresh copy of deps for each request to avoid state pollution
    deps = replace(StateDeps(Deps()))
    return await AGUIAdapter.dispatch_request(
        request,
        agent=agent,
        deps=deps,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
