from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic_ai.ui.ag_ui import AGUIAdapter
from starlette.requests import Request
from starlette.responses import Response
import logfire

from .agent import ChatState, StateDeps, agent
from app.db import close_db, close_redis, init_db, init_redis


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    db_initialized = False
    redis_initialized = False

    try:
        await init_db()
        db_initialized = True
        print("✓ Database initialized")
    except Exception as e:
        print(f"✗ Database initialization failed: {e}")
        logfire.error("Database initialization failed", error=str(e))

    try:
        await init_redis()
        redis_initialized = True
        print("✓ Redis initialized")
    except Exception as e:
        print(f"✗ Redis initialization failed: {e}")
        logfire.error("Redis initialization failed", error=str(e))

    if db_initialized and redis_initialized:
        print("✓ All connections initialized successfully")
    else:
        print("⚠ Application started with degraded functionality")
        if not db_initialized:
            print("  - Database: unavailable")
        if not redis_initialized:
            print("  - Redis: unavailable")

    yield

    print("Shutting down...")
    # Clean up connections that were successfully initialized
    if db_initialized:
        try:
            await close_db()
            print("✓ Database connection closed")
        except Exception as e:
            print(f"✗ Error closing database: {e}")

    if redis_initialized:
        try:
            await close_redis()
            print("✓ Redis connection closed")
        except Exception as e:
            print(f"✗ Error closing redis: {e}")


app = FastAPI(lifespan=lifespan)

logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_fastapi(app)
logfire.instrument_httpx()
logfire.instrument_pydantic_ai()


@app.post("/")
async def run_agent(request: Request) -> Response:
    return await AGUIAdapter.dispatch_request(
        request, agent=agent, deps=StateDeps(ChatState())
    )
