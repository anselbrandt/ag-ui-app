import logging
from contextlib import asynccontextmanager

import logfire
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic_ai.ui.ag_ui import AGUIAdapter
from starlette.requests import Request
from starlette.responses import Response

from app.db import (
    DatabaseUnavailableError,
    RedisUnavailableError,
    close_db,
    close_redis,
    init_db,
    init_redis,
)

from .agent import ChatState, StateDeps, agent

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    db_initialized = False
    redis_initialized = False

    try:
        await init_db()
        db_initialized = True
        logger.info("Database initialized")
    except Exception as e:
        logger.error("Database initialization failed: %s", e)
        logfire.error("Database initialization failed", error=str(e))

    try:
        await init_redis()
        redis_initialized = True
        logger.info("Redis initialized")
    except Exception as e:
        logger.error("Redis initialization failed: %s", e)
        logfire.error("Redis initialization failed", error=str(e))

    if db_initialized and redis_initialized:
        logger.info("All connections initialized successfully")
    else:
        logger.warning("Application started with degraded functionality")
        if not db_initialized:
            logger.warning("Database: unavailable")
        if not redis_initialized:
            logger.warning("Redis: unavailable")

    yield

    logger.info("Shutting down...")
    if db_initialized:
        try:
            await close_db()
            logger.info("Database connection closed")
        except Exception as e:
            logger.error("Error closing database: %s", e)

    if redis_initialized:
        try:
            await close_redis()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error("Error closing Redis: %s", e)


app = FastAPI(lifespan=lifespan)

logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_fastapi(app)
logfire.instrument_httpx()
logfire.instrument_pydantic_ai()


@app.exception_handler(DatabaseUnavailableError)
async def database_unavailable_handler(
    request: Request, exc: DatabaseUnavailableError
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "Database unavailable. Please try again later."},
    )


@app.exception_handler(RedisUnavailableError)
async def redis_unavailable_handler(
    request: Request, exc: RedisUnavailableError
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "Redis unavailable. Please try again later."},
    )


@app.post("/")
async def run_agent(request: Request) -> Response:
    return await AGUIAdapter.dispatch_request(
        request, agent=agent, deps=StateDeps(ChatState())
    )
