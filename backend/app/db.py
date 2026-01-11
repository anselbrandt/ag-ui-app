import logging
import uuid
from typing import AsyncGenerator

import logfire
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.config import settings
from app.models.base import Base

logger = logging.getLogger(__name__)


class DatabaseUnavailableError(Exception):
    """Raised when database connection is not available."""


class RedisUnavailableError(Exception):
    """Raised when Redis connection is not available."""


engine: AsyncEngine | None = None
async_session: async_sessionmaker[AsyncSession] | None = None
redis_client: redis.Redis | None = None


async def create_tables(engine: AsyncEngine) -> None:
    """Create all tables defined in models if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created successfully")


async def init_db() -> None:
    """Initialize database connection."""
    global engine, async_session

    try:
        engine = create_async_engine(
            settings.database_url,
            poolclass=NullPool,
            echo=settings.db_echo,
            future=True,
            connect_args={
                "prepared_statement_name_func": lambda: f"__asyncpg_{uuid.uuid4()}__",
                "statement_cache_size": 0,
                "prepared_statement_cache_size": 0,
            },
        )

        logfire.instrument_sqlalchemy(engine=engine)

        await create_tables(engine)

        async_session = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

        logger.info("Database connection initialized")
    except Exception as e:
        logger.error("Failed to initialize database: %s", e)
        if engine:
            await engine.dispose()
            engine = None
        async_session = None
        raise


async def init_redis() -> None:
    """Initialize Redis connection."""
    global redis_client

    try:
        logfire.instrument_redis()

        redis_client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            health_check_interval=30,
        )

        await redis_client.ping()
        logger.info("Redis connection initialized")
    except Exception as e:
        logger.error("Failed to initialize Redis: %s", e)
        if redis_client:
            await redis_client.close()
            redis_client = None
        raise


async def close_db() -> None:
    """Close database connection."""
    global engine, async_session
    if engine:
        await engine.dispose()
        engine = None
        async_session = None
        logger.info("Database connection closed")


async def close_redis() -> None:
    """Close Redis connection."""
    global redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None
        logger.info("Redis connection closed")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    if not async_session:
        raise DatabaseUnavailableError("Database connection not initialized")

    async with async_session() as session:
        yield session


def get_session() -> async_sessionmaker[AsyncSession]:
    """Get session maker for background tasks."""
    if not async_session:
        raise DatabaseUnavailableError("Database connection not initialized")
    return async_session


async def get_redis() -> redis.Redis:
    """Get Redis client."""
    if not redis_client:
        raise RedisUnavailableError("Redis connection not initialized")
    return redis_client
