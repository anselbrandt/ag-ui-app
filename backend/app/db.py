import uuid
from typing import AsyncGenerator, Optional

import logfire
import redis.asyncio as redis
from dotenv import load_dotenv
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.config import settings
from app.models.base import Base

load_dotenv()

DATABASE_URL = settings.database_url
REDIS_URL = settings.redis_url

engine: Optional[AsyncEngine] = None
async_session: Optional[async_sessionmaker[AsyncSession]] = None
redis_client: Optional[redis.Redis] = None


async def create_tables(engine: AsyncEngine) -> None:
    """Create all tables defined in models if they don't exist"""

    models = []

    print(f"Checking and creating tables for {len(models)} models...")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        print("Tables created successfully (if they didn't exist)")


async def init_db() -> None:
    """Initialize database connection"""
    global engine, async_session

    try:
        engine = create_async_engine(
            DATABASE_URL,
            poolclass=NullPool,  # Required for Supabase
            echo=True,  # Set to False in production
            future=True,
            connect_args={  # important settings for asyncpg
                "prepared_statement_name_func": lambda: f"__asyncpg_{uuid.uuid4()}__",
                "statement_cache_size": 0,
                "prepared_statement_cache_size": 0,
            },
        )

        logfire.instrument_sqlalchemy(engine=engine)

        # Create tables first to ensure base schema exists
        await create_tables(engine)

        async_session = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

        print("Database connection initialized")
    except Exception as e:
        print(f"Failed to initialize database: {e}")
        if engine:
            await engine.dispose()
            engine = None
        async_session = None
        raise


async def init_redis() -> None:
    """Initialize Redis connection"""
    global redis_client

    try:
        logfire.instrument_redis()

        redis_client = redis.from_url(
            REDIS_URL, encoding="utf-8", decode_responses=True, health_check_interval=30
        )

        await redis_client.ping()
        print("Redis connection initialized")
    except Exception as e:
        print(f"Failed to initialize Redis: {e}")
        if redis_client:
            await redis_client.close()
            redis_client = None
        raise


async def close_db() -> None:
    """Close database connection"""
    global engine, async_session
    if engine:
        await engine.dispose()
        engine = None
        async_session = None
        print("Database connection closed")


async def close_redis() -> None:
    """Close Redis connection"""
    global redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None
        print("Redis connection closed")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session"""
    if not async_session:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable. Please check your database connection and try again.",
        )

    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


def get_session() -> async_sessionmaker[AsyncSession]:
    """Get session maker for background tasks"""
    if not async_session:
        raise RuntimeError("Database not initialized")
    return async_session


async def get_redis() -> redis.Redis:
    """Get Redis client"""
    if not redis_client:
        raise HTTPException(
            status_code=503,
            detail="Redis unavailable. Please check your Redis connection and try again.",
        )
    return redis_client
