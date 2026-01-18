import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from httpx import AsyncClient

from schemas import (
    LinkedinSearchRequest,
    LinkedinSearchResponse,
    SearchFilterInclude,
)

load_dotenv()

UNIPILE_DSN = os.getenv("UNIPILE_DSN", "")
UNIPILE_API_KEY = os.getenv("UNIPILE_API_KEY", "")
UNIPILE_ACCOUNT_ID = os.getenv("UNIPILE_ACCOUNT_ID", "")

outdir = Path("output")
outdir.mkdir(exist_ok=True)
outpath = outdir / "search_results.json"


async def main():
    url = f"{UNIPILE_DSN}/api/v1/linkedin/search"

    headers = {
        "X-API-KEY": UNIPILE_API_KEY,
        "accept": "application/json",
        "content-type": "application/json",
    }

    request = LinkedinSearchRequest(
        api="sales_navigator",
        category="people",
        location=SearchFilterInclude(include=["104116203"]),
        school=SearchFilterInclude(include=["4373"]),
        company=SearchFilterInclude(include=["1480"]),
    )

    async with AsyncClient() as client:
        response = await client.post(
            url,
            headers=headers,
            params={"account_id": UNIPILE_ACCOUNT_ID},
            json=request.model_dump(exclude_none=True),
        )

        response.raise_for_status()

        data = LinkedinSearchResponse.model_validate(response.json())

        with open(outpath, "w") as f:
            json.dump(data.model_dump(), f, indent=2)
        print(f"Response saved to {outpath}")


if __name__ == "__main__":
    asyncio.run(main())
