import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from httpx import AsyncClient

from schemas import LinkedinSearchParametersResponse, SearchParametersRequest

load_dotenv()

UNIPILE_DSN = os.getenv("UNIPILE_DSN", "")
UNIPILE_API_KEY = os.getenv("UNIPILE_API_KEY", "")
UNIPILE_ACCOUNT_ID = os.getenv("UNIPILE_ACCOUNT_ID", "")

outdir = Path("output")
outdir.mkdir(exist_ok=True)
outpath = outdir / "param_results_school.json"


async def main():
    url = f"{UNIPILE_DSN}/api/v1/linkedin/search/parameters"

    headers = {
        "X-API-KEY": UNIPILE_API_KEY,
        "accept": "application/json",
    }

    request = SearchParametersRequest(
        account_id=UNIPILE_ACCOUNT_ID,
        type="SCHOOL",
        limit=10,
        keywords="UBC",
    )

    async with AsyncClient() as client:
        response = await client.get(
            url,
            params=request.model_dump(exclude_none=True),
            headers=headers,
            timeout=30.0,
        )

        response.raise_for_status()

        data = LinkedinSearchParametersResponse.model_validate(response.json())

        with open(outpath, "w") as f:
            json.dump(data.model_dump(), f, indent=2)
        print(f"Response saved to {outpath}")


if __name__ == "__main__":
    asyncio.run(main())
