import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from httpx import AsyncClient

from schemas import LinkedinProfileResponse

load_dotenv()

UNIPILE_DSN = os.getenv("UNIPILE_DSN", "")
UNIPILE_API_KEY = os.getenv("UNIPILE_API_KEY", "")
UNIPILE_ACCOUNT_ID = os.getenv("UNIPILE_ACCOUNT_ID", "")

public_identifier = "devonemacey"
provider_id = "ACoAAAIK3bMB5UKEWP8_CNjsdlzUn0SfCTvYJJA"

outdir = Path("output")
outdir.mkdir(exist_ok=True)
outpath = outdir / "linkedin_profile.json"


async def main(public_identifier_or_provider_id: str) -> LinkedinProfileResponse:
    url = f"{UNIPILE_DSN}/api/v1/users/{public_identifier_or_provider_id}"

    headers = {
        "X-API-KEY": UNIPILE_API_KEY,
        "accept": "application/json",
        "content-type": "application/json",
    }

    async with AsyncClient(timeout=60.0) as client:
        response = await client.get(
            url,
            headers=headers,
            params={"account_id": UNIPILE_ACCOUNT_ID, "linkedin_sections": "*"},
        )

        response.raise_for_status()

        data = response.json()
        profile = LinkedinProfileResponse.model_validate(data)

        with open(outpath, "w") as f:
            json.dump(profile.model_dump(exclude_none=True), f, indent=2)
        print(f"Response saved to {outpath}")

        return profile


if __name__ == "__main__":
    asyncio.run(main(public_identifier))
