from httpx import AsyncClient
from app.config import settings
from app.schemas import UnipileAccountResponse, LinkedinInmailBalance


class Unipile:
    _BASE_URL = settings.unipile_dsn
    _ACCOUNT_ID = settings.unipile_account_id
    _HEADERS = {
        "X-API-KEY": settings.unipile_api_key.get_secret_value(),
        "accept": "application/json",
    }

    @staticmethod
    async def _request(method: str, endpoint: str, **kwargs) -> dict:
        url = f"{Unipile._BASE_URL}{endpoint}"
        kwargs.setdefault("headers", Unipile._HEADERS)
        kwargs.setdefault("timeout", 30.0)

        async with AsyncClient() as client:
            response = await getattr(client, method)(url, **kwargs)
            response.raise_for_status()
            return response.json()

    @staticmethod
    async def get_account() -> UnipileAccountResponse:
        data = await Unipile._request(
            "get",
            f"/api/v1/accounts/{Unipile._ACCOUNT_ID}",
            params={"account_id": Unipile._ACCOUNT_ID, "linkedin_sections": "*"},
        )
        return UnipileAccountResponse(**data)

    @staticmethod
    async def get_inmail_balance() -> LinkedinInmailBalance:
        data = await Unipile._request(
            "get",
            "/api/v1/linkedin/inmail_balance",
            params={"account_id": Unipile._ACCOUNT_ID},
        )
        return LinkedinInmailBalance(**data)
