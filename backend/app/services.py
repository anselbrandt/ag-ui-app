import logfire
from httpx import AsyncClient, HTTPStatusError
from app.config import settings
from app.schemas import (
    UnipileAccountResponse,
    LinkedinInmailBalance,
    SearchParametersRequest,
    LinkedinSearchParametersResponse,
    LinkedinSearchRequest,
    LinkedinSearchResponse,
    LinkedinProfileResponse,
)


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

    @staticmethod
    async def get_search_parameters(
        type: str,
        service: str = "CLASSIC",
        keywords: str | None = None,
        limit: int | None = None,
    ) -> LinkedinSearchParametersResponse:
        request = SearchParametersRequest(
            account_id=Unipile._ACCOUNT_ID,
            service=service,
            type=type,
            keywords=keywords,
            limit=limit,
        )
        data = await Unipile._request(
            "get",
            "/api/v1/linkedin/search/parameters",
            params=request.model_dump(exclude_none=True),
        )
        return LinkedinSearchParametersResponse.model_validate(data)

    @staticmethod
    async def search_linkedin(
        request: LinkedinSearchRequest,
    ) -> LinkedinSearchResponse:
        headers = {
            **Unipile._HEADERS,
            "content-type": "application/json",
        }
        # limit must be passed as a query parameter, not in the request body
        params: dict[str, str | int] = {"account_id": Unipile._ACCOUNT_ID}
        if request.limit is not None:
            params["limit"] = request.limit
        data = await Unipile._request(
            "post",
            "/api/v1/linkedin/search",
            headers=headers,
            params=params,
            json=request.model_dump(exclude_none=True, exclude={"limit"}),
        )
        return LinkedinSearchResponse.model_validate(data)

    @staticmethod
    async def get_linkedin_profile(
        public_identifier_or_provider_id: str,
    ) -> LinkedinProfileResponse:
        data = await Unipile._request(
            "get",
            f"/api/v1/users/{public_identifier_or_provider_id}",
            params={"account_id": Unipile._ACCOUNT_ID, "linkedin_sections": "*"},
        )
        return LinkedinProfileResponse.model_validate(data)
