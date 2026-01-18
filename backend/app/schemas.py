from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Organization(BaseModel):
    name: str
    messaging_enabled: bool
    mailbox_urn: str
    organization_urn: str


class LinkedInIM(BaseModel):
    model_config = {"populate_by_name": True}

    id: str
    public_identifier: str = Field(alias="publicIdentifier")
    username: str
    premium_id: str = Field(alias="premiumId")
    premium_features: list[str] = Field(alias="premiumFeatures")
    premium_contract_id: str = Field(alias="premiumContractId")
    organizations: list[Organization]


class ConnectionParams(BaseModel):
    im: LinkedInIM


class Source(BaseModel):
    id: str
    status: str


class UnipileAccountResponse(BaseModel):
    object: Literal["Account"]
    connection_params: ConnectionParams
    name: str
    type: str
    created_at: datetime
    sources: list[Source]
    id: str
    groups: list  # Empty in example, update type when known


class LinkedinInmailBalance(BaseModel):
    object: Literal["LinkedinInmailBalance"]
    premium: int | None = None
    recruiter: int | None = None
    sales_navigator: int | None = None


# =============================================================================
# Search Parameters Endpoint Models (/api/v1/linkedin/search/parameters)
# =============================================================================


class SearchParametersRequest(BaseModel):
    """Query parameters for GET /api/v1/linkedin/search/parameters"""

    account_id: str
    service: Literal["CLASSIC", "SALES_NAVIGATOR", "RECRUITER"] | str = "CLASSIC"
    type: Literal[
        # CLASSIC types
        "LOCATION", "PEOPLE", "CONNECTIONS", "COMPANY", "SCHOOL",
        "INDUSTRY", "SERVICE", "JOB_FUNCTION", "JOB_TITLE", "EMPLOYMENT_TYPE", "SKILL",
        # SALES_NAVIGATOR types
        "GROUPS", "SALES_INDUSTRY", "DEPARTMENT", "PERSONA", "ACCOUNT_LISTS",
        "LEAD_LISTS", "TECHNOLOGIES", "SAVED_ACCOUNTS", "SAVED_SEARCHES",
        "RECENT_SEARCHES", "REGION", "POSTAL_CODE",
        # RECRUITER types (GROUPS, DEPARTMENT, SAVED_SEARCHES shared with SALES_NAVIGATOR)
        "HIRING_PROJECTS", "SAVED_FILTERS", "DEGREE",
    ] | str
    keywords: str | None = None
    limit: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Max results (default 10, max 50 for Classic, max 100 for Sales Navigator/Recruiter)",
    )


class LinkedinSearchParameter(BaseModel):
    """Individual search parameter result"""

    object: str
    title: str
    id: str
    picture_url: str | None = None


class Paging(BaseModel):
    page_count: int


class LinkedinSearchParametersResponse(BaseModel):
    """Response from GET /api/v1/linkedin/search/parameters"""

    object: str
    items: list[LinkedinSearchParameter]
    paging: Paging


# =============================================================================
# Search Endpoint Models (/api/v1/linkedin/search)
# =============================================================================


class Tenure(BaseModel):
    years: int | None = None
    months: int | None = None


class DateInfo(BaseModel):
    month: int | None = None
    year: int | None = None


class Position(BaseModel):
    company: str | None = None
    company_id: str | None = None
    description: str | None = None
    location: str | None = None
    industry: list[str] | None = None
    role: str | None = None
    tenure_at_company: Tenure | None = None
    tenure_at_role: Tenure | None = None
    skills: list[str] | None = None
    company_url: str | None = None
    start: DateInfo | None = None


class LinkedinSearchItem(BaseModel):
    type: str
    id: str
    industry: str | None = None
    name: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    member_urn: str | None = None
    public_identifier: str | None = None
    public_profile_url: str | None = None
    profile_url: str | None = None
    profile_picture_url: str | None = None
    profile_picture_url_large: str | None = None
    network_distance: str | None = None
    location: str | None = None
    headline: str | None = None
    summary: str | None = None
    pending_invitation: bool | None = None
    current_positions: list[Position] | None = None
    open_profile: bool | None = None
    premium: bool | None = None
    shared_connections_count: int | None = None


class SearchFilterInclude(BaseModel):
    include: list[str] | None = None
    exclude: list[str] | None = None


class TenureFilter(BaseModel):
    min: int | None = None


class RoleFilter(BaseModel):
    keywords: str | None = None
    priority: str | None = None  # "MUST_HAVE" | "DOESNT_HAVE"
    scope: str | None = None  # "CURRENT_OR_PAST"


class SkillFilter(BaseModel):
    id: str | None = None
    priority: str | None = None  # "MUST_HAVE" | "DOESNT_HAVE"


class CompanyHeadcount(BaseModel):
    min: int | None = None
    max: int | None = None


class LinkedinSearchRequest(BaseModel):
    """Request body for POST /api/v1/linkedin/search"""

    api: Literal["classic", "sales_navigator", "recruiter"] | str | None = None
    category: Literal["people", "companies", "jobs", "posts"] | str | None = None
    url: str | None = None  # Alternative: provide full LinkedIn search URL
    keywords: str | None = None
    location: SearchFilterInclude | None = None
    school: SearchFilterInclude | None = None
    company: SearchFilterInclude | None = None
    industry: SearchFilterInclude | None = None
    company_headcount: list[CompanyHeadcount] | None = None
    tenure: list[TenureFilter] | None = None
    profile_language: list[str] | None = None
    has_job_offers: bool | None = None
    network_distance: list[int] | None = None
    role: SearchFilterInclude | None = None
    skills: list[SkillFilter] | None = None
    cursor: str | None = None  # Cursor for pagination from previous search response
    limit: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Max results per page (default 25, max 50 for classic, max 100 for sales_navigator/recruiter)",
    )


class SearchConfig(BaseModel):
    """Echo of search request parameters in response"""

    params: LinkedinSearchRequest | None = None


class SearchPaging(BaseModel):
    start: int | None = None
    page_count: int | None = None
    total_count: int | None = None


class LinkedinSearchResponse(BaseModel):
    object: str
    items: list[LinkedinSearchItem]
    config: SearchConfig | None = None
    paging: SearchPaging | None = None
    cursor: str | None = None
