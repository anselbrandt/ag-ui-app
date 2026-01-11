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
