from typing import Literal

from pydantic import BaseModel


# =============================================================================
# Profile Endpoint Models (/api/v1/users/{public_identifier_or_provider_id})
# =============================================================================


class ProfileContactSocial(BaseModel):
    """Social media link in contact info"""

    type: str | None = None
    name: str | None = None


class ProfileContactInfo(BaseModel):
    """Contact information for a LinkedIn profile"""

    emails: list[str] | None = None
    phones: list[str] | None = None
    adresses: list[str] | None = None  # Note: API uses "adresses" (typo in API)
    socials: list[ProfileContactSocial] | None = None


class ProfileBirthdate(BaseModel):
    """Birthdate information"""

    month: int | None = None
    day: int | None = None


class ProfilePrimaryLocale(BaseModel):
    """Primary locale settings"""

    country: str | None = None
    language: str | None = None


class ProfileCreatorWebsite(BaseModel):
    """Creator website information"""

    url: str | None = None
    description: str | None = None


class ProfileInvitation(BaseModel):
    """Connection invitation status"""

    type: Literal["SENT", "RECEIVED"] | str | None = None
    status: Literal["PENDING", "ACCEPTED", "DECLINED"] | str | None = None


class ProfileWorkExperience(BaseModel):
    """Work experience entry"""

    id: str | None = None
    position: str | None = None
    company_id: str | None = None
    company_url: str | None = None
    company_picture_url: str | None = None
    industry: list[str] | None = None
    company: str | None = None
    location: str | None = None
    description: str | None = None
    skills: list[str] | None = None
    current: bool | None = None
    status: str | None = None
    start: str | None = None
    end: str | None = None


class ProfileVolunteeringExperience(BaseModel):
    """Volunteering experience entry"""

    company: str | None = None
    description: str | None = None
    role: str | None = None
    cause: str | None = None
    start: str | None = None
    end: str | None = None


class ProfileEducation(BaseModel):
    """Education entry"""

    id: str | None = None
    degree: str | None = None
    grade: str | None = None
    description: str | None = None
    activities: str | None = None
    school: str | None = None
    school_id: str | None = None
    school_url: str | None = None
    school_picture_url: str | None = None
    skills: list[str] | None = None
    field_of_study: str | None = None
    start: str | None = None
    end: str | None = None


class ProfileSkill(BaseModel):
    """Skill entry on a profile"""

    name: str | None = None
    endorsement_count: int | None = None
    endorsement_id: int | None = None
    insights: list[str] | None = None
    endorsed: bool | None = None


class ProfileLanguage(BaseModel):
    """Language proficiency"""

    name: str | None = None
    proficiency: str | None = None


class ProfileCertification(BaseModel):
    """Certification entry"""

    name: str | None = None
    organization: str | None = None
    url: str | None = None


class ProfileProject(BaseModel):
    """Project entry"""

    name: str | None = None
    description: str | None = None
    skills: list[str] | None = None
    start: str | None = None
    end: str | None = None


class ProfileRecommendationActor(BaseModel):
    """Person who gave or received a recommendation"""

    first_name: str | None = None
    last_name: str | None = None
    provider_id: str | None = None
    headline: str | None = None
    public_identifier: str | None = None
    public_profile_url: str | None = None
    profile_picture_url: str | None = None


class ProfileRecommendation(BaseModel):
    """A recommendation entry"""

    text: str | None = None
    caption: str | None = None
    actor: ProfileRecommendationActor | None = None


class ProfileRecommendations(BaseModel):
    """Container for received and given recommendations"""

    received: list[ProfileRecommendation] | None = None
    received_total_count: int | None = None
    given: list[ProfileRecommendation] | None = None
    given_total_count: int | None = None


class ProfileTag(BaseModel):
    """Tag on a profile"""

    id: str | None = None
    name: str | None = None


class ProfileNoteAuthor(BaseModel):
    """Author of a note"""

    id: str | None = None
    seat_id: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    public_profile_url: bool | str | None = None  # API shows bool but could be string


class ProfileNote(BaseModel):
    """Note on a profile"""

    project_id: str | None = None
    content: str | None = None
    created_at: int | None = None
    author: ProfileNoteAuthor | None = None


class ProfileRecruitingActivityActor(BaseModel):
    """Actor in recruiting activity"""

    provider_id: str | None = None
    full_name: str | None = None
    picture_url: str | None = None


class ProfileRecruitingActivity(BaseModel):
    """Recruiting activity entry"""

    created_at: str | None = None
    created_by: ProfileRecruitingActivityActor | None = None
    updated_at: str | None = None
    updated_by: ProfileRecruitingActivityActor | None = None
    event: (
        Literal[
            "MESSAGE",
            "NOTE_ADDED",
            "LINK_ADDED",
            "CANDIDATE_ADDED",
            "PROFILE_VIEW",
            "UNHANDLED_EVENT",
        ]
        | str
        | None
    ) = None
    content: str | None = None
    status: Literal["PENDING", "ACCEPTED", "DECLINED"] | str | None = None
    project_id: str | None = None
    project_name: str | None = None
    state: str | None = None


class LinkedinProfileResponse(BaseModel):
    """Response from GET /api/v1/users/{public_identifier_or_provider_id}"""

    object: str | None = None
    provider: Literal["LINKEDIN"] | str | None = None
    provider_id: str | None = None
    public_identifier: str | None = None
    member_urn: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    headline: str | None = None
    summary: str | None = None
    contact_info: ProfileContactInfo | None = None
    birthdate: ProfileBirthdate | None = None
    primary_locale: ProfilePrimaryLocale | None = None
    location: str | None = None
    websites: list[str] | None = None
    creator_website: ProfileCreatorWebsite | None = None
    profile_picture_url: str | None = None
    profile_picture_url_large: str | None = None
    background_picture_url: str | None = None
    hashtags: list[str] | None = None
    can_send_inmail: bool | None = None
    is_open_profile: bool | None = None
    is_premium: bool | None = None
    is_influencer: bool | None = None
    is_creator: bool | None = None
    is_hiring: bool | None = None
    is_open_to_work: bool | None = None
    is_saved_lead: bool | None = None
    is_crm_imported: bool | None = None
    is_relationship: bool | None = None
    is_self: bool | None = None
    invitation: ProfileInvitation | None = None
    work_experience: list[ProfileWorkExperience] | None = None
    work_experience_total_count: int | None = None
    volunteering_experience: list[ProfileVolunteeringExperience] | None = None
    volunteering_experience_total_count: int | None = None
    education: list[ProfileEducation] | None = None
    education_total_count: int | None = None
    skills: list[ProfileSkill] | None = None
    skills_total_count: int | None = None
    languages: list[ProfileLanguage] | None = None
    languages_total_count: int | None = None
    certifications: list[ProfileCertification] | None = None
    certifications_total_count: int | None = None
    projects: list[ProfileProject] | None = None
    projects_total_count: int | None = None
    recommendations: ProfileRecommendations | None = None
    tags: list[ProfileTag] | None = None
    notes: list[ProfileNote] | None = None
    recruiting_activity: list[ProfileRecruitingActivity] | None = None
    throttled_sections: list[str] | None = None
    follower_count: int | None = None
    connections_count: int | None = None
    shared_connections_count: int | None = None
    network_distance: (
        Literal["FIRST_DEGREE", "SECOND_DEGREE", "THIRD_DEGREE", "OUT_OF_NETWORK"]
        | str
        | None
    ) = None
    public_profile_url: str | None = None


# =============================================================================
# Search Parameters Endpoint Models (/api/v1/linkedin/search/parameters)
# =============================================================================


class SearchParametersRequest(BaseModel):
    """Query parameters for GET /api/v1/linkedin/search/parameters"""

    account_id: str
    type: Literal["LOCATION", "COMPANY", "SCHOOL", "INDUSTRY", "SKILL"] | str
    keywords: str | None = None
    limit: int | None = None


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
