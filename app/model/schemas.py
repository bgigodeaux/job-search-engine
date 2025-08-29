from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Experience(BaseModel):
    company: str
    role: str
    start_date: str
    end_date: Optional[str] = None
    description: str


class Education(BaseModel):
    institution: str
    degree: str
    year_of_graduation: int
    description: str


class RawCandidate(BaseModel):
    first_name: str
    last_name: str
    email: str
    skills: List[str]
    experiences: List[Experience]
    education: List[Education]
    # Allow other fields from the original JSON
    extra_data: Dict[str, Any] = Field(default_factory=dict, alias="extra")

    class Config:
        # Allows populating extra_data with fields not explicitly defined
        extra = "allow"


class RawJob(BaseModel):
    job_title: str
    job_description: str
    required_skills: List[str]
    company_name: str
    location: str
    # Allow other fields
    extra_data: Dict[str, Any] = Field(default_factory=dict, alias="extra")

    class Config:
        extra = "allow"


class EngineeredCandidateFeatures(BaseModel):
    total_years_of_experience: float
    seniority_level: str
    education_level: str
    skill_keywords: List[str]
    candidate_summary: str


class EngineeredJobFeatures(BaseModel):
    extracted_skills: List[str]
    seniority_level: str
    required_experience_years: float
    job_summary_for_embedding: str


class ProcessedCandidate(BaseModel):
    original_data: RawCandidate
    engineered_features: EngineeredCandidateFeatures
    embedding: List[float]


class ProcessedJob(BaseModel):
    original_data: RawJob
    engineered_features: EngineeredJobFeatures
    embedding: List[float]


class RankedCandidate(BaseModel):
    candidate: ProcessedCandidate
    score: float
