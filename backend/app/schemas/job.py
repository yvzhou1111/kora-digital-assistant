from typing import Optional, List, Dict, Any
from datetime import datetime, date
from pydantic import BaseModel, UUID4, validator, Field


class JobBase(BaseModel):
    """职位基础模型"""
    title: Optional[str] = None
    description: Optional[str] = None
    requirements: Optional[List[str]] = None
    responsibilities: Optional[List[str]] = None
    location: Optional[str] = None
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    salary_currency: Optional[str] = None
    employment_type: Optional[str] = None
    experience_level: Optional[str] = None
    status: Optional[str] = None
    deadline: Optional[date] = None


class JobCreate(JobBase):
    """职位创建模型"""
    title: str
    description: str
    
    @validator("status")
    def validate_status(cls, v):
        if v is None:
            return "draft"
        allowed_statuses = ["draft", "active", "closed", "archived"]
        if v not in allowed_statuses:
            raise ValueError(f"状态必须是以下之一: {', '.join(allowed_statuses)}")
        return v


class JobUpdate(JobBase):
    """职位更新模型"""
    pass


class JobInDBBase(JobBase):
    """数据库中的职位基础模型"""
    id: UUID4
    employer_id: UUID4
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True


class Job(JobInDBBase):
    """API响应中的职位模型"""
    employer_name: Optional[str] = None
    application_count: Optional[int] = None


class JobApplicationBase(BaseModel):
    """职位申请基础模型"""
    cover_letter: Optional[str] = None
    resume_url: Optional[str] = None
    use_digital_twin: Optional[bool] = False
    notes: Optional[str] = None


class JobApplicationCreate(JobApplicationBase):
    """职位申请创建模型"""
    pass


class JobApplicationUpdate(JobApplicationBase):
    """职位申请更新模型"""
    status: Optional[str] = None
    
    @validator("status")
    def validate_status(cls, v):
        if v is None:
            return None
        allowed_statuses = ["applied", "reviewed", "interviewing", "offered", "rejected", "withdrawn", "hired"]
        if v not in allowed_statuses:
            raise ValueError(f"状态必须是以下之一: {', '.join(allowed_statuses)}")
        return v


class JobApplicationInDBBase(JobApplicationBase):
    """数据库中的职位申请基础模型"""
    id: UUID4
    job_id: UUID4
    individual_id: UUID4
    status: str
    match_score: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True


class JobApplication(JobApplicationInDBBase):
    """API响应中的职位申请模型"""
    job_title: Optional[str] = None
    applicant_name: Optional[str] = None 