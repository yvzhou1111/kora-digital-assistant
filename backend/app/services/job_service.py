from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_
import uuid

from app.models.job import Job
from app.models.job_application import JobApplication
from app.models.employer_profile import EmployerProfile
from app.models.individual_profile import IndividualProfile
from app.schemas.job import JobCreate, JobUpdate, JobApplicationCreate


def get_job(db: Session, job_id: uuid.UUID) -> Optional[Job]:
    """获取职位"""
    return db.query(Job).filter(Job.id == job_id).first()


def get_jobs(
    db: Session, 
    skip: int = 0, 
    limit: int = 100,
    employer_id: Optional[uuid.UUID] = None,
    status: Optional[str] = None,
    location: Optional[str] = None,
    employment_type: Optional[str] = None,
    experience_level: Optional[str] = None,
    search: Optional[str] = None
) -> List[Job]:
    """获取职位列表"""
    query = db.query(Job)
    
    # 筛选条件
    if employer_id:
        query = query.filter(Job.employer_id == employer_id)
    
    if status:
        query = query.filter(Job.status == status)
    else:
        # 默认只显示活跃的职位
        query = query.filter(Job.status == "active")
    
    if location:
        query = query.filter(Job.location.ilike(f"%{location}%"))
    
    if employment_type:
        query = query.filter(Job.employment_type == employment_type)
    
    if experience_level:
        query = query.filter(Job.experience_level == experience_level)
    
    if search:
        query = query.filter(
            or_(
                Job.title.ilike(f"%{search}%"),
                Job.description.ilike(f"%{search}%")
            )
        )
    
    # 排序和分页
    return query.order_by(Job.created_at.desc()).offset(skip).limit(limit).all()


def create_job(db: Session, job_in: JobCreate, employer_id: uuid.UUID) -> Job:
    """创建职位"""
    db_job = Job(
        employer_id=employer_id,
        title=job_in.title,
        description=job_in.description,
        requirements=job_in.requirements,
        responsibilities=job_in.responsibilities,
        location=job_in.location,
        salary_min=job_in.salary_min,
        salary_max=job_in.salary_max,
        salary_currency=job_in.salary_currency,
        employment_type=job_in.employment_type,
        experience_level=job_in.experience_level,
        status=job_in.status or "draft",
        deadline=job_in.deadline
    )
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job


def update_job(db: Session, job: Job, job_in: JobUpdate) -> Job:
    """更新职位"""
    update_data = job_in.dict(exclude_unset=True)
    
    for field, value in update_data.items():
        if hasattr(job, field) and value is not None:
            setattr(job, field, value)
    
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def delete_job(db: Session, job: Job) -> None:
    """删除职位"""
    db.delete(job)
    db.commit()


def apply_for_job(db: Session, job_id: uuid.UUID, user_id: uuid.UUID, application_in: JobApplicationCreate) -> JobApplication:
    """申请职位"""
    # 获取个人资料ID
    individual_profile = db.query(IndividualProfile).filter(IndividualProfile.user_id == user_id).first()
    if not individual_profile:
        raise ValueError("用户没有个人资料")
    
    db_application = JobApplication(
        job_id=job_id,
        individual_id=individual_profile.user_id,
        cover_letter=application_in.cover_letter,
        resume_url=application_in.resume_url,
        use_digital_twin=application_in.use_digital_twin,
        notes=application_in.notes,
        status="applied"
    )
    db.add(db_application)
    db.commit()
    db.refresh(db_application)
    return db_application


def get_job_applications(
    db: Session, 
    job_id: uuid.UUID, 
    skip: int = 0, 
    limit: int = 100,
    status: Optional[str] = None
) -> List[JobApplication]:
    """获取职位申请列表"""
    query = db.query(JobApplication).filter(JobApplication.job_id == job_id)
    
    if status:
        query = query.filter(JobApplication.status == status)
    
    return query.order_by(JobApplication.created_at.desc()).offset(skip).limit(limit).all()


def get_user_job_applications(
    db: Session, 
    user_id: uuid.UUID, 
    job_id: Optional[uuid.UUID] = None,
    skip: int = 0, 
    limit: int = 100,
    status: Optional[str] = None
) -> List[JobApplication]:
    """获取用户的职位申请"""
    # 获取个人资料ID
    individual_profile = db.query(IndividualProfile).filter(IndividualProfile.user_id == user_id).first()
    if not individual_profile:
        return []
    
    query = db.query(JobApplication).filter(JobApplication.individual_id == individual_profile.user_id)
    
    if job_id:
        query = query.filter(JobApplication.job_id == job_id)
    
    if status:
        query = query.filter(JobApplication.status == status)
    
    return query.order_by(JobApplication.created_at.desc()).offset(skip).limit(limit).all()


def update_job_application_status(db: Session, application_id: uuid.UUID, status: str, notes: Optional[str] = None) -> JobApplication:
    """更新职位申请状态"""
    application = db.query(JobApplication).filter(JobApplication.id == application_id).first()
    if not application:
        raise ValueError("职位申请不存在")
    
    application.status = status
    if notes:
        application.notes = notes
    
    db.add(application)
    db.commit()
    db.refresh(application)
    return application 