from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
import uuid

from app.core.deps import get_db, get_current_active_user
from app.models.user import User
from app.models.job import Job
from app.models.job_application import JobApplication
from app.schemas.job import (
    Job as JobSchema,
    JobCreate,
    JobUpdate,
    JobApplication as JobApplicationSchema,
    JobApplicationCreate
)
from app.services.job_service import (
    get_job,
    get_jobs,
    create_job,
    update_job,
    delete_job,
    apply_for_job,
    get_job_applications,
    get_user_job_applications,
    update_job_application_status
)

router = APIRouter()


@router.post("", response_model=JobSchema)
async def create_new_job(
    job_in: JobCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    创建新职位
    """
    # 检查用户是否为雇主
    if current_user.role != "employer":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有雇主可以创建职位"
        )
    
    return create_job(db, job_in, current_user.id)


@router.get("", response_model=List[JobSchema])
async def read_jobs(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    location: Optional[str] = None,
    employment_type: Optional[str] = None,
    experience_level: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    获取职位列表
    """
    return get_jobs(
        db, 
        skip=skip, 
        limit=limit,
        status=status,
        location=location,
        employment_type=employment_type,
        experience_level=experience_level,
        search=search
    )


@router.get("/my-listings", response_model=List[JobSchema])
async def read_my_job_listings(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    获取当前雇主的职位列表
    """
    # 检查用户是否为雇主
    if current_user.role != "employer":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有雇主可以查看自己的职位列表"
        )
    
    return get_jobs(
        db, 
        employer_id=current_user.id, 
        skip=skip, 
        limit=limit,
        status=status
    )


@router.get("/{job_id}", response_model=JobSchema)
async def read_job(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    获取特定职位
    """
    job = get_job(db, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="职位不存在"
        )
    return job


@router.put("/{job_id}", response_model=JobSchema)
async def update_job_details(
    job_id: uuid.UUID,
    job_in: JobUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    更新职位
    """
    # 检查用户是否为雇主
    if current_user.role != "employer":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有雇主可以更新职位"
        )
    
    job = get_job(db, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="职位不存在"
        )
    
    # 检查是否为职位创建者
    if str(job.employer_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有职位创建者可以更新职位"
        )
    
    return update_job(db, job, job_in)


@router.delete("/{job_id}")
async def delete_job_listing(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    删除职位
    """
    # 检查用户是否为雇主
    if current_user.role != "employer":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有雇主可以删除职位"
        )
    
    job = get_job(db, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="职位不存在"
        )
    
    # 检查是否为职位创建者
    if str(job.employer_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有职位创建者可以删除职位"
        )
    
    delete_job(db, job)
    return {"message": "职位已删除"}


@router.post("/{job_id}/apply", response_model=JobApplicationSchema)
async def apply_to_job(
    job_id: uuid.UUID,
    application_in: JobApplicationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    申请职位
    """
    # 检查用户是否为个人用户
    if current_user.role != "individual":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有个人用户可以申请职位"
        )
    
    job = get_job(db, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="职位不存在"
        )
    
    # 检查职位是否开放
    if job.status != "active":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="该职位不接受申请"
        )
    
    # 检查是否已申请
    existing_applications = get_user_job_applications(db, current_user.id, job_id)
    if existing_applications:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="您已经申请过该职位"
        )
    
    return apply_for_job(db, job_id, current_user.id, application_in)


@router.get("/{job_id}/applications", response_model=List[JobApplicationSchema])
async def read_job_applications(
    job_id: uuid.UUID,
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    获取职位的申请列表
    """
    # 检查用户是否为雇主
    if current_user.role != "employer":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有雇主可以查看职位申请"
        )
    
    job = get_job(db, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="职位不存在"
        )
    
    # 检查是否为职位创建者
    if str(job.employer_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有职位创建者可以查看申请"
        )
    
    return get_job_applications(db, job_id, skip=skip, limit=limit, status=status)


@router.get("/applications/my", response_model=List[JobApplicationSchema])
async def read_my_applications(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    获取当前用户的职位申请
    """
    # 检查用户是否为个人用户
    if current_user.role != "individual":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有个人用户可以查看自己的职位申请"
        )
    
    return get_user_job_applications(db, current_user.id, None, skip=skip, limit=limit, status=status)


@router.put("/{job_id}/applications/{application_id}", response_model=JobApplicationSchema)
async def update_application_status(
    job_id: uuid.UUID,
    application_id: uuid.UUID,
    status: str = Query(..., description="申请状态"),
    notes: Optional[str] = Query(None, description="备注"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    更新职位申请状态
    """
    # 检查用户是否为雇主
    if current_user.role != "employer":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有雇主可以更新申请状态"
        )
    
    job = get_job(db, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="职位不存在"
        )
    
    # 检查是否为职位创建者
    if str(job.employer_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有职位创建者可以更新申请状态"
        )
    
    return update_job_application_status(db, application_id, status, notes)
