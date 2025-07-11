from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Body
from sqlalchemy.orm import Session

from app import schemas
from app.api import deps
from app.models.user import User
from app.services.digital_twin_service import DigitalTwinService
from app.ai.twin_engine import process_data_source

router = APIRouter()


@router.get("/", response_model=List[schemas.DigitalTwin])
def read_digital_twins(
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    获取当前用户的所有数字分身
    """
    service = DigitalTwinService(db)
    return service.get_user_digital_twins(current_user.id)


@router.post("/", response_model=schemas.DigitalTwin)
def create_digital_twin(
    *,
    db: Session = Depends(deps.get_db),
    twin_in: schemas.DigitalTwinCreate,
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    创建新的数字分身
    """
    service = DigitalTwinService(db)
    return service.create_digital_twin(twin_in, current_user.id)


@router.get("/{twin_id}", response_model=schemas.DigitalTwin)
def read_digital_twin(
    *,
    db: Session = Depends(deps.get_db),
    twin_id: str,
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    获取指定数字分身
    """
    service = DigitalTwinService(db)
    twin = service.get_digital_twin(twin_id)
    if not twin:
        raise HTTPException(status_code=404, detail="Digital twin not found")
    if twin.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return twin


@router.put("/{twin_id}", response_model=schemas.DigitalTwin)
def update_digital_twin(
    *,
    db: Session = Depends(deps.get_db),
    twin_id: str,
    twin_in: schemas.DigitalTwinUpdate,
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    更新数字分身
    """
    service = DigitalTwinService(db)
    twin = service.get_digital_twin(twin_id)
    if not twin:
        raise HTTPException(status_code=404, detail="Digital twin not found")
    if twin.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return service.update_digital_twin(twin_id, twin_in)


@router.delete("/{twin_id}", response_model=schemas.DigitalTwin)
def delete_digital_twin(
    *,
    db: Session = Depends(deps.get_db),
    twin_id: str,
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    删除数字分身
    """
    service = DigitalTwinService(db)
    twin = service.get_digital_twin(twin_id)
    if not twin:
        raise HTTPException(status_code=404, detail="Digital twin not found")
    if twin.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    service.delete_digital_twin(twin_id)
    return twin


@router.get("/{twin_id}/data-sources", response_model=List[schemas.TwinDataSource])
def read_data_sources(
    *,
    db: Session = Depends(deps.get_db),
    twin_id: str,
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    获取数字分身的所有数据源
    """
    service = DigitalTwinService(db)
    twin = service.get_digital_twin(twin_id)
    if not twin:
        raise HTTPException(status_code=404, detail="Digital twin not found")
    if twin.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return service.get_data_sources(twin_id)


@router.post("/{twin_id}/data-sources", response_model=schemas.TwinDataSource)
def create_data_source(
    *,
    db: Session = Depends(deps.get_db),
    twin_id: str,
    source_type: str = Form(...),
    name: str = Form(...),
    content: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    metadata: Optional[dict] = Body(None),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    为数字分身添加数据源
    """
    service = DigitalTwinService(db)
    twin = service.get_digital_twin(twin_id)
    if not twin:
        raise HTTPException(status_code=404, detail="Digital twin not found")
    if twin.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    # 如果上传了文件，读取文件内容
    file_path = None
    if file:
        file_content = file.file.read()
        content = file_content.decode("utf-8")
        file_path = file.filename
    
    return service.add_data_source(
        twin_id=twin_id,
        source_type=source_type,
        name=name,
        content=content,
        file_path=file_path,
        url=url,
        metadata=metadata
    )


@router.get("/{twin_id}/conversations", response_model=List[schemas.Conversation])
def read_conversations(
    *,
    db: Session = Depends(deps.get_db),
    twin_id: str,
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    获取数字分身的所有对话
    """
    service = DigitalTwinService(db)
    twin = service.get_digital_twin(twin_id)
    if not twin:
        raise HTTPException(status_code=404, detail="Digital twin not found")
    if twin.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return service.get_conversations(twin_id)


@router.post("/{twin_id}/conversations", response_model=schemas.Conversation)
def create_conversation(
    *,
    db: Session = Depends(deps.get_db),
    twin_id: str,
    conversation_in: schemas.ConversationCreate,
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    为数字分身创建新对话
    """
    service = DigitalTwinService(db)
    twin = service.get_digital_twin(twin_id)
    if not twin:
        raise HTTPException(status_code=404, detail="Digital twin not found")
    if twin.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return service.create_conversation(twin_id, conversation_in.title)


@router.get("/{twin_id}/conversations/{conversation_id}/messages", response_model=List[schemas.Message])
def read_messages(
    *,
    db: Session = Depends(deps.get_db),
    twin_id: str,
    conversation_id: str,
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    获取对话的所有消息
    """
    service = DigitalTwinService(db)
    twin = service.get_digital_twin(twin_id)
    if not twin:
        raise HTTPException(status_code=404, detail="Digital twin not found")
    if twin.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return service.get_messages(conversation_id)


@router.post("/{twin_id}/conversations/{conversation_id}/messages", response_model=schemas.MessageResponse)
def create_message(
    *,
    db: Session = Depends(deps.get_db),
    twin_id: str,
    conversation_id: str,
    message_in: schemas.MessageCreate,
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    向对话发送消息并获取数字分身回复
    """
    service = DigitalTwinService(db)
    twin = service.get_digital_twin(twin_id)
    if not twin:
        raise HTTPException(status_code=404, detail="Digital twin not found")
    if twin.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return service.send_message_to_twin(conversation_id, message_in.content)


@router.post("/{twin_id}/generate-bio", response_model=schemas.DigitalTwin)
def generate_bio(
    *,
    db: Session = Depends(deps.get_db),
    twin_id: str,
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    生成数字分身的生物特征信息
    """
    service = DigitalTwinService(db)
    twin = service.get_digital_twin(twin_id)
    if not twin:
        raise HTTPException(status_code=404, detail="Digital twin not found")
    if twin.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    success = service.generate_twin_bio(twin_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to generate bio")
    
    return service.get_digital_twin(twin_id)


@router.post("/{twin_id}/generate-clusters", response_model=schemas.DigitalTwin)
def generate_clusters(
    *,
    db: Session = Depends(deps.get_db),
    twin_id: str,
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    为数字分身生成聚类
    """
    service = DigitalTwinService(db)
    twin = service.get_digital_twin(twin_id)
    if not twin:
        raise HTTPException(status_code=404, detail="Digital twin not found")
    if twin.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    success = service.generate_twin_clusters(twin_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to generate clusters")
    
    return service.get_digital_twin(twin_id)


@router.get("/{twin_id}/clusters", response_model=List[schemas.TwinCluster])
def get_clusters(
    *,
    db: Session = Depends(deps.get_db),
    twin_id: str,
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    获取数字分身的所有聚类
    """
    service = DigitalTwinService(db)
    twin = service.get_digital_twin(twin_id)
    if not twin:
        raise HTTPException(status_code=404, detail="Digital twin not found")
    if twin.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    return service.get_twin_clusters(twin_id)


@router.get("/{twin_id}/shades", response_model=List[schemas.TwinShade])
def get_shades(
    *,
    db: Session = Depends(deps.get_db),
    twin_id: str,
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    获取数字分身的所有特征
    """
    service = DigitalTwinService(db)
    twin = service.get_digital_twin(twin_id)
    if not twin:
        raise HTTPException(status_code=404, detail="Digital twin not found")
    if twin.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    return service.get_twin_shades(twin_id)
