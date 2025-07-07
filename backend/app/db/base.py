# Import all the models, so that Base has them before being
# imported by Alembic
from app.db.base_class import Base  # noqa
from app.models.user import User  # noqa
from app.models.oauth_account import OAuthAccount  # noqa
from app.models.user_setting import UserSetting  # noqa
from app.models.session import Session  # noqa
from app.models.digital_twin import DigitalTwin  # noqa
from app.models.twin_data_source import TwinDataSource, TwinChunk  # noqa
from app.models.twin_cluster import TwinCluster, TwinShade  # noqa
from app.models.conversation import Conversation  # noqa
from app.models.message import Message  # noqa
from app.models.individual_profile import IndividualProfile  # noqa
from app.models.experience import Experience  # noqa
from app.models.project import Project  # noqa
from app.models.education import Education  # noqa
from app.models.skill import Skill  # noqa
from app.models.employer_profile import EmployerProfile  # noqa
from app.models.job import Job  # noqa
from app.models.job_application import JobApplication  # noqa
from app.models.expert_profile import ExpertProfile  # noqa
from app.models.knowledge_base import KnowledgeBase, KnowledgeContent  # noqa
from app.models.consultation_service import ConsultationService  # noqa 