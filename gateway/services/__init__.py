from services.file_service import FileService
from .auth_service import AuthService
from configuration.configuration import SUPABASE_URL, SUPABASE_KEY, GOOGLE_REDIRECT_URI
from services.rabbitmq_service import RabbitMQService
from configuration.rabbitmq_configuration import (
    RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_USERNAME, RABBITMQ_PASSWORD
)

rabbitmq_service = RabbitMQService(
    host=RABBITMQ_HOST,
    port=RABBITMQ_PORT,
    username=RABBITMQ_USERNAME,
    password=RABBITMQ_PASSWORD
)

auth_service = AuthService(SUPABASE_URL, SUPABASE_KEY, GOOGLE_REDIRECT_URI)
file_service = FileService()
rabbitmq_service = RabbitMQService( host=RABBITMQ_HOST, port=RABBITMQ_PORT, username=RABBITMQ_USERNAME, password=RABBITMQ_PASSWORD )
