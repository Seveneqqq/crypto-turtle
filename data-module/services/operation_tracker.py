import functools
from datetime import datetime

from services.operation_history_service import OperationHistoryService
from services.operation_file_logger_service import OperationFileLoggerService


history_service = OperationHistoryService()
logger_service = OperationFileLoggerService()


def track_operation(service_name: str, operation_name: str, operation_group: str):

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            start_time = datetime.utcnow()

            request_payload = {
                "args": args,
                "kwargs": kwargs
            }

            try:

                result = func(*args, **kwargs)

                finish_time = datetime.utcnow()

                duration = int((finish_time - start_time).total_seconds() * 1000)

                filename = None
                file_path = None
                rabbitmq_msg_id = None

                if isinstance(result, dict):

                    filename = result.get("filename")
                    file_path = result.get("path")
                    rabbitmq_msg_id = result.get("rabbit_msg_id")

                history_service.save_operation(
                    service_name=service_name,
                    operation_name=operation_name,
                    operation_group=operation_group,
                    status="success",
                    request_payload=request_payload,
                    response_payload=result,
                    filename=filename,
                    file_path=file_path,
                    rabbitmq_msg_id=rabbitmq_msg_id,
                    started_at=start_time.isoformat(),
                    finished_at=finish_time.isoformat(),
                    duration_ms=duration
                )

                logger_service.log(
                    service_name=service_name,
                    operation_name=operation_name,
                    status="success",
                    request_payload=request_payload,
                    response_payload=result,
                    filename=filename,
                    file_path=file_path,
                    rabbitmq_msg_id=rabbitmq_msg_id,
                    started_at=start_time.isoformat(),
                    finished_at=finish_time.isoformat(),
                    duration_ms=duration
                )

                return result

            except Exception as e:

                finish_time = datetime.utcnow()

                duration = int((finish_time - start_time).total_seconds() * 1000)

                history_service.save_operation(
                    service_name=service_name,
                    operation_name=operation_name,
                    operation_group=operation_group,
                    status="failed",
                    request_payload=request_payload,
                    error_message=str(e),
                    started_at=start_time.isoformat(),
                    finished_at=finish_time.isoformat(),
                    duration_ms=duration
                )

                logger_service.log(
                    service_name=service_name,
                    operation_name=operation_name,
                    status="failed",
                    request_payload=request_payload,
                    error_message=str(e),
                    started_at=start_time.isoformat(),
                    finished_at=finish_time.isoformat(),
                    duration_ms=duration
                )

                raise

        return wrapper

    return decorator