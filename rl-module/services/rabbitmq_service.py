import json
import pika
import threading
import time

class RabbitMQService:
    def __init__(self, host: str, port: int, username: str, password: str, request_service=None):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.request_service = request_service

    def connect(self):
        credentials = pika.PlainCredentials(self.username, self.password)
        return pika.BlockingConnection(pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host='/',
            credentials=credentials,
            heartbeat=600,  
            blocked_connection_timeout=300
        ))

    def publish(self, queue: str, message: dict):
        try:
            connection = self.connect()
            channel = connection.channel()
            channel.queue_declare(queue=queue, durable=True)

            channel.basic_publish(
                exchange="",
                routing_key=queue,
                body=json.dumps(message),
                properties=pika.BasicProperties(delivery_mode=2)
            )
            print(f" [x] Sent {message} to {queue}")
        except pika.exceptions.AMQPConnectionError as e:
            print(f"[ERROR] Failed to publish message: {e}")
        finally:
            if 'connection' in locals() and connection.is_open:
                connection.close()

    def handle_request(self, data):
        print(f"Handling request: {data}")
        if self.request_service:
            self.request_service.handle_request(data)
        else:
            print("No request_service attached!")

    def consume(self, queue: str, callback):
        def _consume():
            while True:
                try:
                    print("Starting consumer thread")
                    connection = self.connect()
                    print(f"Connected to RabbitMQ at {self.host}:{self.port}")
                    channel = connection.channel()
                    channel.queue_declare(queue=queue, durable=True)

                    def wrapped_callback(ch, method, properties, body):
                        def process_message():
                            try:
                                message_str = body.decode("utf-8") if isinstance(body, bytes) else body
                                data = json.loads(message_str)
                                callback(data)
                            except Exception as e:
                                print(f"Processing error: {e}")
                            finally:
                                try:
                                    ch.basic_ack(delivery_tag=method.delivery_tag)
                                except Exception as e:
                                    print(f"Ack error: {e}")

                        threading.Thread(target=process_message, daemon=True).start()

                    channel.basic_consume(
                        queue=queue,
                        on_message_callback=wrapped_callback,
                        auto_ack=False
                    )

                    print(f" [*] Waiting for messages on {queue}")
                    channel.start_consuming()

                except pika.exceptions.AMQPConnectionError as e:
                    print(f"[WARNING] Connection lost, retrying in 5s... {e}")
                    time.sleep(5)
                except Exception as e:
                    print(f"[ERROR] Unexpected consumer error, retrying in 5s... {e}")
                    time.sleep(5)

        threading.Thread(target=_consume, daemon=True).start()
