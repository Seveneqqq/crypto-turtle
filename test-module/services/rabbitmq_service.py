import json
import pika
import threading

class RabbitMQService:
    def __init__(self, host: str, port: int, username: str, password: str):
        self.host = host
        self.port = port
        self.username = username
        self.password = password

    def connect(self):
        credentials = pika.PlainCredentials(self.username, self.password)
        return pika.BlockingConnection(pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host='/',
            credentials=credentials
        ))

    def publish(self, queue: str, message: dict):
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
        connection.close()

    def handle_request(self, data):
        #Tutaj będzie obsługa requesta z gateway
        print(f"Handling request: {data}")
        #Na koniec bedzie za pomoca publish zwracany wynik do gatewaya

    def consume(self, queue: str, callback):
        def _consume():
            print("Starting consumer thread") 
            connection = self.connect()
            print(f"Connected to RabbitMQ at {self.host}:{self.port}")
            channel = connection.channel()
            channel.queue_declare(queue=queue, durable=True)

            def wrapped_callback(ch, method, properties, body):
                try:
                    if not body:
                        print("Received empty message body, skipping...")
                        return

                    message_str = body.decode("utf-8") if isinstance(body, bytes) else body
                    data = json.loads(message_str)
                    callback(data)

                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e} | Raw body: {body}")
                except Exception as e:
                    print(f"Unexpected error in consumer: {e}")
                finally:
                    ch.basic_ack(delivery_tag=method.delivery_tag)


            channel.basic_consume(
                queue=queue,
                on_message_callback=wrapped_callback,
                auto_ack=False
            )

            print(f" [*] Waiting for messages on {queue}")
            channel.start_consuming()

        # uruchom konsumpcję w osobnym wątku
        threading.Thread(target=_consume, daemon=True).start()
