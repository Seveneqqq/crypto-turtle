from configuration.rabbitmq_queues import TEST_RESPONSE_QUEUE
from configuration.rabbitmq_queues import XAI_REQUEST_QUEUE


class RabbitMQRequestService:
    def __init__(self, rabbitmq_service, simulation_service, visualization_service, data_quality_service, realtime_generate_decision_service):

        self.rabbitmq_service = rabbitmq_service
        self.simulation_service = simulation_service
        self.visualization_service = visualization_service
        self.data_quality_service = data_quality_service
        self.realtime_generate_decision_service = realtime_generate_decision_service

        self.aggregator = CandleAggregator(
            on_complete=self._handle_complete_batch,
            initial_wait_s=10,
            wait_s=5
        )

    def handle_request(self, data):
        print(data)
        match data.get("action"):
            case "simulate":
                print("Getting simulate data")
                result = self.simulation_service.run_simulation(
                    rabbit_msg_id=data.get("rabbit_msg_id"),
                    balance_start=data.get("balance_start"),
                    add_fee=data.get("add_fee"),
                    sim_mode=data.get("sim_mode"),
                    rlagent_deterministic=data.get("rlagent_deterministic"),
                    slippage=data.get("slippage"),
                    wallet_percentage_limit=data.get("wallet_percentage_limit"),
                    max_wallet_percentage_crypto=data.get("max_wallet_percentage_crypto"),
                    stop_loss=data.get("stop_loss"),
                    threshold_buy=data.get("threshold_buy"),
                    threshold_sell=data.get("threshold_sell"),
                    mincost=data.get("mincost"),
                    tx_fraction=data.get("tx_fraction"),
                    max_fraction_when_confident=data.get("max_fraction_when_confident"),
                    data_files=data.get("data_files"),
                    model_paths=data.get("model_paths"),
                    agent_rl=data.get("agent_rl"),
                    plots=data.get("plots"),
                    logs=data.get("logs"),
                    results=data.get("results")

                )
                if result.get("result") == "success":
                    print("Successfully retrieved simulate data")
                    self.rabbitmq_service.publish(TEST_RESPONSE_QUEUE, result)
                else:
                    print("Failed to retrieve simulate data")
                    self.rabbitmq_service.publish(TEST_RESPONSE_QUEUE, {
                        "result": "failed",
                        "message": "Failed to retrieve simulation data"
                    })
            case "visualize":
                print("Getting visualize data")
                result = self.visualization_service.visualize(
                    rabbit_msg_id=data.get("rabbit_msg_id"),
                    simulation_files=data.get("simulation_files")

                )
                if result.get("result") == "success":
                    print("Successfully retrieved visualize data")
                    self.rabbitmq_service.publish(TEST_RESPONSE_QUEUE, result)
                else:
                    print("Failed to retrieve visualize data")
                    self.rabbitmq_service.publish(TEST_RESPONSE_QUEUE, {
                        "result": "failed",
                        "message": "Failed to retrieve visualize data"
                    })
            
            case "calculate_quality":
                print("Calculating data quality")
                result = self.data_quality_service.calculate_quality(
                    rabbit_msg_id=data.get("rabbit_msg_id"),
                    data_dir=data.get("data_dir"),
                    data_file=data.get("data_file"),
                )
                if result.get("result") == "success":
                    print("Successfully calculated data quality")
                    self.rabbitmq_service.publish(TEST_RESPONSE_QUEUE, result)
                else:
                    print("Failed to calculate data quality")
                    self.rabbitmq_service.publish(TEST_RESPONSE_QUEUE, result)

    def _handle_complete_batch(self, batch: dict):
        # batch = {"btcusdt": {candles: ...}, "ethusdt": {candles: ...}}
        print(batch)
        result = self.realtime_generate_decision_service.live_decision(
            results=list(batch.values())
        )
        if result and result.get("result") == "success":
            print("Successfully processed realtime data")

            self.rabbitmq_service.publish(TEST_RESPONSE_QUEUE, result)

        else:
            print("Failed to process realtime data")

            self.rabbitmq_service.publish(TEST_RESPONSE_QUEUE, {
                "result": "failed",
                "message": "Failed to process realtime data"
            })

    def handle_realtime_request(self, data):
        print("Getting realtime request ...")

        match data.get("event"):

            case "processed_candles":
                print("Processing realtime candles")

                result = {
                    "symbol": data.get("symbol"),
                    "interval": data.get("interval"),
                    "candles": data.get("data")
                }
                #TODO: test_result - zaślepka, dla poprawnego (niezdublowanego) działania przed wprowadzeniem streamingu wielu symboli, podmienić results=results na result=result
                test_result = {
                    "symbol": "TESTUSDT",
                    "interval": data.get("interval"),
                    "candles": data.get("data")
                }
                results = [result,test_result]

                decisions_result = self.realtime_generate_decision_service.live_decision(results=results)

                if decisions_result and decisions_result.get("result") == "success":
                    print("Successfully processed realtime data")

                    self.rabbitmq_service.publish(TEST_RESPONSE_QUEUE, decisions_result)
                    self.rabbitmq_service.publish(XAI_REQUEST_QUEUE, decisions_result)

                else:
                    print("Failed to process realtime data")

                    self.rabbitmq_service.publish(TEST_RESPONSE_QUEUE, {
                        "result": "failed",
                        "message": "Failed to process realtime data"
                    })
                    self.rabbitmq_service.publish(XAI_REQUEST_QUEUE, {
                        "result": "failed",
                        "message": "Failed to process realtime data"
                    })

                #TODO: Agregator dla streamingu wielu symboli + dorobić zwracanie do agregatora
                # self.aggregator.add(
                #     symbol=data.get("symbol"),
                #     result={
                #         "symbol": data.get("symbol"),
                #         "interval": data.get("interval"),
                #         "candles": data.get("data")
                #     }
                # )

                #Tutaj bedzie publikowanie danych z decision engine wraz z danymi ktore przyszly do xai i do serwisu ktory bedzie wykonywal operacje na binance

import threading
import time

class CandleAggregator:
    def __init__(self, on_complete, initial_wait_s: int = 10, wait_s: int = 5, idle_reset_min: int = 10):
        self.on_complete = on_complete
        self.initial_wait_s = initial_wait_s
        self.wait_s = wait_s
        self.idle_reset_s = idle_reset_min * 60

        self.known_symbols: set = set()
        self.current_batch: dict = {}
        self.is_first_batch = True
        self.batch_timer: threading.Timer | None = None
        self.last_flush_time: float | None = None
        self.lock = threading.Lock()

    def _is_idle_too_long(self) -> bool:
        if self.last_flush_time is None:
            return False
        return time.time() - self.last_flush_time > self.idle_reset_s

    def add(self, symbol: str, result: dict):
        with self.lock:
            # 1. Rejestracja symbolu (musi być przed jakimkolwiek flushem)
            if symbol not in self.known_symbols:
                self.known_symbols.add(symbol)
                print(f"[Aggregator] Rejestracja: {symbol}. Znane: {self.known_symbols}")

            # 2. Dodanie do paczki
            self.current_batch[symbol] = result

            # 3. Sprawdź czy mamy komplet (zbiory muszą być identyczne)
            if self.known_symbols == self.current_batch.keys():
                print(f"[Aggregator] Komplet ({len(self.current_batch)}) - Flush natychmiastowy")
                if self.batch_timer:
                    self.batch_timer.cancel()
                    self.batch_timer = None
                self._flush_unlocked()
                return

            # 4. Jeśli nie ma kompletu i timer nie tyka, wystartuj go
            # Używamy krótkiego czasu (wait_s), bo symbole "przychodzą jednocześnie"
            if self.batch_timer is None:
                wait = self.initial_wait_s if self.is_first_batch else self.wait_s
                print(f"[Aggregator] Brak kompletu, czekam {wait}s na resztę...")
                self.batch_timer = threading.Timer(wait, self._flush)
                self.batch_timer.start()

    def _flush(self):
        with self.lock:
            self._flush_unlocked()

    def _flush_unlocked(self):
        batch = self.current_batch.copy()

        missing = self.known_symbols - batch.keys()
        if missing:
            print(f"[Aggregator] Symbols timed out, removing: {missing}")
            self.known_symbols -= missing

        self.current_batch = {}
        self.batch_timer = None
        self.is_first_batch = False
        self.last_flush_time = time.time()

        threading.Thread(target=self.on_complete, args=(batch,), daemon=True).start()