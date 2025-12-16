from configuration.rabbitmq_queues import DATA_REQUEST_QUEUE


class DataService:

    def __init__(self, sqlite_service, rabbitmq_service):
        self.sqlite_service = sqlite_service
        self.rabbitmq_service = rabbitmq_service

    # -------------------- COMMON --------------------

    def _send(self, payload: dict):
        rabbit_msg_id = self.sqlite_service.generate_unique_rabbit_msg_id()
        payload["rabbit_msg_id"] = rabbit_msg_id

        self.rabbitmq_service.publish(
            queue=DATA_REQUEST_QUEUE,
            message=payload
        )

        self.sqlite_service.insert_message(
            rabbit_msg_id=rabbit_msg_id,
            state="created",
            request=str(payload)
        )

    def handle_response(self, message: dict):
        self.sqlite_service.update_message(
            rabbit_msg_id=message.get("rabbit_msg_id"),
            state=message.get("result"),
            response=str(message)
        )

    # -------------------- OPTUNA CLASSIC --------------------

    def request_tune_data(
        self,
        filename_candidates,
        steps,
        n_trials,
        technical_sets,
        preprocessing_methods,
        output_folders,
        metrics,
        top_k_combinations,
        max_features_in_combo
    ):
        payload = {
            "action": "tune_data",
            "filename_candidates": filename_candidates,
            "steps": steps,
            "n_trials": n_trials,
            "technical_sets": technical_sets,
            "preprocessing_methods": preprocessing_methods,
            "output_folders": output_folders,
            "metrics": metrics,
            "top_k_combinations": top_k_combinations,
            "max_features_in_combo": max_features_in_combo
        }

        self._send(payload)

    # -------------------- OPTUNA LSTM --------------------

    def request_tune_data_lstm(
        self,
        filename_candidates,
        steps,
        n_trials,
        technical_sets,
        preprocessing_methods,
        output_folders,
        metrics,
        top_k_combinations,
        max_features_in_combo
    ):
        payload = {
            "action": "tune_lstm",
            "filename_candidates": filename_candidates,
            "steps": steps,
            "n_trials": n_trials,
            "technical_sets": technical_sets,
            "preprocessing_methods": preprocessing_methods,
            "output_folders": output_folders,
            "metrics": metrics,
            "top_k_combinations": top_k_combinations,
            "max_features_in_combo": max_features_in_combo
        }

        self._send(payload)

    # -------------------- DATA PROCESSING --------------------

    def process_data(
            self,
            filename,
            steps,
            preprocessing_method,
            output_folders,
            horizon=1,
            feature_sets=None
        ):
            payload = {
                "action": "process_data",
                "filename": filename,
                "steps": steps,
                "preprocessing_method": preprocessing_method,
                "output_folders": output_folders,
                "horizon": horizon,
                "feature_sets": feature_sets or []
            }
        
            self._send(payload)

    def process_data_lstm(
        self,
        filename,
        steps,
        preprocessing_method,
        output_folders,
        technical_indicators_set,
        selected_features
    ):
        payload = {
            "action": "process_data_lstm",
            "filename": filename,
            "steps": steps,
            "preprocessing_method": preprocessing_method,
            "output_folders": output_folders,
            "technical_indicators_set": technical_indicators_set,
            "selected_features": selected_features
        }

        self._send(payload)

    # -------------------- HISTORICAL DATA --------------------

    def request_historical_data(self, symbol, interval, start_dt, end_dt, filename):
        payload = {
            "action": "get_historical_data",
            "symbol": symbol,
            "interval": interval,
            "start_dt": start_dt,
            "end_dt": end_dt,
            "filename": filename
        }

        self._send(payload)
