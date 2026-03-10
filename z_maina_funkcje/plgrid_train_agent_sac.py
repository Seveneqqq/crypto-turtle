from z_maina_funkcje.module_loader import block_train_sac

if __name__ == "__main__":

    block_train_sac().train_agent_sac(
        rabbitmq_msg_id=None,
        model_files={
            "BTC": "btc_model_c_v1_1210_11.pkl"
        },
        data_files={
			"BTC": "processed_ROBUST_set4_BTCUSDT-agent_1m_5m_2025-10-11_2d07d1b6-baf3-4ed8-bdc0-390156e92dc6.csv"
		},
        total_timesteps=500_000,
        learning_rate=0.0003,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        episode_timesteps=1024,
        ml_threshold=0.45,
        live_logging=True,
        experiment_version="v0"
    )


