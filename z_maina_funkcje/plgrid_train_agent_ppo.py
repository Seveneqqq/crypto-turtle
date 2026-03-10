from z_maina_funkcje.module_loader import block_train_ppo

if __name__ == "__main__":

    block_train_ppo().train_agent_ppo(
        rabbitmq_msg_id=None,
        model_files={
            #"BTC": "btc_model_c_v1_1210_11.pkl",
            #"BTC": "L003_btc_model_B001_1512",
            "ETH": "eth_model_c_v1_1210_11.pkl",
        },
        data_files={
			#"BTC": "processed_ROBUST_set4_BTCUSDT-agent_1m_5m_2025-10-11_2d07d1b6-baf3-4ed8-bdc0-390156e92dc6.csv",
            "ETH": "processed_ROBUST_set4_ETHUSDT-agent_1m_5m_2025-10-11_1a539d7e-844e-4176-97d8-e5b7ebbd1160.csv"
		},
        total_timesteps=5000000,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=512,
        gamma=0.999,
        ent_coef=0.001,
        #n_epochs=10,
        #gae_lambda=0.95,
        #clip_range=0.2,
        episode_timesteps=10240,
        ml_threshold=0.50,
        live_logging=True,
        experiment_version="v1"
    )

