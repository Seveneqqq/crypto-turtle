from module_loader import block_train_ppo

if __name__ == "__main__":

    block_train_ppo().train_agent_ppo(
        rabbitmq_msg_id=None,
        model_files={
            "BTC": "btc_model_c_v1_1210_11.pkl"
        },
        data_files={
			"BTC": "processed_ROBUST_set4_BTCUSDT-agent_1m_5m_2025-10-11_2d07d1b6-baf3-4ed8-bdc0-390156e92dc6.csv"
		},
        total_timesteps=5000000,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.97,
        ent_coef=0.05
    )


