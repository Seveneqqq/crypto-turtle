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
        learning_rate=0.0001,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=128,
        gamma=0.99,
        #tau=0.01,
        #train_freq=4,
        #gradient_steps=1,
        #exploration_fraction=0.1,
        #exploration_initial_eps=1.0,
        #exploration_final_eps=0.05,
        episode_timesteps=1024,
        ml_threshold=0.45,
        #dqn_bins=3,
        live_logging=True,
        experiment_version="v0"
    )

