from module_loader import block_ppo_tuner

if __name__ == "__main__":

    block_ppo_tuner().run_tuning(
        train_data_files = {
            "BTC": "processed_ROBUST_set4_BTCUSDT-agent_1m_5m_2025-10-11_2d07d1b6-baf3-4ed8-bdc0-390156e92dc6.csv",
        },
        eval_data_files = {
            "BTC": "processed_ROBUST_set4_BTCUSDT-simulation_1m_2m_2025-10-11_8a076fe5-6d3c-4b1c-a490-9be3a3174517.csv"
        },
        model_files = {
            "BTC": "btc_model_c_v1_1210_11.pkl"
        },
        config = {
            "episode_timesteps": ["C", 512, 1024, 2048],
            "learning_rate"    : ["FL", 1e-5, 1e-3],
            "gamma"            : ["F", 0.95, 0.999],
            "batch_size"       : ["C", 512, 1024],
            "ml_threshold"     : ["F", 0.45, 0.65],
            "total_timesteps"  : ["C", 10000, 500000, 100000],

            # PPO-specific
            "n_steps"   : ["C", 2048, 4096],
            "n_epochs"  : ["C", 10],
            #"gae_lambda": [],
            "clip_range": ["C", 0.2],
            "ent_coef"  : ["C", 0.001],
        },
        n_trials  = 10,
        #n_jobs    = 1,
        #timeout   = None,
        #study_name= None
        live_logging = True,
        experiment_version="v0"
    )

