# Agent rl 

    {
        "action": "train_agent_ppo",
        "model_file": "btc_model_simple_c_v1_2209_1.pkl",
        "data_file": "test_20250922_230353.csv",
        "total_timesteps": 10000,
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "ent_coef": 0.0
    }


# symulacja

    {
        "action": "simulate",
        "balance_start": 1000.0,
        "add_fee": true,
        "sim_mode": "ml"/"rl"/"mlrl",
        "rlagent_deterministic": true,
        "slippage": 0,
        "wallet_percentage_limit": 0.9,
        "max_wallet_percentage_crypto": 1.0,
        "stop_loss": 1.0,
         "threshold_buy": 0.50,
         "threshold_sell": 0.50,
        "mincost": 1.0,
        "tx_fraction": 0.01,
        "max_fraction_when_confident": 0.5,
        "data_files": {
            "BTC": "test_20250922_230353.csv"
        },
        "models": {
            "BTC": "btc_model_simple_c_v1_2209_1.pkl"
        },
        "agents": {
            "BTC": "ppo.zip"
        },
        "plots": true,
        "logs": false,
        "results": false
    }