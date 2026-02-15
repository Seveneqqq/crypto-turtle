from module_loader import block_simulation

if __name__ == "__main__":

    block_simulation().run_simulation(
        rabbit_msg_id=None,
        model_paths={
            "BTC": "btc_model_c_v1_1210_11.pkl",
            #"BTC": "L003_btc_model_B001_1512",
            #"ETH": "eth_model_c_v1_1210_11.pkl",
        },
        data_files={
            "BTC": "processed_ROBUST_set4_BTCUSDT-simulation_1m_2m_2025-10-11_8a076fe5-6d3c-4b1c-a490-9be3a3174517.csv",
            #"ETH": "processed_ROBUST_set4_ETHUSDT-simulation_1m_2m_2025-10-11_ebb1db90-7ab0-4351-b9a4-908f0da5b244.csv"
            #"BTC": "processed_ROBUST_set4_BTCUSDT_1s_2m_2026-01-11_9d42fe9e-b4ad-413f-ad84-aaaaaf53705c.csv"
            #"BTC": "processed_ROBUST_set4_BTCUSDT-agent_1m_5m_2025-10-11_2d07d1b6-baf3-4ed8-bdc0-390156e92dc6.csv"
        },
        agent_rl="agent_ppo_2026-02-15_01-07-25.zip",
        balance_start=1000.0,
        add_fee=True,
        sim_mode="ml",
        rlagent_deterministic=True,
        slippage=0,
        wallet_percentage_limit=0.6219,
        max_wallet_percentage_crypto=0.8456,
        stop_loss=1.0,
        threshold_buy=0.5387,
        threshold_sell=0.5524,
        mincost=1.013,
        tx_fraction=0.1156,
        max_fraction_when_confident=0.9646,
        plots=True,
        logs=True,
        results=True,
        experiment_version="v0"
    )

