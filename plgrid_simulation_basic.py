from module_loader import block_simulation

if __name__ == "__main__":

    block_simulation().run_simulation(
        rabbit_msg_id=None,
        model_paths={
            "BTC": "btc_model_c_v1_1210_11.pkl"
        },
        data_files={
            "BTC": "processed_ROBUST_set4_BTCUSDT-simulation_1m_2m_2025-10-11_8a076fe5-6d3c-4b1c-a490-9be3a3174517.csv"
        },
        agent_rl="agent_ppo_2026-01-05_00-17-31.zip",
        balance_start=1000.0,
        add_fee=True,
        sim_mode="rl",
        rlagent_deterministic=True,
        slippage=0,
        wallet_percentage_limit=0.9,
        max_wallet_percentage_crypto=1.0,
        stop_loss=1.0,
        threshold_buy=0.5,
        threshold_sell=0.5,
        mincost=1.0,
        tx_fraction=0.01,
        max_fraction_when_confident=0.5,
        plots=True,
        logs=True,
        results=True
    )

