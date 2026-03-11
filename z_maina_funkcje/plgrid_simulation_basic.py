from module_loader import block_simulation

if __name__ == "__main__":

    block_simulation().run_simulation(
        rabbit_msg_id=None,
        model_paths={
            #"BTC": "btc_model_c_v1_1210_11.pkl",
            #"BTC": "L003_btc_model_B001_1512",
            #"ETH": "eth_model_c_v1_1210_11.pkl",
            "ETH": "X065_eth_xgb_model_E103_0903"
            #"ETH":"X063_eth_xgb_model_E102_0903"
            #"BTC": "X067_btc_xgb_model_B103_0903"
            #"ETH": "eth_model_c_v1_0603_0.pkl"
            #"ETH": "eth_model_fe_2_c_v1_0910_ref.pkl"
            #"ETH": "eth_model_c_v1_2202_3.pkl"
        },
        data_files={
            #"BTC": "processed_ROBUST_set4_BTCUSDT-simulation_1m_2m_2025-10-11_8a076fe5-6d3c-4b1c-a490-9be3a3174517.csv",
            #"ETH": "realtime_data111.csv"
            #"ETH": "processed_ROBUST_set4_ETHUSDT-simulation_1m_2m_2025-10-11_ebb1db90-7ab0-4351-b9a4-908f0da5b244.csv",
            "ETH":  "xgb_eth_raw8_short.csv"
            #"BTC": "btc_eth_raw8_short.csv"
            #"ETH": "fe_v2_eth_test_20251009_ref.csv"
            #"ETH": "ETHUSDT_1m_6m_2026-02-22_22cc9313-e635-4c2c-9a8e-2360745d150d.csv"
            #"BTC": "processed_ROBUST_set4_BTCUSDT_1s_2m_2026-01-11_9d42fe9e-b4ad-413f-ad84-aaaaaf53705c.csv"
            #"BTC": "processed_ROBUST_set4_BTCUSDT-agent_1m_5m_2025-10-11_2d07d1b6-baf3-4ed8-bdc0-390156e92dc6.csv"
        },

        # agent_rl="agent_ppo_2026-02-15_01-07-25.zip",
        # balance_start=1000.0,
        # add_fee=True,
        # sim_mode="ml",
        # rlagent_deterministic=True,
        # slippage=0,
        # wallet_percentage_limit=0.8145955986522517,
        # max_wallet_percentage_crypto=0.8963629415450937,
        # stop_loss=1.0,
        # threshold_buy=0.20027788112568007,
        # threshold_sell=0.7994459815279178,
        # mincost=97.49847558308255,
        # tx_fraction=0.1680413746296665,
        # max_fraction_when_confident=0.20358543756356234,
        # plots=True,
        # logs=True,
        # results=True,
        # experiment_version="v0"

        #ETH XGB

        agent_rl="agent_ppo_2026-02-24_23-06-08.zip",
        balance_start=1000.0,
        add_fee=True,
        sim_mode="ml",
        rlagent_deterministic=True,
        slippage=0,
        wallet_percentage_limit=0.959,
        max_wallet_percentage_crypto=0.967,
        stop_loss=1.0,
        threshold_buy=0.6,
        threshold_sell=0.6,
        mincost=4.30,
        tx_fraction=0.0113,
        max_fraction_when_confident=0.369,
        plots=True,
        logs=True,
        results=True,
        experiment_version="v0"


        # # ETH new
        # agent_rl="agent_ppo_2026-02-24_23-06-08.zip",
        # balance_start=1000.0,
        # add_fee=True,
        # sim_mode="ml",
        # rlagent_deterministic=True,
        # slippage=0,
        # wallet_percentage_limit=0.6439158996392444,
        # max_wallet_percentage_crypto=0.8739489383305068,
        # stop_loss=1.0,
        # threshold_buy=0.5470628217105851,
        # threshold_sell= 0.6200350636580595,
        # mincost=1.0022504036489839,
        # tx_fraction=0.015397748926383079,
        # max_fraction_when_confident=0.6648443549617613,
        # plots=True,
        # logs=True,
        # results=True,
        # experiment_version="v0"

        #ETH BEST
        # agent_rl="agent_ppo_2026-02-24_23-06-08.zip",
        # balance_start=1000.0,
        # add_fee=True,
        # sim_mode="ml",
        # rlagent_deterministic=True,
        # slippage=0,
        # wallet_percentage_limit=0.9834968734849104,
        # max_wallet_percentage_crypto=0.9391418056045563,
        # stop_loss=1.0,
        # threshold_buy=0.5192878230846011,
        # threshold_sell=0.5209501281716447,
        # mincost=1.3578146192056635,
        # tx_fraction=0.08929427393763487,
        # max_fraction_when_confident=0.9162558206679969,
        # plots=True,
        # logs=True,
        # results=True,
        # experiment_version="v0"

        #BTC BEST

        # agent_rl = "agent_ppo_2026-02-15_01-07-25.zip",
        # balance_start = 1000.0,
        # add_fee = True,
        # sim_mode = "ml",
        # rlagent_deterministic = True,
        # slippage = 0,
        # wallet_percentage_limit = 0.6219,
        # max_wallet_percentage_crypto = 0.8456,
        # stop_loss = 1.0,
        # threshold_buy = 0.5387,
        # threshold_sell = 0.5524,
        # mincost = 1.013,
        # tx_fraction = 0.1156,
        # max_fraction_when_confident = 0.9646,
        # plots = True,
        # logs = True,
        # results = True,
        # experiment_version = "v0"
    )

