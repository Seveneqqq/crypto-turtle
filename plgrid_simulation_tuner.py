from module_loader import block_simulation_tuner

if __name__ == "__main__":

    block_simulation_tuner().run_tuning(
        eval_data_files = {
            #"BTC": "processed_ROBUST_set4_BTCUSDT-simulation_1m_2m_2025-10-11_8a076fe5-6d3c-4b1c-a490-9be3a3174517.csv",
            "ETH": "processed_ROBUST_set4_ETHUSDT-simulation_1m_2m_2025-10-11_ebb1db90-7ab0-4351-b9a4-908f0da5b244.csv"
        },
        model_files = {
            #"BTC": "btc_model_c_v1_1210_11.pkl",
            "ETH": "eth_model_c_v1_1210_11.pkl"
        },
        config = {
            "balance_start":                ["C", 1000.0],
            "threshold_buy":                ["F", 0.2, 0.8],
            "threshold_sell":               ["F", 0.2, 0.8],
            "wallet_percentage_limit":      ["F", 0.6, 1.0],
            "max_wallet_percentage_crypto": ["F", 0.8, 1.0],
            "stop_loss":                    ["C", 1.0],
            "tx_fraction":                  ["F", 0.0, 0.3],
            "max_fraction_when_confident":  ["F", 0.2, 1.0],
            "slippage":                     ["C", 0.0],
            "mincost":                      ["F", 1.0, 100.0],
        },
        n_trials  = 200,
        #n_jobs    = 1,
        #timeout   = None,
        #study_name= None
        live_logging = True,
        experiment_version="v0"
    )

