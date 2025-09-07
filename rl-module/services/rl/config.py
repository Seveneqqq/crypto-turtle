CONFIG = {
    "balance_start": 1000.0,  # USD
    "time_step": 1,           # liczba sekund między kolejnymi tickami symulacji
    "threshold_buy" : 0.6,
    "threshold_sell": 0.6,
    "data_files": {
        #"BTC": "BTCUSDT_1m_1m_2025-09-06.csv",
        "BTC": "BTCUSDT_4h_1m_2025-09-03.csv",
        #"ETH": "data/ETHUSDT_1m_1m_2025-09-06.csv",
        # kolejne dane
    },
    "models": {
        "BTC": "btc_model_cls.pkl",
        #"ETH": "eth_model_cls.pkl",
        # kolejne modele
    }
}