# ============================================================================
# DECISION ENGINE METHODS
# ============================================================================

def v1_decision_rl_config(self, scaled):
    fr_min, fr_max = 0.01, 1.0
    fraction_buy = fr_min + scaled[0] * (fr_max - fr_min)
    fraction_sell = fr_min + scaled[1] * (fr_max - fr_min)

    self.config.fraction_buy = float(fraction_buy)
    self.config.fraction_sell = float(fraction_sell)


def v1_decision_calculate_fraction(self, order):
    """V1: Dopisuje fraction zdefiniowane przez agenta rl"""
    if order.action == "buy":
        order.fraction = self.config.fraction_buy
    else:
        order.fraction = self.config.fraction_sell
    return order





# ============================================================================
# WALLET METHODS
# ============================================================================




# ============================================================================
# TRADING ENV METHODS
# ============================================================================




# ============================================================================
# EXPERIMENT REGISTRY
# ============================================================================

EXPERIMENTS = {
    "v0": {
        "description": "Podstawowa wersja core_logic",
        "n_params_per_symbol": 4,

        "decision_methods": {},

        "wallet_methods": {},

        "env_methods": {}
    },
    "v1": {
        "description": "Fraction liczone przez agenta",
        "n_params_per_symbol": 2,

        "decision_methods": {
            "calculate_fraction": v1_decision_calculate_fraction,
            "rl_config": v1_decision_rl_config,
        },

        "wallet_methods": {},

        "env_methods": {}
    },
}

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

