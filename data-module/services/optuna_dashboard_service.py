from .optuna_results_loader_service import OptunaResultsLoaderService
from .optuna_analytics_service import OptunaAnalyticsService
from .optuna_visualization_service import OptunaVisualizationService
from .optuna_advanced_analytics_service import OptunaAdvancedAnalyticsService
from .optuna_research_visualization_service import OptunaResearchVisualizationService

# NOWE IMPORTY
from .optuna_model_comparison_service import OptunaModelComparisonService
from .optuna_model_comparison_visualization_service import OptunaModelComparisonVisualizationService


class OptunaDashboardService:

    def generate_dashboard(self):

        loader = OptunaResultsLoaderService()

        df = loader.load_all_results()

        analytics = OptunaAnalyticsService(df)

        viz = OptunaVisualizationService()

        feature_imp = analytics.feature_importance()
        preprocessing = analytics.preprocessing_impact()
        horizon = analytics.horizon_impact()

        viz.plot_feature_importance(feature_imp)
        viz.plot_preprocessing(preprocessing)
        viz.plot_horizon(horizon)
        viz.plot_seq_len(df)
        viz.plot_loss_distribution(df)

        # =================================================
        # ADVANCED ANALYTICS
        # =================================================

        advanced = OptunaAdvancedAnalyticsService(df)

        research_viz = OptunaResearchVisualizationService()

        feature_table = advanced.build_feature_table()

        research_viz.feature_heatmap(feature_table)

        dataset_rank = advanced.dataset_ranking()

        research_viz.dataset_ranking(dataset_rank)

        top_combos = advanced.top_feature_combos()

        research_viz.top_feature_combos(top_combos)

        # =================================================
        # MODEL COMPARISON (SKLEARN VS LSTM)
        # =================================================

        comparison = OptunaModelComparisonService(df)

        comparison_viz = OptunaModelComparisonVisualizationService()

        sklearn_rank = comparison.sklearn_feature_ranking()

        lstm_rank = comparison.lstm_feature_ranking()

        comparison_viz.plot_sklearn_features(sklearn_rank)

        comparison_viz.plot_lstm_features(lstm_rank)

        # =================================================

        return {
            "result": "success",
            "rows_analyzed": len(df)
        }