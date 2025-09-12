from services.sklearn_test_service import SklearnTestService
# do lokalnego testowania
if __name__ == "__main__":
    test_service = SklearnTestService()
    results = test_service.test_model(
        model_name="btc_model_fe_v3_t_v1_c_v1_1209_0.pkl",
        data_file="../../shared/files/btc_test_data/btc_test.csv",
        cv_splits=3,
        target_version="t_v1"
    )
    print("Wyniki testu:", results)
