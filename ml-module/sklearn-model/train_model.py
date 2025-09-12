from services.sklearn_train_service import SklearnTrainService
# do lokalnego trenowania
if __name__ == "__main__":
    train_service = SklearnTrainService()
    model_path = train_service.train_model(
        df_path="../shared/files/btc_train.csv",
        fe_version="fe_v3",
        target_version="t_v1",
        clf_version="c_v1"
    )
    print("Model zapisany w:", model_path)
