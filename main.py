import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf


_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}#components/get_data",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:         
            print("Running basic_cleaning step...")
            _ = mlflow.run(
                f"{config['main']['components_repository']}#src/basic_cleaning",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "input_artifact": config["basic_cleaning"]["input_artifact"],
                    "output_artifact": config["basic_cleaning"]["output_artifact"],
                    "output_type": config["basic_cleaning"]["output_type"],
                    "output_description": config["basic_cleaning"]["output_description"],
                    "min_price": float(config["basic_cleaning"]["min_price"]),
                    "max_price": float(config["basic_cleaning"]["max_price"]),
                },
            )

        if "data_check" in active_steps:
            # Run the data_check step
            _ = mlflow.run(
                f"{config['main']['components_repository']}#src/data_check",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "input_artifact": config["data_check"]["input_artifact"],
                    "ref_data": config["data_check"]["ref_data"],
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_rows": config["data_check"]["min_rows"],
                    "max_rows": config["data_check"]["max_rows"],
                    "min_price": config["data_check"]["min_price"],
                    "max_price": config["data_check"]["max_price"],
                },
            )

        if "data_split" in active_steps:
            mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "input": config["split"]["input_artifact"],  # Reference from config.yml
                    "test_size": config["split"]["test_size"],
                    "random_seed": config["split"]["random_seed"],
                    "stratify_by": config["split"]["stratify_by"],
                },
            )


        if "train_random_forest" in active_steps:
            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config_path = os.path.abspath("rf_config.json")
            with open(rf_config_path, "w+") as fp:
                rf_config = OmegaConf.to_container(config["modeling"]["random_forest"], resolve=True)
                json.dump(rf_config, fp)


            # Run the train_random_forest step with mlflow
            mlflow.run(
                os.path.join(config['main']['components_repository'], "train_random_forest"),
                "main",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": rf_config_path,
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": "random_forest_export",
                },
            )

        if "test_regression_model" in active_steps:
            run = wandb.init(job_type="test_model")

            logger.info("Downloading artifacts")

            # Download input artifact (model)
            model_local_path = run.use_artifact(config["test_regression_model"]["mlflow_model"]).download()

            # Download test dataset
            test_dataset_path = run.use_artifact(config["test_regression_model"]["test_dataset"]).file()

            # Read test dataset
            X_test = pd.read_csv(test_dataset_path)
            y_test = X_test.pop("price")

            logger.info("Loading model and performing inference on test set")
            sk_pipe = mlflow.sklearn.load_model(model_local_path)
            y_pred = sk_pipe.predict(X_test)

            logger.info("Scoring")
            r_squared = sk_pipe.score(X_test, y_test)
            mae = mean_absolute_error(y_test, y_pred)

            logger.info(f"Score: {r_squared}")
            logger.info(f"MAE: {mae}")

            # Log MAE and r2
            run.summary['r2'] = r_squared
            run.summary['mae'] = mae


if __name__ == "__main__":
    go()
