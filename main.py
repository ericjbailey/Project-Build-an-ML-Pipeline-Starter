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
    "test_regression_model"
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
            mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "sample": config["download"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:         
            print("Running basic_cleaning step...")
            mlflow.run(
                f"{config['main']['src_repository']}/basic_cleaning",
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
            mlflow.run(
                f"{config['main']['src_repository']}/data_check",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "input_artifact": config["data_check"]["input_artifact"],
                    "output_artifact": config["data_check"]["output_artifact"],
                    "kl_threshold": str(config["data_check"]["kl_threshold"]),
                    "min_rows": config["data_check"]["min_rows"],
                    "max_rows": config["data_check"]["max_rows"],
                    "min_price": str(config["data_check"]["min_price"]),
                    "max_price": str(config["data_check"]["max_price"]),
                },
            )

        if "data_split" in active_steps:
            mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "input_artifact": config["split"]["input_artifact"],  # Reference from config.yml
                    "test_size": config["split"]["test_size"],
                    "random_seed": config["split"]["random_seed"],
                    "stratify_by": config["split"]["stratify_by"],
                },
            )


        if "train_random_forest" in active_steps:
            # Extract the random forest config
            rf_config = OmegaConf.to_container(config["modeling"]["random_forest"], resolve=True)

            # Build hydra_options string from any list-type hyperparameters
            hydra_options = []
            for param, value in rf_config.items():
                if isinstance(value, list):
                    hydra_options.append(f"modeling.random_forest.{param}=" + ",".join(map(str, value)))
            hydra_options_str = " ".join(hydra_options)

            # Remove list-type params from rf_config before saving to JSON
            single_value_rf_config = {k: v for k, v in rf_config.items() if not isinstance(v, list)}

            # Serialize cleaned rf_config to JSON
            rf_config_path = os.path.abspath("rf_config.json")
            with open(rf_config_path, "w+") as fp:
                json.dump(single_value_rf_config, fp)

            print(f"Running train_random_forest step with hydra_options: {hydra_options_str}")

            # Run the train_random_forest step with mlflow
            mlflow.run(
                f"{config['main']['src_repository']}/train_random_forest",
                "main",
                version="main",
                env_manager="conda",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": rf_config_path,
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": "random_forest_export",
                    "hydra_options": hydra_options_str,  # ✅ Pass sweep options here
                },
            )

        if "test_regression_model" in active_steps:
            mlflow.run(
                f"{config['main']['components_repository']}/test_regression_model",
                "main",
                version="main",
                env_manager="conda",
                parameters={
                    "mlflow_model": config["modeling"]["mlflow_model"],
                    "test_dataset": config["modeling"]["test_dataset"],
                },
            )


if __name__ == "__main__":
    go()
