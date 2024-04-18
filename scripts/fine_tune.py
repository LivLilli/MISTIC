from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
from typing import Dict, Any, Union
from optuna import Trial
import os

train_path = os.path.join('..', 'data', 'train_data.csv')
test_path = os.path.join('..', 'data', 'test_data.csv')
output_path = os.path.join('../results')
dataset = load_dataset("csv", data_files={"train": train_path, "test": test_path})
model_id = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def model_init(params: Dict[str, Any]) -> SetFitModel:
    params = params or {}
    max_iter = params.get("max_iter", 100)
    solver = params.get("solver", "liblinear")
    params = {
        "head_params": {
            "max_iter": max_iter,
            "solver": solver,
        }
    }
    return SetFitModel.from_pretrained(model_id, **params)



def hp_space(trial: Trial) -> Dict[str, Union[float, int, str]]:

    return {
        "body_learning_rate": trial.suggest_float("body_learning_rate", 1e-6, 1e-3, log=True),
        "num_epochs": trial.suggest_int("num_epochs", 2, 4),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "seed": trial.suggest_int("seed", 1, 40),
        "max_iter": trial.suggest_int("max_iter", 20, 30, 40),
        "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear"]),
    }

trainer = SetFitTrainer(
    model_init=model_init,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    loss_class=CosineSimilarityLoss,
    metric="accuracy",
    column_mapping={"splitted_text": "text", "CATEGORY": "label"} # Map dataset columns to text
)

best_run = trainer.hyperparameter_search(direction="maximize", hp_space=hp_space, n_trials=10)
trainer.apply_hyperparameters(best_run.hyperparameters, final_model=True) # replaces model_init with a fixed model
trainer.train()
metrics = trainer.evaluate()
trainer.model.save_pretrained(output_path)
