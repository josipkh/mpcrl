"""Requires Optuna: https://optuna.org/"""

try:
    import optuna
except:
    raise ImportError("optuna is required")
from dpg_vehicle import main as dpg_main
from test_configs import experiment_configs

experiment_config = experiment_configs["test"].copy()
experiment_config["show_plots"] = False

def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)
    experiment_config["learning_rate"] = learning_rate
    experiment_config["episodes"] = 10
    print(f"Learning rate: {learning_rate}")
    return dpg_main(dpg_config=experiment_config)

study = optuna.create_study()
study.optimize(objective, n_trials=10)

values = [trial.values[0] for trial in study.trials]
learning_rates = [trial.params["learning_rate"] for trial in study.trials]

print(f"{'Values':>10}  {'LRs':>10}")
print("-" * 23)
for a, b in zip(values, learning_rates):
    print(f"{a:10.6f}  {b:10.6f}")

print("Best parameters:")
print(study.best_params)

