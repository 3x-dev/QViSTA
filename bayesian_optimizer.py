import optuna
import pandas as pd

def suggest_hyperparameters(trial):
    hidden_size = trial.suggest_int('hidden_size', 4, 32, step=4)
    num_heads = trial.suggest_int('num_heads', 2, 8, step=2)
    num_transformer_blocks = trial.suggest_int('num_transformer_blocks', 2, 6)
    mlp_hidden_size = trial.suggest_int('mlp_hidden_size', 4, 32, step=4)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])

    # Ensure hidden_size is divisible by num_heads
    if hidden_size % num_heads != 0:
        return None  # Skip this combination

    return {
        'hidden_size': hidden_size,
        'num_heads': num_heads,
        'num_transformer_blocks': num_transformer_blocks,
        'mlp_hidden_size': mlp_hidden_size,
        'batch_size': batch_size
    }

# Function to simulate an objective function for hyperparameter suggestion
def objective(trial):
    config = suggest_hyperparameters(trial)
    if config is None:
        raise optuna.exceptions.TrialPruned()
    return 1.0  # This dummy objective is only for suggesting hyperparameters, not for actual optimization

# Create an Optuna study
study = optuna.create_study(direction='minimize')

# Optimize to generate hyperparameters
study.optimize(objective, n_trials=100)

# Get the suggested hyperparameters
hyperparameter_list = [trial.params for trial in study.trials if trial.params]

# Convert to a DataFrame for review
hyperparameter_df = pd.DataFrame(hyperparameter_list)
print(hyperparameter_df)

# Save the configurations to a CSV file for review
hyperparameter_df.to_csv('suggested_hyperparameter_configurations.csv', index=False)
