import os
import sys
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from tqdm import tqdm
import optuna
from ray.tune import Callback
from ray.tune.suggest import SearchAlgorithm
from ray.tune.suggest import Searcher


# Custom callback to update the progress bar
class TqdmProgressCallback(Callback):
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="Hyperparameter Tuning Progress", unit="trial")
        self.total = total

    def on_trial_complete(self, iteration, trials, trial, **info):
        self.pbar.update(1)
        if len([t for t in trials if t.status.is_finished()]) == self.total:
            self.pbar.close()

    def on_trial_error(self, iteration, trials, trial, **info):
        print(f"Trial {trial.trial_id} failed with parameters: {trial.config} and error: {trial.error_file}")


# Define the training function
def train_transformer(config):
    try:
        sys.path.append('/content/drive/My Drive/Synopsys ISEF 23-24/Data')
        from quantum_transformers.training import train_and_evaluate
        from quantum_transformers.transformers import VisionTransformer
        from quantum_transformers.quantum_layer import get_circuit

        model = VisionTransformer(
            num_classes=4,
            patch_size=64,
            hidden_size=config['hidden_size'],
            num_heads=config['num_heads'],
            num_transformer_blocks=config['num_transformer_blocks'],
            mlp_hidden_size=config['mlp_hidden_size'],
            quantum_attn_circuit=get_circuit(),
            quantum_mlp_circuit=get_circuit(),
            pos_embedding='learn'
        )

        test_loss, test_auc, test_fpr, test_tpr, all_logits, all_labels = train_and_evaluate(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            num_classes=4,
            num_epochs=12,
        )

        tune.report(test_auc=test_auc)
    except Exception as e:
        print(f"Error in trial {config}: {e}")
        raise e


# Custom searcher to provide valid configurations
class ValidConfigSearcher(Searcher):
    def __init__(self, valid_configs):
        super(ValidConfigSearcher, self).__init__()
        self.valid_configs = valid_configs
        self.index = 0

    def suggest(self, trial_id):
        if self.index >= len(self.valid_configs):
            return None
        config = self.valid_configs[self.index]
        self.index += 1
        return config

    def on_trial_complete(self, trial_id, result=None, error=False):
        pass


# Define search space with reduced parameter ranges
search_space = {
    'hidden_size': tune.randint(4, 32),
    'num_heads': tune.randint(2, 6),
    'num_transformer_blocks': tune.randint(2, 6),
    'mlp_hidden_size': tune.randint(4, 32),
}


# Custom function to generate valid configurations
def generate_valid_configs(search_space, num_samples):
    valid_configs = []
    for _ in range(num_samples):
        config = {}
        for key, space in search_space.items():
            config[key] = space.sample()

        if config['hidden_size'] % config['num_heads'] == 0:
            valid_configs.append(config)
    return valid_configs


# Generate valid configurations
num_samples = 20
valid_configs = generate_valid_configs(search_space, num_samples)

# Create custom searcher
valid_config_searcher = ValidConfigSearcher(valid_configs)

# Define scheduler
asha_scheduler = ASHAScheduler(
    metric="test_auc",
    mode="max",
    max_t=50,
    grace_period=1,
    reduction_factor=2
)

# Create the custom progress bar callback
progress_callback = TqdmProgressCallback(total=num_samples)

# Run the tuning with the custom progress bar callback and valid configuration searcher
analysis = tune.run(
    train_transformer,
    resources_per_trial={"cpu": 2, "gpu": 1},
    search_alg=valid_config_searcher,
    num_samples=num_samples,
    scheduler=asha_scheduler,
    callbacks=[progress_callback]
)

# Print the best hyperparameters found
print(f'Best hyperparameters found were: {analysis.best_config}')
