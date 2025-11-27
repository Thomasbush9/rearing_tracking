import optuna
import torch
from behavex.models.train import Trainer

"""Script that contains the HPO class for the project"""

class HPO:
    def __init__(self, project):
        """Initialize HPO with project reference
        
        Args:
            project: Project instance containing config and training data
        """
        self.project = project
        self.study = None
    
    def _objective(self, trial):
        """Optuna objective function to optimize
        
        Args:
            trial: Optuna trial object
            
        Returns:
            float: Metric to optimize (e.g., validation loss, accuracy)
        """
        # Suggest hyperparameters
        hidden_size = trial.suggest_int("hidden_size", 8, 64, log=True)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        
        # Temporarily update project config with trial parameters
        original_config = self.project.config.copy()
        self.project.config["model_defaults"]["training"]["hidden_size"] = hidden_size
        self.project.config["model_defaults"]["training"]["lr"] = learning_rate
        self.project.config["model_defaults"]["training"]["batch_size"] = batch_size
        self.project.config["model_defaults"]["training"]["weight_decay"] = weight_decay
        
        # Create trainer with modified config (disable TensorBoard for HPO)
        trainer = Trainer(project=self.project, enable_tensorboard=False)
        
        # Train and get best test loss
        metric = trainer.train()
        if metric is None:
            metric = float('inf')
        
        # Restore original config
        self.project.config = original_config
        
        return metric
    
    def optimize(self, n_trials: int = 50, direction: str = "minimize"):
        """Run hyperparameter optimization
        
        Args:
            n_trials: Number of optimization trials
            direction: Optimization direction ("minimize" or "maximize")
        """
        self.study = optuna.create_study(direction=direction)
        self.study.optimize(self._objective, n_trials=n_trials)
        
        print(f"\nBest trial:")
        print(f"  Value: {self.study.best_value:.4f}")
        print(f"  Params: {self.study.best_params}")


