import optuna
import torch
import copy
from pathlib import Path
from behavex.models.train import Trainer

"""Script that contains the HPO class for the project"""

class HPO:
    def __init__(self, project, study_name: str = "hpo_study"):
        """Initialize HPO with project reference
        
        Args:
            project: Project instance containing config and training data
            study_name: Name for the Optuna study (used in database)
        """
        self.project = project
        self.study_name = study_name
        self.study = None
        
        # Create HPO directory in project
        self.hpo_dir = self.project.project_dir / "hpo"
        self.hpo_dir.mkdir(parents=True, exist_ok=True)
        
        # Database path
        self.db_path = self.hpo_dir / f"{study_name}.db"
    
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
        original_config = copy.deepcopy(self.project.config)
        self.project.config["model_defaults"]["training"]["hidden_size"] = hidden_size
        self.project.config["model_defaults"]["training"]["lr"] = learning_rate
        self.project.config["model_defaults"]["training"]["batch_size"] = batch_size
        self.project.config["model_defaults"]["training"]["weight_decay"] = weight_decay
        
        try:
            # Create trainer with modified config (disable TensorBoard and model saving for HPO)
            trainer = Trainer(project=self.project, enable_tensorboard=False, save_model=False)
            
            # Train and get best test loss
            metric = trainer.train()
            if metric is None:
                metric = float('inf')
        finally:
            # Always restore original config, even if training fails
            self.project.config = original_config
        
        return metric
    
    def optimize(self, n_trials: int = 50, direction: str = "minimize", storage: str = None):
        """Run hyperparameter optimization
        
        Args:
            n_trials: Number of optimization trials
            direction: Optimization direction ("minimize" or "maximize")
            storage: Storage URL (default: SQLite database in project_dir/hpo/)
        """
        # Use SQLite storage by default
        if storage is None:
            storage = f"sqlite:///{self.db_path}"
        
        # Create or load study with storage
        self.study = optuna.create_study(
            direction=direction,
            study_name=self.study_name,
            storage=storage,
            load_if_exists=True
        )
        
        print(f"Running HPO with {n_trials} trials...")
        print(f"Study saved to: {self.db_path}")
        
        self.study.optimize(self._objective, n_trials=n_trials)
        
        print(f"\nBest trial:")
        print(f"  Value: {self.study.best_value:.4f}")
        print(f"  Params: {self.study.best_params}")
        print(f"\nTo view in Optuna Dashboard, run:")
        print(f"  optuna dashboard {self.db_path}")


