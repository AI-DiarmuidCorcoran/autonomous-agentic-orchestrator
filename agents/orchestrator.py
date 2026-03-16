import torch
from core.learner import ContinualLearner

class AgenticOrchestrator:
    """
    Manages multiple agents and their continual learning cycles.
    Designed for industrial optimization tasks where objectives evolve.
    """
    def __init__(self, config):
        self.config = config
        self.learner = ContinualLearner(
            input_size=config.get('input_size', 10),
            hidden_size=config.get('hidden_size', 64),
            output_size=config.get('output_size', 2)
        )
        self.optimizer = torch.optim.Adam(self.learner.parameters(), lr=1e-3)

    def optimize_system(self, state, target):
        """
        Executes a single optimization step using current knowledge.
        """
        self.learner.train()
        self.optimizer.zero_grad()
        
        output = self.learner(state)
        # Standard MSE loss + EWC penalty
        criterion = torch.nn.MSELoss()
        main_loss = criterion(output, target)
        ewc_penalty = self.learner.ewc_loss()
        
        total_loss = main_loss + ewc_penalty
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()

    def transition_to_new_task(self, data_samples):
        """
        Prepares the agent for a new task by calculating weight importance.
        """
        self.learner.estimate_fisher(data_samples)
        print("Task transition complete. Fisher matrix updated.")
