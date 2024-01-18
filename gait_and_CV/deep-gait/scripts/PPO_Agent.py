import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PPOAgent:
    def __init__(self, policy, optimizer, clip_param, value_loss_coef, entropy_coef):
        self.policy = policy
        self.optimizer = optimizer
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

    def act(self, state):
        # Sample action from the policy
        with torch.no_grad():
            action_probs = self.policy(state)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()

        return action

    def update(self, states, actions, rewards):
        # Convert lists to PyTorch tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.FloatTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)

        # Compute advantages and returns (you might need to customize this based on your reward function)
        advantages = rewards_tensor - self.policy.value_function(states_tensor).detach()
        returns = rewards_tensor

        # Compute the policy loss
        action_probs = self.policy(states_tensor)
        action_dist = Categorical(action_probs)
        log_probs = action_dist.log_prob(actions_tensor)
        ratio = torch.exp(log_probs - log_probs.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Compute the value function loss
        value_loss = nn.MSELoss()(self.policy.value_function(states_tensor), returns)

        # Compute the entropy loss
        entropy_loss = -self.entropy_coef * action_dist.entropy().mean()

        # Compute the total loss
        total_loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss

        # Update the policy
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def save_model(self, path):
        # Save the model parameters
        torch.save(self.policy.state_dict(), 'models/gait_controller_model.pth')

    def load_model(self, path):
        # Load the model parameters
        self.policy.load_state_dict(torch.load('models/gait_controller_model.pth'))
