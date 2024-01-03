# obstacle_avoidance_rl_model.py
# DQN Mdoel for Obstacle Avoidance

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQNModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class ObstacleAvoidanceRLModel:
    def __init__(self, input_size, output_size, learning_rate=0.001, gamma=0.99, epsilon=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the DQN model
        self.model = DQNModel(input_size, output_size).to(self.device)
        self.target_model = DQNModel(input_size, output_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # Define the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Define other hyperparameters
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration-exploitation trade-off

    def select_action(self, state):
        # Epsilon-greedy exploration strategy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.model.output_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.device)
                q_values = self.model(state_tensor)
                return torch.argmax(q_values).item()

    def update(self, state, action, next_state, reward, done):
        state_tensor = torch.FloatTensor(state).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        action_tensor = torch.tensor([action]).to(self.device)
        reward_tensor = torch.tensor([reward]).to(self.device)

        # Q-value prediction for the current state and action
        current_q_value = self.model(state_tensor).gather(1, action_tensor.unsqueeze(1))

        # Q-value prediction for the next state using the target network
        with torch.no_grad():
            max_next_q_value = self.target_model(next_state_tensor).max(1)[0].unsqueeze(1)
            target_q_value = reward_tensor + (1 - done) * self.gamma * max_next_q_value

        # Compute the Huber loss
        loss = nn.SmoothL1Loss()(current_q_value, target_q_value)

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        # Update the target network by copying the weights from the main model
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path):
        # Save the model parameters
        torch.save(self.model.state_dict(), 'models/dqn_obstacle_avoidance_model.pth')

    def load_model(self, path):
        # Load the model parameters
        self.model.load_state_dict(torch.load('models/dqn_obstacle_avoidance_model.pth'))

# Example usage:
# model = ObstacleAvoidanceRLModel(input_size=your_input_size, output_size=your_output_size)
# action = model.select_action(your_state)
# model.update(your_state, action, your_next_state, your_reward, your_done)
# model.update_target_network()
# model.save_model('models/dqn_obstacle_avoidance_model.pth')
