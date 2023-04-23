import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN_Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


class CNN_Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class A2CTrainer:
    def __init__(self, actor, critic, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float, device=device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=device)
        action = torch.tensor(action, dtype=torch.long, device=device)
        reward = torch.tensor(reward, dtype=torch.float, device=device)
        done = torch.tensor(done, dtype=torch.bool, device=device)

        # Get the action probabilities from the policy network
        action_probs = self.actor(state)
        value = self.critic(state)

        # Compute the value loss
        target_value = reward + self.gamma * self.critic(next_state) * (~done).float()
        value_loss = self.criterion(target_value, value)

        # Compute the policy loss
        chosen_action_index = action.argmax(dim=1, keepdim=True)
        log_action_probs = torch.log(action_probs.gather(1, chosen_action_index))
        advantage = (target_value - value).detach()
        policy_loss = -(log_action_probs * advantage).mean()

        # Update the critic network
        self.optimizer_critic.zero_grad()
        value_loss.backward()
        self.optimizer_critic.step()

        # Update the actor network
        self.optimizer_actor.zero_grad()
        policy_loss.backward()
        self.optimizer_actor.step()

