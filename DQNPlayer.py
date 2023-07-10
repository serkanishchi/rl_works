from typing import Dict, List
from Board import *
from Player import *
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple, deque
import random
import math

import matplotlib
import matplotlib.pyplot as plt

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return F.softmax(x)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class DQNPlayer(Player):
    def __init__(self, win_value: float = 1.0, draw_value: float = 0.0,
                 loss_value: float = -1.0, training: bool = True, pre_training_games=500,
                 GAMMA=0.95, EPS_START = 0.95, EPS_END = 0.00, EPS_DECAY = 1000, TAU = 0.001, LR = 1e-3, BATCH_SIZE = 128):

        self.win_value = win_value
        self.draw_value = draw_value
        self.loss_value = loss_value
        self.side = None
        self.states = []
        self.actions = []

        self.q_network = DQN(BOARD_DIM*BOARD_DIM*3, BOARD_DIM*BOARD_DIM).to(device)
        self.q_network.apply(init_weights)
        self.target_network = DQN(BOARD_DIM*BOARD_DIM*3, BOARD_DIM*BOARD_DIM).to(device)
        self.target_network.apply(init_weights)
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=LR, amsgrad=True)
        self.training = training

        self.GAMMA = GAMMA
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.TAU =TAU
        self.LR = LR
        self.BATCH_SIZE = BATCH_SIZE

        self.replay_buffer = ReplayMemory(10000)

        self.game_counter = 0
        self.pre_training_games = pre_training_games
        self.losses = []
        self.steps_done = 0

        super().__init__()

    def new_game(self, side: int):
        self.side = side
        self.states = []
        self.actions = []

    def add_game_to_replay_buffer(self, reward: float):
        game_length = len(self.actions)
        for i in range(game_length - 1):
            state = self.get_state_inputs(self.states[i])
            action = self.actions[i][0]*BOARD_DIM+self.actions[i][1]
            action = torch.tensor([[action]], device=device)
            next_state = self.get_state_inputs(self.states[i+1])
            r = torch.tensor([0], device=device)
            self.replay_buffer.push(state,action,next_state, r)

        final_state = self.get_state_inputs(self.states[game_length - 1])
        final_action = self.actions[game_length - 1][0] * BOARD_DIM + self.actions[game_length - 1][1]
        final_action = torch.tensor([[final_action]], device=device)

        final_r = torch.tensor([reward], device=device)
        self.replay_buffer.push(final_state, final_action, None, final_r)

    def get_state_inputs(self,state):
        flatten_state = state.flatten()
        res = np.array([(flatten_state == self.side).astype(int),
                        (flatten_state == Board.other_side(self.side)).astype(int),
                        (flatten_state == EMPTY).astype(int)])
        inputs = res.reshape(-1)
        inputs = torch.tensor(inputs, dtype=torch.float32, device=device).unsqueeze(0)
        return inputs

    def get_action(self, state: [np.ndarray]):
        inputs = self.get_state_inputs(state)
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action = self.q_network(inputs).max(1)[1].view(1, 1)

        a = action.detach().cpu().numpy().flatten()[0]

        return (a//BOARD_DIM, a%BOARD_DIM)

    def move(self, board: Board) -> (GameResult, bool):
        self.states.append(board.state.copy())
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        action = self.get_action(board.state)
        if board.state[action]!=EMPTY:
            action = board.random_empty_spot()
        if self.training and random.random()<eps_threshold:
            action = board.random_empty_spot()

        self.actions.append(action)

        # We execute the move and return the result
        _, res, finished = board.move(action, self.side)

        return res, finished

    def optimize_model(self):
        if len(self.replay_buffer) < self.BATCH_SIZE:
            return
        transitions = self.replay_buffer.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to behavior_policy
        state_action_values = self.q_network(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_policy; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.losses.append(loss)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
        self.optimizer.step()

    def final_result(self, result: GameResult):
        self.game_counter += 1

        # Compute the final reward based on the game outcome
        if (result == GameResult.NAUGHT_WIN and self.side == NAUGHT) or (
                result == GameResult.CROSS_WIN and self.side == CROSS):
            reward = self.win_value  # type: float
        elif (result == GameResult.NAUGHT_WIN and self.side == CROSS) or (
                result == GameResult.CROSS_WIN and self.side == NAUGHT):
            reward = self.loss_value  # type: float
        elif result == GameResult.DRAW:
            reward = self.draw_value  # type: float
        else:
            raise ValueError("Unexpected game result {}".format(result))

        self.add_game_to_replay_buffer(reward)

        # If we are in training mode we run the optimizer.
        if self.training and (self.game_counter > self.pre_training_games):
            # Perform one step of the optimization (on the policy network)
            self.optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_network_state_dict = self.target_network.state_dict()
            q_network_state_dict = self.q_network.state_dict()
            for key in q_network_state_dict:
                target_network_state_dict[key] = q_network_state_dict[key] * self.TAU + target_network_state_dict[
                    key] * (1 - self.TAU)
            self.target_network.load_state_dict(target_network_state_dict)

            if (self.game_counter%100)==0:
                self.plot_losses()

    def plot_losses(self,show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.losses, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 250:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())