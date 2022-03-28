import math
import random
from collections import namedtuple, deque
from DQN import DQN
from environment import Environment

import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchsummary import summary

MemoryFormat = namedtuple('ReplayMemory',
                          ('state', 'action', 'next_state', 'reward'))


BATCH_SIZE = 256
GAMMA = 1
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000
TARGET_UPDATE = 10


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(MemoryFormat(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent(object):
    def __init__(self):
        self.env = Environment(False)
        self.env.reset()
        self.policy_net = DQN(self.env.n_features(), self.env.n_actions())
        self.target_net = DQN(self.env.n_features(), self.env.n_actions())
        print(self.policy_net)
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=1e-3)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0

    def load_checkpoint(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state, freeze_steps_done=False, use_net=False):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)
        if not freeze_steps_done:
            self.steps_done += 1
        if sample > eps_threshold or use_net:
            with torch.no_grad():
                out = self.policy_net(state)
                out = out.max(1)
                out = out[1]
                out = out.view(1, 1)
                return out
        else:
            return torch.tensor([[random.randrange(self.env.n_actions())]], dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        memos = self.memory.sample(BATCH_SIZE)
        batch = MemoryFormat(*zip(*memos))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)),dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        net_out = self.policy_net(state_batch)
        # 获取当前policy网络的动作对应的Q值
        state_action_values = net_out.gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # 获取target网络对下一状态的Q值估计最大的值，使用它乘以GAMMA，再加上即时Reward作为单步回报的估计值
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # 使用Huber损失函数，获得更稳定和平滑的loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # 进行一步训练
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            # 裁剪梯度，防止梯度爆炸或不稳定。
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    # 评估当前policy网络的优劣，返回数局游戏的平均回报和到达目标位置的次数
    def evaluate(self, eval_env, num_episodes=100):
        total_return = 0.0
        reach_target_count = 0
        random.seed(1)
        for _ in range(num_episodes):
            state = eval_env.reset()
            episode_return = 0.0
            while True:
                action = self.select_action(state, freeze_steps_done=True, use_net=True)
                new_state, reward, is_done, reach_target = eval_env.step(action.item())
                episode_return += reward
                state = new_state
                if reach_target:
                    reach_target_count += 1
                if is_done:
                    break
            total_return += episode_return

        avg_return = total_return / num_episodes
        random.seed(time.time())
        return avg_return, reach_target_count

    def copy_weights_to_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())