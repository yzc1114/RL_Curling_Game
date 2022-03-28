import math
from itertools import count
from environment import Environment
from agent import Agent, TARGET_UPDATE
import matplotlib.pyplot as plt

import torch


def train():
    eval_env = Environment(False)
    num_episodes = 3000
    total_step = 0
    eval_episodes = 10
    best_avg_return = -math.inf
    best_reach_target = 0
    agent = Agent()
    his_avg_return = []
    his_reach_target = []
    for i_episode in range(num_episodes):
        state = agent.env.reset()
        if i_episode % eval_episodes == 0:
            avg_return, reach_target_count = agent.evaluate(eval_env, num_episodes=100)
            print(f"\ntotal step = {'%.3f'%total_step}, done episode = {'%d'%i_episode}, "
                  f"eval avg return = {'%.3f'%avg_return}, current best avg return {'%.3f'%best_avg_return}, "
                  f"eval reach target = {reach_target_count}, current best reach target count {best_reach_target}")
            his_avg_return.append(avg_return)
            his_reach_target.append(reach_target_count)
            if reach_target_count >= best_reach_target:
                torch.save(agent.policy_net.state_dict(), "checkpoints/policy-checkpoint.ckpt")
                print(f"Best reach_target_count updated from {best_reach_target} to {reach_target_count}")
                print(f"Best avg return updated from {best_avg_return} to {avg_return}")
                best_avg_return = avg_return
                best_reach_target = reach_target_count
        print(f"\nnew episode {i_episode}")
        for t in count():
            action = agent.select_action(state)
            total_step += 1
            next_state, reward, done, reach_target = agent.env.step(action.item())
            reward = torch.tensor([reward])

            if done:
                next_state = None

            agent.memory.push(state, action, next_state, reward)

            state = next_state

            print(f"\roptimize model, current episode {i_episode}, inner step = {t}, total step = {total_step}", end='')
            agent.optimize_model()
            if done:
                break
        if i_episode % TARGET_UPDATE == 0:
            agent.copy_weights_to_target_net()
    show_avg_return_fig(avg_returns=his_avg_return)
    show_reach_targets(reach_targets=his_reach_target)


def show_avg_return_fig(avg_returns):
    plt.plot(avg_returns)
    plt.xlabel('episodes($10^-1$)')
    plt.title('avg return')
    plt.savefig('avg_return_fig.png')
    plt.show()


def show_reach_targets(reach_targets):
    plt.plot(reach_targets)
    plt.xlabel('episodes($10^-1$)')
    plt.title('reach target count(%)')
    plt.savefig('reach_target_fig.png')
    plt.show()


def evaluate():
    agent = Agent()
    agent.policy_net.load_state_dict(torch.load("checkpoints_certain/policy-checkpoint.ckpt"))
    eval_env = Environment(False)
    avg_return, reach_target_count = agent.evaluate(eval_env, num_episodes=1000)
    print(f"avg_return: {avg_return}, reach_target_count {reach_target_count}")


if __name__ == '__main__':
    # train()
    evaluate()
