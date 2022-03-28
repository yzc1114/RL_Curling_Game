import numpy as np
import pygame
import sys
import random
import torch

WIDTH = 1000
HEIGHT = 1000
BALL_DIAMETER = 20
BALL_MASS = 1
TARGET_SQUARE_WIDTH = 10
UNIT = 10
MAX_STEP = 300


class Actions:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    action_to_force_map = {
        UP: [0., -5.],
        DOWN: [0., 5.],
        LEFT: [-5., 0.],
        RIGHT: [5., 0.]
    }

    action_to_text_map = {
        None: "None",
        UP: "UP",
        DOWN: "DOWN",
        LEFT: "LEFT",
        RIGHT: "RIGHT"
    }

    @classmethod
    def action_to_force(cls, action):
        return cls.action_to_force_map[action]

    @classmethod
    def n_actions(cls):
        return len(cls.action_to_force_map)

    @classmethod
    def sample(cls):
        return random.randint(0, cls.n_actions() - 1)

    @classmethod
    def force_to_action_text(cls, force):
        force_to_action_map = {
            cls._hase_force(v): k for k, v in cls.action_to_force_map.items()
        }
        force_to_action_map[None] = None
        action = force_to_action_map[cls._hase_force(force)]
        return cls.action_to_text_map[action]

    @classmethod
    def _hase_force(cls, force):
        if force is None:
            return None
        return 100*force[0] + force[1]


class Observation:
    def __init__(self, center_position, v, target_center_position):
        self.relative_target_position_x = target_center_position[0] - center_position[0]
        self.relative_target_position_y = target_center_position[1] - center_position[1]
        self.to_left = center_position[0] - BALL_DIAMETER / 2
        self.to_right = WIDTH - center_position[0] - BALL_DIAMETER / 2
        self.to_top = center_position[1] - BALL_DIAMETER / 2
        self.to_bottom = HEIGHT - center_position[1] - BALL_DIAMETER / 2
        self.speed_x = v[0]
        self.speed_y = v[1]
        self.max_speed = 32.85

    def normalized_vector(self):
        return torch.tensor([
            self.relative_target_position_x/WIDTH,
            self.relative_target_position_y/HEIGHT,
            # self.to_left/WIDTH,
            # self.to_right/WIDTH,
            # self.to_top/HEIGHT,
            # self.to_bottom/HEIGHT,
            self.speed_x/self.max_speed,
            self.speed_y/self.max_speed
        ]).unsqueeze(0)

    @classmethod
    def n_features(cls):
        return 4


class ShadowRect:
    def __init__(self, rect=None):
        self.width = None
        self.height = None
        self.left = None
        self.top = None
        self.bottom = None
        self.right = None
        if rect is not None:
            self.init_from_rect(rect)

    def init_from_rect(self, rect):
        self.width = rect.width
        self.height = rect.height
        self.left = float(rect.left)
        self.top = float(rect.top)
        self.bottom = float(rect.bottom)
        self.right = float(rect.right)

    def move(self, direction) -> 'ShadowRect':
        r = ShadowRect()
        r.width = self.width
        r.height = self.height
        r.top = self.top + direction[1]
        r.bottom = self.bottom + direction[1]
        r.left = self.left + direction[0]
        r.right = self.right + direction[0]
        return r

    def center(self):
        return self.left + self.width / 2, self.top + self.height / 2


class Ball:
    def __init__(self, image=None, init_position=(500, 500), init_v=(0, 5)):
        self.image = image
        if image is not None:
            self.rect = self.image.get_rect()
            self.rect.left = init_position[0]
            self.rect.top = init_position[1]
        else:
            self.rect = pygame.rect.Rect(init_position[0], init_position[1], BALL_DIAMETER, BALL_DIAMETER)
        self.v = list(init_v)
        self.force = None
        self.mass = BALL_MASS
        self.shadow_rect = ShadowRect(rect=self.rect)

    def update(self, force=(-5., -5.), pass_time=0.01):
        distance = [UNIT * vi * pass_time for vi in self.v]
        self.force = force
        self.update_frictional_force()
        self.shadow_rect = self.shadow_rect.move(distance)
        self.check_collision()
        self.update_v(pass_time)
        self.rect.x = self.shadow_rect.left
        self.rect.y = self.shadow_rect.top

    def update_v(self, pass_time=0.01):
        for i, f in enumerate(self.force):
            a = f/self.mass
            self.v[i] = self.v[i] + a * pass_time

    def update_frictional_force(self):
        frictional_force = [-0.005 * (vi**2) for vi in self.v]
        self.force = [f + frictional_force[i] for i, f in enumerate(self.force)]

    def check_collision(self):
        rect = self.shadow_rect
        if rect.left <= 0 or rect.right >= WIDTH:
            self.v[0] = -0.9 * self.v[0]
        if rect.top <= 0 or rect.bottom >= HEIGHT:
            self.v[1] = -0.9 * self.v[1]
        if rect.left < 0:
            rect.left = -rect.left
        if rect.right > WIDTH:
            rect.left -= 2 * (rect.right - WIDTH)
        if rect.top < 0:
            rect.top = - rect.top
        if rect.bottom > HEIGHT:
            rect.top -= 2 * (rect.bottom - HEIGHT)
        rect.bottom = rect.top + rect.height
        rect.right = rect.left + rect.width
        self.shadow_rect = rect

    def get_v(self):
        return self.v[0], self.v[1]


class Environment:
    BLACK = (0, 0, 0)

    def __init__(self, render_mode):
        self.render_mode = render_mode
        if render_mode:
            pygame.init()
            pygame.font.init()
            self.font = pygame.font.SysFont('Console', 14)
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.ball_image = pygame.image.load("intro_ball.gif").convert()
            self.ball_image = pygame.transform.scale(self.ball_image, (20, 20))
        else:
            self.font = None
            self.screen = None
            self.ball_image = None

        self.target_position = None
        self.target_center = None
        self.ball = None
        self.current_step = 0
        self.step_force = None

    @classmethod
    def n_features(cls):
        return Observation.n_features()

    @classmethod
    def n_actions(cls):
        return Actions.n_actions()

    @classmethod
    def sample_action(cls):
        return Actions.sample()

    def render(self):
        if not self.render_mode:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        self.screen.fill(self.BLACK)
        self.screen.blit(self.ball.image, self.ball.rect)
        self.draw_target()
        text_surface = self.font.render(self.get_text(), False, (255, 255, 255))
        self.screen.blit(text_surface, (0, 0))
        pygame.display.flip()

    def get_text(self):
        force_x = 0
        force_y = 0
        if self.ball.force is not None:
            force_x = self.ball.force[0]
            force_y = self.ball.force[1]
        speed_x = 0
        speed_y = 0
        if self.ball.v is not None:
            speed_x = self.ball.v[0]
            speed_y = self.ball.v[1]

        return "second: %4.2f, step: %2d, force: [%5.2f, %5.2f], action: %6s, speed: [%5.2f, %5.2f]" % (
            self.current_step / 10, self.current_step, force_x, force_y,
            Actions.force_to_action_text(self.step_force), speed_x, speed_y,
        )

    @staticmethod
    def _rand_ball_position():
        # 随机生成球的起始位置，画布范围0-1000，球的半径为10，生成球的左上角的坐标，范围为0-980
        x = random.randint(0, WIDTH - BALL_DIAMETER)
        y = random.randint(0, HEIGHT - BALL_DIAMETER)
        return x, y

    @staticmethod
    def _rand_target_position():
        # 随机生成目标位置，画布范围0-1000，目标位置方块的宽度为10，为避免球的半径达不到目标点的情况，要避免点生成在边缘角落
        # 所以将目标位置的中心限制在5-995之间，而生成时需要生成左上角的点的位置，所以限制为0-990
        x = random.randint(0, WIDTH - TARGET_SQUARE_WIDTH)
        y = random.randint(0, WIDTH - TARGET_SQUARE_WIDTH)
        return x, y

    @staticmethod
    def _rand_init_speed():
        v1 = 20 * random.random() - 10
        v2 = 20 * random.random() - 10
        # v2 = 20 * np.random.rand() - 10
        # v1 = 20 * np.random.rand() - 10
        return v1, v2

    def draw_target(self):
        pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(self.target_position[0], self.target_position[1], TARGET_SQUARE_WIDTH, TARGET_SQUARE_WIDTH))

    def reset(self):
        ball_position = self._rand_ball_position()
        self.target_position = self._rand_target_position()
        self.target_center = self.target_position[0] + TARGET_SQUARE_WIDTH / 2, self.target_position[1] + TARGET_SQUARE_WIDTH / 2
        self.ball = Ball(self.ball_image, init_position=ball_position, init_v=self._rand_init_speed())
        self.current_step = 0
        return self.observation().normalized_vector()

    # 每个step执行经过0.1秒，而模拟运动学需要每隔0.01秒运行，所以在内部运行十次update和render
    def step(self, action):
        force = Actions.action_to_force(action)
        self.step_force = force
        for i in range(10):
            self.ball.update(force=force, pass_time=0.01)
            self.render()
        reward = self.reward()
        self.current_step += 1
        done, reach_target = self.is_done()
        return self.observation().normalized_vector(), self.normalize_reward(reward), done, reach_target

    def observation(self):
        return Observation(self.ball.shadow_rect.center(), self.ball.v, self.target_center)

    @staticmethod
    def normalize_reward(reward):
        max_d = np.sqrt(WIDTH ** 2 + HEIGHT ** 2)
        # normalize reward to [0, 1)
        return reward / max_d

    def reward(self):
        # reward为球的中心点到目标中心点的距离的负值
        return -self.distance_to_target()

    def distance_to_target(self):
        ball_center = self.ball.shadow_rect.center()
        target_center = self.target_position[0] + TARGET_SQUARE_WIDTH / 2, self.target_position[1] + TARGET_SQUARE_WIDTH / 2
        d = np.sqrt(np.sum([(ball_center[i] - target_center[i]) ** 2 for i in range(2)]))
        return d

    def is_done(self):
        d = self.distance_to_target()
        # 距离小于球的半径时，结束
        if d < BALL_DIAMETER / 2:
            return True, True
        # 最多执行300步
        if self.current_step >= MAX_STEP:
            return True, False
        return False, False


def do_simulation():
    from agent import Agent
    agent = Agent()
    agent.load_checkpoint("checkpoints_certain/policy-checkpoint.ckpt")
    agent.env = Environment(render_mode=True)
    total_step = 0
    for i in range(10):
        state = agent.env.reset()
        # state = env.reset()
        while True:
            agent.env.render()

            action = agent.select_action(state, use_net=True).item()
            print(f"current step {total_step}, action = {action}, force = {Actions.action_to_force(action)}")

            next_state, reward, done, reach_target = agent.env.step(action)

            state = next_state

            if done:
                break

            total_step += 1

            print("total_step: ", total_step)


if __name__ == '__main__':
    do_simulation()
