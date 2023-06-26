from typing import Optional

import numpy as np
from numpy import cos, pi, sin

from gym import core, spaces
from gym.error import DependencyNotInstalled
import math
import random

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = [
    "Alborz Geramifard",
    "Robert H. Klein",
    "Christoph Dann",
    "William Dabney",
    "Jonathan P. How",
]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann <cdann@cdann.de>"

# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py

class ConstrainedReacherEnv(core.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    dt = 0.2

    LINK_LENGTH_1 = 1.0  # [m]
    LINK_LENGTH_2 = 1.0  # [m]
    LINK_MASS_1 = 1.0  #: [kg] mass of link 1
    LINK_MASS_2 = 1.0  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.0  #: moments of inertia for both links

    MAX_VEL_1 = 1 * pi
    MAX_VEL_2 = 1 * pi

    AVAIL_TORQUE = [-1.0, 0.0, +1.0]

    torque_noise_max = 0.0

    SCREEN_DIM = 500

    #: use dynamics equations from the nips paper or the book
    book_or_nips = "book"
    action_arrow = None
    domain_fig = None
    actions_num = 3

    def __init__(self, min_torque=-1.0, max_torque=1.0, g=0.0, target=None, max_steps=200, epsilon=0.1,
                 reset_target_reached=False, bonus_reward=False, barrier_type='circle', location=(1.5, 1.5), radius=0.5,
                 lambda_cbf=0.5, initial_state=None):
        self.screen = None
        self.clock = None
        self.isopen = True
        # high = np.array(
        #     [1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2, 2.0, 2.0, 2.0, 2.0], dtype=np.float32
        # )
        high = np.array(
            [1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2, 2.0, 2.0], dtype=np.float32
        )
        low = -high
        self.min_torque = min_torque
        self.max_torque = max_torque
        min_torque = np.array([min_torque, min_torque], dtype=np.float32)
        max_torque = np.array([max_torque, max_torque], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=min_torque, high=max_torque, dtype=np.float32)
        self.state = None
        self.g = g

        self.lambda_cbf = lambda_cbf
        if barrier_type == 'circle':
            self.x0 = location[0]
            self.y0 = location[1]
            self.r = radius
        else:
            raise Exception("Not implemented")

        if target:
            self.target = target
            self.random_target = False
        else:
            is_valid = False
            while not is_valid:
                self.target = self._sample_target()
                is_valid = self._check_target(self.target)
            self.random_target = True
        self.max_steps = max_steps
        self.timestep = 0
        self.epsilon = epsilon
        self.reset_target_reached = reset_target_reached
        self.bonus_reward = bonus_reward
        self.initial_state = initial_state

    def h(self, x, y):
        a = np.linalg.norm(np.array([x, y] - np.array([self.x0, self.y0]))) - np.square(self.r)
        return a

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ):
        super().reset(seed=seed)
        is_valid = False
        if not self.random_target:
            self.state = self.initial_state
        else:
            while not is_valid:
                self.state = self.np_random.uniform(low=-np.pi, high=np.pi, size=(4,)).astype(np.float32)
                is_valid = self._check_initial_pos(self.state)
        self.timestep = 0
        if self.random_target:
            is_valid = False
            while not is_valid:
                self.target = self._sample_target()
                is_valid = self._check_target(self.target)
        self.target_reached = False
        if not return_info:
            return self._get_ob()
        else:
            return self._get_ob(), {}

    def _sample_target(self):
        # random angle
        alpha = 2 * math.pi * random.random()
        # random radius
        r = 2.0 * math.sqrt(random.random())
        # calculating coordinates
        x = r * math.cos(alpha)
        y = r * math.sin(alpha)
        return (x, y)

    def _check_target(self, target):
        if self.h(target[0], target[1]) <= 0.0:
            print("Invalid target! Inside the unsafe region!")
            return False
        else:
            print("Valid target found!")
            return True

    def _check_initial_pos(self, state):
        _, p2 = self._get_coordinates(state)
        if self.h(p2[0], p2[1]) <= 0.0:
            print("Invalid initial position! Inside the unsafe region!")
            return False
        else:
            print("Valid initial position found!")
            return True

    def step(self, a):
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        torque = a

        #add torque limit
        torque = np.clip(torque, a_min=self.min_torque, a_max=self.max_torque)

        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(
                -self.torque_noise_max, self.torque_noise_max
            )

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.append(s, torque)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])

        ns[0] = wrap(ns[0], -pi, pi)
        ns[1] = wrap(ns[1], -pi, pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.prev_state = self.state
        self.state = ns

        done = self._is_done()
        reward = self._get_reward()
        alpha = self._get_barrier_value(self.prev_state, self.state)
        # if alpha >= 0.0:
        #     alpha = 0.0
        #print("alpha", alpha)

        self.timestep += 1
        return (self._get_ob(), reward, done, alpha, self.target_reached)

    def _get_ob(self):
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        return np.array(
            [cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3], self.target[0], self.target[1]], dtype=np.float32
        )

    def _get_coordinates(self, state):
        p1 = [-self.LINK_LENGTH_1 * cos(state[0]), self.LINK_LENGTH_1 * sin(state[0]),]
        p2 = [p1[0] - self.LINK_LENGTH_2 * cos(state[0] + state[1]), p1[1] + self.LINK_LENGTH_2 * sin(state[0] + state[1]),]
        return p1, p2

    def _get_distance_to_target(self):
        _, p2 = self._get_coordinates(self.state)
        distance = np.linalg.norm(np.array(p2) - np.array(self.target))
        # print("=======")
        # print("distance", distance)
        # if distance <= self.epsilon:
        #     print("target reached!")
        return distance

    def _get_reward(self):
        distance = self._get_distance_to_target()
        reward = -distance
        if self.bonus_reward and distance <= self.epsilon:
            reward += 20.0
        return reward

    def _get_barrier_value(self, state, next_state):
        _, p2 = self._get_coordinates(state)
        h = self.h(p2[0], p2[1])
        _, p2 = self._get_coordinates(next_state)
        h_next = self.h(p2[0], p2[1])
        a = h_next - (1 - self.lambda_cbf)*h
        if a < 0:
            a = 1.0
            print("Safety constraint violated!")
        else:
            a = 0.0
        return a

    def _is_done(self):
        if self.reset_target_reached:
            distance = self._get_distance_to_target()
            if distance <= self.epsilon:
                self.target_reached = True
                print("Target Reached!!!")
                return True
            if self.timestep == self.max_steps - 1:
                print("Timeout!!!")
                return True
            if distance > self.epsilon and self.timestep < self.max_steps - 1:
                return False
        else:
            if self.timestep == self.max_steps - 1:
                print("Timeout!!!")
                return True
            else:
                return False
    def _dsdt(self, s_augmented):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = self.g
        a1 = s_augmented[-2]
        a2 = s_augmented[-1]
        s = s_augmented[:-1]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        d1 = (
            m1 * lc1**2
            + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * cos(theta2))
            + I1
            + I2
        )
        d2 = m2 * (lc2**2 + l1 * lc2 * cos(theta2)) + I2
        phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2**2 * sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2)
            + phi2
        )
        if self.book_or_nips == "nips":
            # the following line is consistent with the description in the
            # paper
            ddtheta2 = (a2 + d2 / d1 * phi1 - phi2) / (m2 * lc2**2 + I2 - d2**2 / d1)
        else:
            # the following line is consistent with the java implementation and the
            # book
            ddtheta2 = (
                a2 + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * sin(theta2) - phi2
            ) / (m2 * lc2**2 + I2 - d2**2 / d1)
        ddtheta1 = -(a1 + d2 * ddtheta2 + phi1) / d1
        return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0)

    def render(self, mode="human"):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.SCREEN_DIM, self.SCREEN_DIM))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        self.surf.fill((255, 255, 255))
        s = self.state

        bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
        scale = self.SCREEN_DIM / (bound * 2)
        offset = self.SCREEN_DIM / 2

        if s is None:
            return None

        safe_set = (scale * self.x0 + offset, scale * self.y0 + offset)
        pygame.gfxdraw.filled_circle(self.surf, int(safe_set[0]), int(safe_set[1]), int(scale * self.r), (255, 64, 64)) # red

        p1 = [
            -self.LINK_LENGTH_1 * cos(s[0]) * scale,
            self.LINK_LENGTH_1 * sin(s[0]) * scale,
        ]

        p2 = [
            p1[0] - self.LINK_LENGTH_2 * cos(s[0] + s[1]) * scale,
            p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1]) * scale,
        ]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0] - pi / 2, s[0] + s[1] - pi / 2]
        link_lengths = [self.LINK_LENGTH_1 * scale, self.LINK_LENGTH_2 * scale]

        for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
            x = x + offset
            y = y + offset
            l, r, t, b = 0, llen, 0.1 * scale, -0.1 * scale
            coords = [(l, b), (l, t), (r, t), (r, b)]
            transformed_coords = []
            for coord in coords:
                coord = pygame.math.Vector2(coord).rotate_rad(th)
                coord = (coord[0] + x, coord[1] + y)
                transformed_coords.append(coord)
            gfxdraw.aapolygon(self.surf, transformed_coords, (0, 204, 204))
            gfxdraw.filled_polygon(self.surf, transformed_coords, (0, 204, 204))

            gfxdraw.aacircle(self.surf, int(x), int(y), int(0.1 * scale), (204, 204, 0))
            gfxdraw.filled_circle(
                self.surf, int(x), int(y), int(0.1 * scale), (204, 204, 0)
            )

        # drawing target position and initial position
        target = (scale*self.target[1] + offset, scale * self.target[0] + offset)
        pygame.gfxdraw.filled_circle(self.surf, int(target[0]), int(target[1]), 5, (0, 0, 255)) # blue

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

def wrap(x, m, M):
    """Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.

    Args:
        x: a scalar
        m: minimum possible value in range
        M: maximum possible value in range

    Returns:
        x: a scalar, wrapped
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x


def bound(x, m, M=None):
    """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].

    Args:
        x: scalar
        m: The lower bound
        M: The upper bound

    Returns:
        x: scalar, bound between min (m) and Max (M)
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)


def rk4(derivs, y0, t):
    """
    Integrate 1-D or N-D system of ODEs using 4-th order Runge-Kutta.

    Example for 2D system:

        >>> def derivs(x):
        ...     d1 =  x[0] + 2*x[1]
        ...     d2 =  -3*x[0] + 4*x[1]
        ...     return d1, d2

        >>> dt = 0.0005
        >>> t = np.arange(0.0, 2.0, dt)
        >>> y0 = (1,2)
        >>> yout = rk4(derivs, y0, t)

    Args:
        derivs: the derivative of the system and has the signature ``dy = derivs(yi)``
        y0: initial state vector
        t: sample times

    Returns:
        yout: Runge-Kutta approximation of the ODE
    """

    try:
        Ny = len(y0) - 1
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0[:Ny]

    for i in np.arange(len(t) - 1):

        this = t[i]
        dt = t[i + 1] - this
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0))
        k2 = np.asarray(derivs(y0 + dt2 * k1))
        k3 = np.asarray(derivs(y0 + dt2 * k2))
        k4 = np.asarray(derivs(y0 + dt * k3))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    # We only care about the final timestep and we cleave off action value which will be zero
    return yout[-1][:4]