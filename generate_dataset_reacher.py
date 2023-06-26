import numpy as np
import os
import torch
import argparse
from logger import Logger
from envs.reacher import ConstrainedReacherEnv

parser = argparse.ArgumentParser()

parser.add_argument('--num-episodes-trainset', type=int, default=300,
                    help='Number of episodes for creating the trainset.')
parser.add_argument('--num-episodes-testset', type=int, default=30,
                    help='Number of episodes for creating the testset.')
parser.add_argument('--num-steps', type=int, default=100,
                    help='Number of steps in each episode.')
parser.add_argument('--state-dim', type=int, default=8,
                    help='State dimension.')
parser.add_argument('--action-dim', type=int, default=2,
                    help='Action dimension.')
parser.add_argument('--max_torque', type=float, default=1,
                    help='Maximum torque for the actuators.')
parser.add_argument('--epsilon', type=float, default=0.3,
                    help='Tolerance for reaching the target.')
parser.add_argument('--obstacle-location', type=float, default=(1.5, 1.5),
                    help='Location of the obstacle.')
parser.add_argument('--obstacle-radius', type=float, default=1.0,
                    help='Radius of the obstacle (circular obstacle).')
parser.add_argument('--lambda-cbf', type=float, default=0.99,
                    help='Lambda coefficient control barrier function.')
parser.add_argument('--test', default=False,
                    help='Generate training or testing dataset.')
parser.add_argument('--training-dataset', type=str, default='reacher_train.pkl',
                    help='Training dataset.')
parser.add_argument('--testing-dataset', type=str, default='reacher_test.pkl',
                    help='Testing dataset.')
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed (default: 1).')
parser.add_argument('--render', default=False,
                    help='Render environment.')

args = parser.parse_args()

test = args.test
if test:
    data_file_name = args.testing_dataset
    num_episodes = args.num_episodes_testset
else:
    data_file_name = args.training_dataset
    num_episodes = args.num_episodes_trainset
seed = args.seed
max_steps = args.num_steps
env = ConstrainedReacherEnv(min_torque=-args.max_torque, max_torque=args.max_torque, target=None, max_steps=max_steps,
                            epsilon=args.epsilon, reset_target_reached=False, bonus_reward=False, barrier_type='circle',
                            location=args.obstacle_location, radius=args.obstacle_radius, lambda_cbf=args.lambda_cbf)
state_dim = args.state_dim
action_dim = args.action_dim

# Make directory for saving the datasets
directory = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(directory + '/data/')
logger = Logger(folder)
if not os.path.exists(folder):
    os.makedirs(folder)

# Set seeds
env.action_space.seed(seed)
np.random.seed(seed)

for episode in range(num_episodes):
    state = env.reset()
    print('Episode: ', episode)
    for step in range(max_steps):
        print("Step", step)
        action = env.action_space.sample()
        next_state, reward, done, cbf_value, _ = env.step(action)
        if args.render:
            env.render()
        logger.obslog((state, action, reward, next_state, done, cbf_value))
        state = next_state
        if done:
            break

logger.save_obslog(filename=data_file_name)