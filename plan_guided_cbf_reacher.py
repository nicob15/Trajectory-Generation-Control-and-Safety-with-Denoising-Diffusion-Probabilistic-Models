import numpy as np
import os
import torch
import argparse

from logger import Logger
from utils import stack_frames

from Envs.reacher import ConstrainedReacherEnv
from guided_policy import ValueGuide, CbfGuide
from diffuser.models.temporal import TemporalUnet, ValueFunction
from models import GaussianDiffusion as ProbDiffusion
from models import CbfDiffusion
from guided_policy import n_step_doubleguided_p_sample
from guided_policy import n_step_guided_p_sample
import diffuser.utils as utils
import einops
import copy
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'


import copy
import time


parser = argparse.ArgumentParser()

parser.add_argument('--env-name', type=str, default='Reacher-v1',
                    help='Environment name.')
parser.add_argument('--num-episodes', type=int, default=25,
                    help='Number of episodes.')
parser.add_argument('--state-dim', type=int, default=8,
                    help='Number of episodes.')
parser.add_argument('--observation-dim-w', type=int, default=84,
                    help='Width of the input measurements (RGB images).')
parser.add_argument('--observation-dim-h', type=int, default=84,
                    help='Height of the input measurements (RGB images).')
parser.add_argument('--save-traj', default=False,
                    help='Generate training or testing dataset.')
parser.add_argument('--trajectory-dataset', type=str, default='cbf_reacher_trajectories.pkl',
                    help='Training dataset.')
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed (default: 1).')

args = parser.parse_args()


def save_frames_as_mp4(frames, filename='doublependulum_animation.mp4', folder='Results/', idx=0):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)

    save_dir = folder + 'videos/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    FFwriter = animation.FFMpegWriter(fps=10)
    anim.save(save_dir + str(idx) + '_' + filename, writer=FFwriter)

def save_frames_as_png(frames, filename='snapshots', folder='Results/'):
    save_dir = folder + 'videos/snapshots/'
    for i in range(len(frames)):
        plt.imshow(frames[i])
        plt.savefig(save_dir + '/' + filename + '_' + str(i) + '.png')
        #plt.close()

env_name = args.env_name
save_traj = args.save_traj
if save_traj:
    data_file_name = args.trajectory_dataset
obs_dim1 = args.observation_dim_w
obs_dim2 = args.observation_dim_h
num_episodes = args.num_episodes
seed = args.seed

max_steps = 200
value_only = False
env = ConstrainedReacherEnv(max_torque=1.0, target=(0.0, 1.4), max_steps=max_steps, epsilon=0.3, reset_target_reached=True,
                            bonus_reward=False, barrier_type='circle', location=(1.5, 1.5), radius=1.0, gamma=0.99, initial_state=np.array([-2.8, 0.0, -0.1, -0.1]))
state_dim = env.observation_space.shape[0]
action_dim = 2

directory = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(directory + '/Data/')
logger = Logger(folder)

# Set seeds
#env.seed(seed)
env.action_space.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

#default for pendulum-v1
horizon = 16  # planning horizon
observation_dim = state_dim
transition_dim = observation_dim + action_dim
nr_diffusion_steps = 50
use_attention = True

exp = 'Reacher'
mtype = 'ProbDiffusion'
noise_level = 0.0
save_pth_dir = directory + '/Results/' + str(exp) + '/' + str(mtype) + '/Noise_level_' + str(noise_level)
if not os.path.exists(save_pth_dir):
    os.makedirs(save_pth_dir)

model = TemporalUnet(horizon, transition_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=use_attention)


if value_only:
    diffusion = ProbDiffusion(model, horizon, observation_dim, action_dim, n_timesteps=nr_diffusion_steps,
                          loss_type='l2', clip_denoised=False, predict_epsilon=False,
                          action_weight=1.0, loss_discount=1.0, loss_weights=None)
else:
    diffusion = CbfDiffusion(model, horizon, observation_dim, action_dim, n_timesteps=nr_diffusion_steps,
                          loss_type='l2', clip_denoised=False, predict_epsilon=False,
                          action_weight=1.0, loss_discount=1.0, loss_weights=None)
ema_diffusion = copy.deepcopy(diffusion).cuda()

#checkpoint = torch.load(save_pth_dir + '/random_policy/200_epochs/ProbDiff_Model_latest.pth')
checkpoint = torch.load(save_pth_dir + '/ProbDiff_Model_latest.pth')
#checkpoint = torch.load(save_pth_dir + '/optimal_policy/ProbDiff_Model_latest.pth')
#checkpoint = torch.load(save_pth_dir + '/random_policy/ProbDiff_Model_16-05-2023_16h-27m-55s.pth')
diffusion.model.load_state_dict(checkpoint['model'])
ema_diffusion.model.load_state_dict(checkpoint['ema_model'])

value_model = ValueFunction(horizon, transition_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=use_attention, final_sigmoid=False)
value_diffusion = ValueGuide(value_model)
ema_value_diffusion = copy.deepcopy(value_diffusion).cuda()
#checkpoint = torch.load(save_pth_dir + '/random_policy/200_epochs/ValueDiff_Model_latest.pth')
checkpoint = torch.load(save_pth_dir + '/ValueDiff_Model_latest.pth')
#checkpoint = torch.load(save_pth_dir + '/optimal_policy/ValueDiff_Model_latest.pth')
#checkpoint = torch.load(save_pth_dir + '/random_policy/ValueDiff_Model_16-05-2023_16h-27m-55s.pth')
value_diffusion.model.load_state_dict(checkpoint['model'])
ema_value_diffusion.model.load_state_dict(checkpoint['ema_model'])

cbf_model = ValueFunction(horizon, transition_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=use_attention, out_dim=2*horizon, final_sigmoid=True)
cbf_diffusion = CbfGuide(cbf_model)
ema_cbf_diffusion = copy.deepcopy(cbf_diffusion).cuda()
#checkpoint = torch.load(save_pth_dir + '/random_policy/200_epochs/CbfDiff_Model_latest.pth')
#checkpoint = torch.load(save_pth_dir + '/CbfDiff_Model_latest.pth')
#checkpoint = torch.load(save_pth_dir + '/optimal_policy/CbfDiff_Model_latest.pth')

#checkpoint = torch.load(save_pth_dir + '/CbfDiff_Model_22-05-2023_15h-34m-07s.pth')
checkpoint = torch.load(save_pth_dir + '/CbfDiff_Model_latest.pth')

cbf_diffusion.model.load_state_dict(checkpoint['model'])
ema_cbf_diffusion.model.load_state_dict(checkpoint['ema_model'])

batch_size = 64
x_start = np.zeros((batch_size, horizon, transition_dim))
nr_targets = 1

cond = {}
frames = []
save_gif = True
successes = 0
failures = 0
safety_violations = 0
total_rews = []
nr_steps = []

diffusion.eval()
value_diffusion.eval()
cbf_diffusion.eval()
ema_diffusion.eval()
ema_value_diffusion.eval()
ema_cbf_diffusion.eval()

start = time.time()
for episode in range(nr_targets):
    state = env.reset()
    frames.append(env.render(mode='rgb_array'))
    print('Episode: ', episode)
    total_rew = 0
    for step in range(max_steps):
        print('Step: ', step)
        cond = {0: torch.from_numpy(state).cuda()}
        cond = utils.to_torch(cond, dtype=torch.float32, device='cuda:0')
        cond = utils.apply_dict(
               einops.repeat,
               cond,
               'd -> repeat d', repeat=batch_size,
               )
        if value_only:
            samples = ema_diffusion(cond, guide=ema_value_diffusion,
                                sample_fn=n_step_guided_p_sample)
        else:
            samples = ema_diffusion(cond, guide=[ema_value_diffusion, ema_cbf_diffusion], sample_fn=n_step_doubleguided_p_sample)
        action = samples.trajectories[0, 0, 0:action_dim].cpu().numpy()
        value = samples.values.cpu().numpy()
        print("value", value[0])
        print("action", action)
        next_state, reward, done, cbf_value, target_reached = env.step(action)
        if cbf_value > 0.0:
            safety_violations += 1
        print("reward", reward)
        total_rew += reward
        if save_gif:
            frames.append(env.render(mode='rgb_array'))
        if save_traj:
            logger.obslog((state, action, reward, next_state, done))
        env.render()
        state = next_state

        if done:
            if target_reached:
                successes += 1
            else:
                failures += 1
            print("total reward", total_rew)
            print("Successful episodes:", successes)
            print("Unsuccessful episodes:", failures)
            total_rews.append(total_rew)
            nr_steps.append(step)
            save_frames_as_mp4(frames, folder=save_pth_dir + '/', idx=episode)
            save_frames_as_png(frames, folder=save_pth_dir + '/')
            frames = []
            break

end = time.time()
print("Total time:", end - start)

print("Successful episodes:", successes)
print("Unsuccessful episodes:", failures)
print("Total number of constrain violations:", safety_violations)
total_rews = np.array(total_rews)
print("Average reward:", total_rews.mean())
print("Std reward:", total_rews.std())
nr_steps = np.array(nr_steps)
print("Average nr steps:", nr_steps.mean())
print("Std nr steps:", nr_steps.std())

if save_traj:
    logger.save_obslog(filename=data_file_name)