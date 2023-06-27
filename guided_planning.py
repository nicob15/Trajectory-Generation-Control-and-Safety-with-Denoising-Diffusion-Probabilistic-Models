import numpy as np
import os
import torch
import argparse
from utils.logger import Logger
from envs.reacher import ConstrainedReacherEnv
from utils.guided_policy import ValueGuide, CbfGuide
from diffuser.models.temporal import TemporalUnet, ValueFunction
from utils.models import GaussianDiffusion as ProbDiffusion
from utils.models import CbfDiffusion
from utils.guided_policy import n_step_doubleguided_p_sample
from utils.guided_policy import n_step_guided_p_sample
import diffuser.utils as utils
import einops
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
import copy
import time

parser = argparse.ArgumentParser()

parser.add_argument('--num-targets', type=int, default=1,
                    help='Number of targets.')
parser.add_argument('--num-steps', type=int, default=100,
                    help='Number of steps.')
parser.add_argument('--batch-size', type=int, default=64,
                    help='Batch size (number of generated planning).')
parser.add_argument('--state-dim', type=int, default=8,
                    help='Number of episodes.')
parser.add_argument('--action-dim', type=int, default=2,
                    help='Action dimension.')
parser.add_argument('--max_torque', type=float, default=1,
                    help='Maximum torque for the actuators.')
parser.add_argument('--epsilon', type=float, default=0.3,
                    help='Tolerance for reaching the target.')
parser.add_argument('--target-location', type=float, default=(1.5, 1.5),
                    help='Location of the obstacle.')
parser.add_argument('--obstacle-location', type=float, default=(1.5, 1.5),
                    help='Location of the obstacle.')
parser.add_argument('--obstacle-radius', type=float, default=1.0,
                    help='Radius of the obstacle (circular obstacle).')
parser.add_argument('--lambda-cbf', type=float, default=0.99,
                    help='Lambda coefficient control barrier function.')
parser.add_argument('--sequence-length', type=int, default=16,
                    help='Sequence length.')
parser.add_argument('--diffusion-steps', type=int, default=50,
                    help='Number of diffusion steps.')
parser.add_argument('--use-attention', default=True,
                    help='Use attention layer in temporal U-net.')
parser.add_argument('--valuye-only', default=False,
                    help='Use only value function model for guided planning '
                         'or the combination of value and cbf classifier.')
parser.add_argument('--observation-dim-w', type=int, default=84,
                    help='Width of the input measurements (RGB images).')
parser.add_argument('--observation-dim-h', type=int, default=84,
                    help='Height of the input measurements (RGB images).')
parser.add_argument('--save-traj', default=False,
                    help='Generate training or testing dataset.')
parser.add_argument('--trajectory-dataset', type=str, default='reacher_trajectories.pkl',
                    help='Training dataset.')
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed (default: 1).')
parser.add_argument('--experiment', type=str, default='Reacher',
                    help='Experiment.')
parser.add_argument('--model-type', type=str, default='ProbDiffusion',
                    help='Model type.')
parser.add_argument('--num-classes', type=int, default=2,
                    help='Number of classes (safe and unsafe).')


args = parser.parse_args()


def save_frames_as_mp4(frames, filename='reacher_animation.mp4', folder='results/', idx=0):

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

def save_frames_as_png(frames, filename='snapshots', folder='results/'):
    save_dir = folder + 'videos/snapshots'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in range(len(frames)):
        plt.imshow(frames[i])
        plt.savefig(save_dir + '/' + filename + '_' + str(i) + '.png')

save_traj = args.save_traj
if save_traj:
    data_file_name = args.trajectory_dataset
max_steps = args.num_steps
seed = args.seed

value_only = False
env = ConstrainedReacherEnv(min_torque=-args.max_torque, max_torque=args.max_torque, target=args.target_location,
                            max_steps=max_steps, epsilon=args.epsilon, reset_target_reached=True, bonus_reward=False,
                            barrier_type='circle', location=args.obstacle_location, radius=args.obstacle_radius,
                            lambda_cbf=args.lambda_cbf, initial_state=np.array([-2.8, 0.0, -0.1, -0.1]))
state_dim = args.state_dim
action_dim = args.action_dim

directory = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(directory + '/data/')
logger = Logger(folder)

# Set seeds
env.action_space.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

horizon = args.sequence_length  # planning horizon
observation_dim = state_dim
transition_dim = observation_dim + action_dim
nr_diffusion_steps = args.diffusion_steps
use_attention = args.use_attention

exp = args.experiment
mtype = args.model_type
save_pth_dir = directory + '/results/' + str(exp) + '/' + str(mtype)
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
checkpoint = torch.load(save_pth_dir + '/ProbDiff_Model_best.pth')
diffusion.model.load_state_dict(checkpoint['model'])
ema_diffusion.model.load_state_dict(checkpoint['ema_model'])

value_model = ValueFunction(horizon, transition_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=use_attention, final_sigmoid=False)
value_diffusion = ValueGuide(value_model)
ema_value_diffusion = copy.deepcopy(value_diffusion).cuda()
checkpoint = torch.load(save_pth_dir + '/ValueDiff_Model_best.pth')
value_diffusion.model.load_state_dict(checkpoint['model'])
ema_value_diffusion.model.load_state_dict(checkpoint['ema_model'])

cbf_model = ValueFunction(horizon, transition_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=use_attention, out_dim=args.num_classes*horizon, final_sigmoid=True)
cbf_diffusion = CbfGuide(cbf_model)
ema_cbf_diffusion = copy.deepcopy(cbf_diffusion).cuda()
checkpoint = torch.load(save_pth_dir + '/CbfDiff_Model_best.pth')
cbf_diffusion.model.load_state_dict(checkpoint['model'])
ema_cbf_diffusion.model.load_state_dict(checkpoint['ema_model'])

batch_size = args.batch_size
x_start = np.zeros((batch_size, horizon, transition_dim))
nr_targets = args.num_targets

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
            samples = ema_diffusion(cond, guide=[ema_value_diffusion, ema_cbf_diffusion],
                                    sample_fn=n_step_doubleguided_p_sample)
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