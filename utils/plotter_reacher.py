import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import torch
from torchvision.utils import save_image
import os
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import Normalizer
from matplotlib.patches import Ellipse
from matplotlib import cm
from matplotlib import animation
import gc
import einops

import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'


def compute_PCA(input, dim=2):
    pca = PCA(n_components=dim)
    return pca.fit_transform(input)

def saveMultipage(filename, figs=None, dpi=100):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

def normalize(x):
    transformer = Normalizer().fit(x)
    return transformer.transform(x)

def closeAll():
    plt.close('all')

def plot_mean_trajectories(s1, s2, save_dir, name='true_angles', legend='ground truth'):

    a1 = np.arctan2(s1[:, :, :, 1], s1[:, :, :, 0]).reshape(s1.shape[0], -1)
    a2 = np.arctan2(s2[:, :, :, 1], s2[:, :, :, 0]).reshape(s2.shape[0], -1)
    a1_dot = s1[:, :, :, 2].reshape(s1.shape[0], -1)
    a2_dot = s2[:, :, :, 2].reshape(s1.shape[0], -1)
    timesteps = np.arange(0, a1.shape[1], 1)

    a1_mean = np.mean(a1, axis=0)
    a2_mean = np.mean(a2, axis=0)
    a1_dot_mean = np.mean(a1_dot, axis=0)
    a2_dot_mean = np.mean(a2_dot, axis=0)
    a1_std = np.std(a1, axis=0)
    a2_std = np.std(a2, axis=0)
    a1_dot_std = np.std(a1_dot, axis=0)
    a2_dot_std = np.std(a2_dot, axis=0)

    fig, ax = plt.subplots(4)
    fig.suptitle(name)

    legend1 = 'theta_1'

    p1 = ax[0].scatter(timesteps, a1_mean, c=timesteps, cmap='cool', zorder=2)
    ax[0].plot(timesteps, a1_mean, color='gray', zorder=1)
    ax[0].fill_between(timesteps, a1_mean+a1_std, a1_mean-a1_std, facecolor='gray', alpha=0.5, zorder=0)
    ax[0].legend([p1], [legend1])
    cbar = fig.colorbar(p1)
    cbar.set_label('timesteps', rotation=90)

    legend2 = 'theta_2'

    p2 = ax[1].scatter(timesteps, a2_mean, c=timesteps, cmap='cool', zorder=2)
    ax[1].plot(timesteps, a2_mean, color='gray', zorder=1)
    ax[1].fill_between(timesteps, a2_mean+a2_std, a2_mean-a2_std, facecolor='gray', alpha=0.5, zorder=0)
    ax[1].legend([p2], [legend2])
    cbar = fig.colorbar(p2)
    cbar.set_label('timesteps', rotation=90)

    legend3 = 'theta_1_dot'

    p3 = ax[2].scatter(timesteps, a1_dot_mean, c=timesteps, cmap='cool', zorder=2)
    ax[2].plot(timesteps, a1_dot_mean, color='gray', zorder=1)
    ax[2].fill_between(timesteps, a1_dot_mean+a1_dot_std, a1_dot_mean-a1_dot_std, facecolor='gray', alpha=0.5, zorder=0)
    ax[2].legend([p3], [legend3])
    cbar = fig.colorbar(p3)
    cbar.set_label('timesteps', rotation=90)

    legend4 = 'theta_2_dot'

    p4 = ax[3].scatter(timesteps, a2_dot_mean, c=timesteps, cmap='cool', zorder=2)
    ax[3].plot(timesteps, a2_dot_mean, color='gray', zorder=1)
    ax[3].fill_between(timesteps, a2_dot_mean+a2_dot_std, a2_dot_mean-a2_dot_std, facecolor='gray', alpha=0.5, zorder=0)
    ax[3].legend([p4], [legend4])
    cbar = fig.colorbar(p4)
    cbar.set_label('timesteps', rotation=90)

    plt.savefig(save_dir + '/' + name + '_timesteps.png')
def save_trajectory_as_mp4(s1, s2, s1_gt, s2_gt, save_dir, name='traj_.mp4'):
    a1 = np.arctan2(s1[:, :, 1], s1[:, :,  0]).reshape((-1))
    a2 = np.arctan2(s2[:, :, 1], s2[:, :, 0]).reshape((-1))
    a1_dot = s1[:, :, 2].reshape(-1)
    a2_dot = s2[:, :, 2].reshape(-1)

    a1_gt = np.arctan2(s1_gt[:, :, 1], s1_gt[:, :,  0]).reshape((-1))
    a2_gt = np.arctan2(s2_gt[:, :, 1], s2_gt[:, :, 0]).reshape((-1))
    a1_dot_gt = s1_gt[:, :, 2].reshape(-1)
    a2_dot_gt = s2_gt[:, :, 2].reshape(-1)

    timesteps = np.arange(0, s1.shape[1], 1)

    fig, ax = plt.subplots(4)
    fig.suptitle(name)

    # animation function. This is called sequentially
    def animate(i):
        ax[0].plot(timesteps[:i + 1], a1_gt[:i + 1], color='orange', zorder=0)
        ax[0].scatter(timesteps[:i+1], a1_gt[:i+1], c=timesteps[:i+1], cmap='winter', zorder=1)

        ax[0].plot(timesteps[:i + 1], a1[:i + 1], color='silver', zorder=2)
        p1 = ax[0].scatter(timesteps[:i+1], a1[:i+1], c=timesteps[:i+1], cmap='cool', zorder=3)

        ax[1].plot(timesteps[:i + 1], a2_gt[:i + 1], color='orange', zorder=0)
        ax[1].scatter(timesteps[:i+1], a2_gt[:i+1], c=timesteps[:i+1], cmap='winter', zorder=1)

        ax[1].plot(timesteps[:i+1], a2[:i+1], color='silver', zorder=2)
        p2 = ax[1].scatter(timesteps[:i+1], a2[:i+1], c=timesteps[:i+1], cmap='cool', zorder=3)

        ax[2].plot(timesteps[:i + 1], a1_dot_gt[:i + 1], color='orange', zorder=0)
        ax[2].scatter(timesteps[:i+1], a1_dot_gt[:i+1], c=timesteps[:i+1], cmap='winter', zorder=1)

        ax[2].plot(timesteps[:i + 1], a1_dot[:i + 1], color='silver', zorder=2)
        p3 = ax[2].scatter(timesteps[:i+1], a1_dot[:i+1], c=timesteps[:i+1], cmap='cool', zorder=3)

        ax[3].plot(timesteps[:i + 1], a2_dot_gt[:i + 1], color='orange', zorder=0)
        ax[3].scatter(timesteps[:i+1], a2_dot_gt[:i+1], c=timesteps[:i+1], cmap='winter', zorder=1)

        ax[3].plot(timesteps[:i+1], a2_dot[:i+1], color='silver', zorder=2)
        p4 = ax[3].scatter(timesteps[:i+1], a2_dot[:i+1], c=timesteps[:i+1], cmap='cool', zorder=3)

        return p1, p2, p3, p4,

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=s1.shape[1], interval=50, save_count=s1.shape[1],
                                   cache_frame_data=True)

    FFwriter = animation.FFMpegWriter(fps=2)
    anim.save(save_dir + '/' + name, writer=FFwriter)

    plt.close()

def plot_trajectories(s1, s2, save_dir, name='true_angles', legend='ground truth'):

    a1 = np.arctan2(s1[:, :, 1], s1[:, :,  0]).reshape(-1)
    a2 = np.arctan2(s2[:, :, 1], s2[:, :, 0]).reshape(-1)
    a1_dot = s1[:, :, 2].reshape(-1)
    a2_dot = s2[:, :, 2].reshape(-1)
    timesteps = np.arange(0, s1.shape[1], 1)

    fig, ax = plt.subplots(4)
    fig.suptitle(name)

    legend1 = 'theta_1'

    p1 = ax[0].scatter(timesteps, a1, c=timesteps, cmap='cool', zorder=1)
    ax[0].plot(timesteps, a1, zorder=0)
    ax[0].legend([p1], [legend1])
    cbar = fig.colorbar(p1)
    cbar.set_label('timesteps', rotation=90)

    legend2 = 'theta_2'

    p2 = ax[1].scatter(timesteps, a2, c=timesteps, cmap='cool', zorder=1)
    ax[1].plot(timesteps, a2, zorder=0)
    ax[1].legend([p2], [legend2])
    cbar = fig.colorbar(p2)
    cbar.set_label('timesteps', rotation=90)

    legend3 = 'theta_1_dot'

    p3 = ax[2].scatter(timesteps, a1_dot, c=timesteps, cmap='cool', zorder=1)
    ax[2].plot(timesteps, a1_dot, zorder=0)
    ax[2].legend([p3], [legend3])
    cbar = fig.colorbar(p3)
    cbar.set_label('timesteps', rotation=90)

    legend4 = 'theta_2_dot'

    p4 = ax[3].scatter(timesteps, a2_dot, c=timesteps, cmap='cool', zorder=1)
    ax[3].plot(timesteps, a2_dot, zorder=0)
    ax[3].legend([p4], [legend4])
    cbar = fig.colorbar(p4)
    cbar.set_label('timesteps', rotation=90)


    plt.savefig(save_dir + '/' + name + '_timesteps.png')

def save_diffusion_as_mp4(diffusion, save_dir, name='traj_.mp4'):
    # First set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots(4)
    fig.suptitle(name)

    # animation function. This is called sequentially
    def animate(i):
        s1 = diffusion[i, :, :, 2:5]
        s2 = diffusion[i, :, :, 5:8]
        a1 = np.arctan2(s1[:, :, 1], s1[:, :, 0]).reshape((-1))
        a2 = np.arctan2(s2[:, :, 1], s2[:, :, 0]).reshape((-1))
        timesteps = np.arange(0, a1.shape[0], 1)

        legend1 = 'theta_1'
        colors = cm.cool(np.linspace(0, 1, diffusion.shape[0]))
        p1 = ax[0].scatter(timesteps, a1, color=colors[i])
        ax[0].plot(timesteps, a1, color=colors[i])

        legend2 = 'theta_2'
        p2 = ax[1].scatter(timesteps, a2, c=colors[i])
        ax[1].plot(timesteps, a2, color=colors[i])

        legend3 = 'theta_1_dot'
        a1_dot = s1[:, :, 2].reshape(-1)
        p3 = ax[2].scatter(timesteps, a1_dot, c=colors[i])
        ax[2].plot(timesteps, a1_dot, color=colors[i])

        legend4 = 'theta_2_dot'
        a2_dot = s2[:, :, 2].reshape(-1)
        p4 = ax[3].scatter(timesteps, a2_dot, c=colors[i])
        ax[3].plot(timesteps, a2_dot, color=colors[i])
        return p1, p2, p3, p4,

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=diffusion.shape[0], interval=50, save_count=diffusion.shape[0],
                                   cache_frame_data=True)

    FFwriter = animation.FFMpegWriter(fps=2)
    anim.save(save_dir + '/' + name, writer=FFwriter)

    plt.close()

def plot_diffusion(diffusion, save_dir, name='diffusion steps'):
    fig, ax = plt.subplots(4)
    fig.suptitle(name)

    for i in range(diffusion.shape[0]):
        s1 = diffusion[i, :, :, 0:3]
        s2 = diffusion[i, :, :, 3:6]
        a1 = np.arctan2(s1[:, :, 1], s1[:, :, 0]).reshape((-1))
        a2 = np.arctan2(s2[:, :, 1], s2[:, :, 0]).reshape((-1))
        timesteps = np.arange(0, a1.shape[0], 1)

        legend1 = 'theta_1'
        colors = cm.cool(np.linspace(0, 1, diffusion.shape[0]))
        p1 = ax[0].scatter(timesteps, a1, color=colors[i])
        ax[0].plot(timesteps, a1, color=colors[i])

        legend2 = 'theta_2'

        p2 = ax[1].scatter(timesteps, a2, c=colors[i])
        ax[1].plot(timesteps, a2, color=colors[i])

        legend3 = 'theta_1_dot'
        a1_dot = s1[:, :, 2].reshape(-1)
        p3 = ax[2].scatter(timesteps, a1_dot, c=colors[i])
        ax[2].plot(timesteps, a1_dot, color=colors[i])

        legend4 = 'theta_2_dot'
        a2_dot = s2[:, :, 2].reshape(-1)
        p4 = ax[3].scatter(timesteps, a2_dot, c=colors[i])
        ax[3].plot(timesteps, a2_dot, color=colors[i])

    ax[0].legend([p1], [legend1])
    ax[1].legend([p2], [legend2])
    ax[2].legend([p3], [legend3])
    ax[3].legend([p4], [legend4])
    plt.savefig(save_dir + '/' + name + '_diffusion_steps.png')
    closeAll()

def run_diffusion(model, cond, **diffusion_kwargs):

    ## format `conditions` input for model
    conditions = cond

    samples, _, diffusion = model.conditional_sample(conditions, return_chain=True, verbose=False, **diffusion_kwargs)

    diffusion = diffusion.cpu().numpy()
    samples = samples.cpu().numpy()

    observations = diffusion[:, :, :, :]

    observations = einops.rearrange(observations, 'batch steps horizon dim -> steps batch horizon dim')

    return observations, samples


def plot_results(diffusion, ema_diffusion, test_loader, save_dir, num_timesteps=50, true_dyn=False, epoch=0,
                 plot_mean_traj=True, num_realizations=100):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)



    if true_dyn:
        save_dir_1 = save_dir + '/ema_diffusion/'
        if not os.path.exists(save_dir_1):
            os.makedirs(save_dir_1)

        data = test_loader.sample_sequence_batch(batch_size=1, sequence_len=num_timesteps)  #### sample batch of sequences

        #### num_timesteps, batch_size, state_dim ###
        states = torch.from_numpy(data['states']).cuda()
        #### batch_size, num_timesteps, state_dim ###
        states = states.permute(1, 0, 2)

        s1 = states[:, :, 0:3]
        s2 = states[:, :, 3:6]

        plot_trajectories(s1.cpu().numpy(), s2.cpu().numpy(), save_dir_1, name='true_angles_' + str(epoch))

        conditions = {0: states[0, 0, :].reshape(1, 8),
                      num_timesteps - 1: states[0, -1, :].reshape(1, 8)
                     }

        diff, traj = run_diffusion(ema_diffusion, conditions)

        plot_diffusion(diff, save_dir_1, name='diffusion_steps_ ' + str(epoch))
        save_diffusion_as_mp4(diff, save_dir_1, name='diffusion_step_' + str(epoch) + '.mp4')

        s1_pred = traj[:, :, ema_diffusion.action_dim: ema_diffusion.action_dim+s1.shape[2]].reshape(1, num_timesteps, 3)
        s2_pred = traj[:, :, ema_diffusion.action_dim+s1.shape[2]:-2].reshape(1, num_timesteps, 3)

        plot_trajectories(s1_pred, s2_pred, save_dir_1, name='pred_angles_' + str(epoch))
        save_trajectory_as_mp4(s1_pred, s2_pred, s1.cpu().numpy(), s2.cpu().numpy(), save_dir_1, name='pred_traj_vs_true_traj_' + str(epoch) + '.mp4')

        if plot_mean_traj:
            s1_list = []
            s2_list = []
            for k in range(num_realizations):
                diff, traj = run_diffusion(ema_diffusion, conditions)
                s1_pred = traj[:, :, 2:5].reshape(1, num_timesteps, 3)
                s2_pred = traj[:, :, 5:8].reshape(1, num_timesteps, 3)
                s1_list.append(s1_pred)
                s2_list.append(s2_pred)
            plot_mean_trajectories(np.array(s1_list), np.array(s2_list), save_dir_1, name='pred_mean_traj_'+ str(epoch))

        save_dir_2 = save_dir + '/diffusion/'
        if not os.path.exists(save_dir_2):
            os.makedirs(save_dir_2)

        plot_trajectories(s1.cpu().numpy(), s2.cpu().numpy(), save_dir_2, name='true_angles_' + str(epoch))

        conditions = {0: states[0, 0, :].reshape(1, 8),
                    num_timesteps - 1: states[0, -1, :].reshape(1, 8)}

        diff, traj = run_diffusion(diffusion, conditions)

        plot_diffusion(diff, save_dir_2, name='diffusion_steps_ ' + str(epoch))
        save_diffusion_as_mp4(diff, save_dir_2, name='diffusion_step_' + str(epoch) + '.mp4')

        s1_pred = traj[:, :, diffusion.action_dim: diffusion.action_dim + s1.shape[2]].reshape(1, num_timesteps, 3)
        s2_pred = traj[:, :, diffusion.action_dim + s1.shape[2]:-2].reshape(1, num_timesteps, 3)

        plot_trajectories(s1_pred, s2_pred, save_dir_2, name='pred_angles_' + str(epoch))
        save_trajectory_as_mp4(s1_pred, s2_pred, s1.cpu().numpy(), s2.cpu().numpy(), save_dir_2, name='pred_traj_vs_true_traj_' + str(epoch) + '.mp4')

        if plot_mean_traj:
            s1_list = []
            s2_list = []
            for k in range(num_realizations):
                diff, traj = run_diffusion(diffusion, conditions)
                s1_pred = traj[:, :, 2:5].reshape(1, num_timesteps, 3)
                s2_pred = traj[:, :, 5:8].reshape(1, num_timesteps, 3)
                s1_list.append(s1_pred)
                s2_list.append(s2_pred)
            plot_mean_trajectories(np.array(s1_list), np.array(s2_list), save_dir_2, name='pred_mean_traj_' + str(epoch))

    else:
        raise Exception("Not implemented")
