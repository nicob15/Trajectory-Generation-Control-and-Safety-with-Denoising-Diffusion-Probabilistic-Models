import copy

import torch
import torch.utils
import os
from utils.utils import load_pickle
from utils.replay_buffer import ReplayBufferBarrier as ReplayBuffer
import numpy as np
from utils.plotter_reacher import plot_results
from utils.models import GaussianDiffusion as ProbDiffusion
from diffuser.models.temporal import TemporalUnet, ValueFunction
from utils.models import ValueDiffusion
from utils.trainer import train_ProbDiffusion as train
from utils.trainer import train_ValueDiffusion as train_valuefunction
from utils.trainer import train_CbfDiffusion as train_cbfdiffusion
from utils.trainer import test_ValueDiffusion as test_valuefunction
from utils.trainer import test_ProbDiffusion as test_probdiffusion
from utils.trainer import test_CbfDiffusion as test_cbfdiffusion
import gc
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size.')
parser.add_argument('--num-epochs', type=int, default=100,
                    help='Number of training epochs.')
parser.add_argument('--learning-rate', type=float, default=2e-4,
                    help='Learning rate.')
parser.add_argument('--reg-coefficient', type=float, default=0.0,
                    help='L2 regularization coefficient.')

parser.add_argument('--training', default=True,
                    help='Train the models.')
parser.add_argument('--plotting', default=True,
                    help='Plot the results.')
parser.add_argument('--num-samples-plot', type=int, default=200,
                    help='Number of independent sampling from the distribution.')

parser.add_argument('--hidden-dim', type=int, default=256,
                    help='Number of hidden units in MLPs.')
parser.add_argument('--action-dim', type=int, default=2,
                    help='Dimensionality of the action space.')
parser.add_argument('--state-dim', type=int, default=8,
                    help='Dimensionality of the true state space.')
parser.add_argument('--sequence-length', type=int, default=16,
                    help='Sequence length.')
parser.add_argument('--diffusion-steps', type=int, default=50,
                    help='Number of diffusion steps.')
parser.add_argument('--num-classes', type=int, default=2,
                    help='Number of classes (safe and unsafe).')
parser.add_argument('--experiment', type=str, default='Reacher',
                    help='Experiment.')
parser.add_argument('--model-type', type=str, default='ProbDiffusion',
                    help='Model type.')
parser.add_argument('--training-dataset', type=str, default='reacher_train.pkl',
                    help='Training dataset.')
parser.add_argument('--testing-dataset', type=str, default='reacher_test.pkl',
                    help='Testing dataset.')
parser.add_argument('--log-interval', type=int, default=10,
                    help='How many batches to wait before saving')
parser.add_argument('--plot-interval', type=int, default=50,
                    help='How many batches to wait before plotting')
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed (default: 1).')
parser.add_argument('--use-attention', default=True,
                    help='Use attention layer in temporal U-net.')
parser.add_argument('--train-value', default=True,
                    help='Train value function or not.')
parser.add_argument('--train-cbfvalue', default=True,
                    help='Train cbf value function or not.')
parser.add_argument('--load', default=False,
                    help='Load trained model.')
parser.add_argument('--load-value', default=False,
                    help='Load pre-trained value function or not.')
parser.add_argument('--load-cbfvalue', default=False,
                    help='Load pre-trained cbf value function or not.')
args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('CUDA available:', torch.cuda.is_available())
    torch.cuda.manual_seed(args.seed)

# training hyperparameters
batch_size = args.batch_size
max_epoch = args.num_epochs

training = args.training
plotting = args.plotting
num_samples_plot = args.num_samples_plot

# learning rate
lr = args.learning_rate
reg_coef = args.reg_coefficient

# build model
act_dim = args.action_dim
state_dim = args.state_dim
h_dim = args.hidden_dim
sequence_length = args.sequence_length
num_classes = args.num_classes

# experiment and model type
exp = args.experiment
mtype = args.model_type
training_dataset = args.training_dataset
testing_dataset = args.testing_dataset

log_interval = args.log_interval
plot_interval = args.plot_interval

load = args.load
use_attention = args.use_attention

train_value = args.train_value
load_value = args.load_value

train_cbfvalue = args.train_cbfvalue
load_cbfvalue = args.load_cbfvalue

def main(exp='Reacher', mtype='ProbDiffusion', training_dataset='reacher_train.pkl', testing_dataset='reacher_test.pkl'):

    # load data
    directory = os.path.dirname(os.path.abspath(__file__))

    folder = os.path.join(directory + '/data', training_dataset)
    folder_test = os.path.join(directory + '/data', testing_dataset)

    data = load_pickle(folder)
    data_test = load_pickle(folder_test)

    now = datetime.now()
    save_pth_dir = directory + '/results/' + str(exp) + '/' + str(mtype)
    if not os.path.exists(save_pth_dir):
        os.makedirs(save_pth_dir)

    horizon = sequence_length
    observation_dim = state_dim
    action_dim = act_dim
    transition_dim = observation_dim + action_dim
    nr_diffusion_steps = args.diffusion_steps

    model = TemporalUnet(horizon, transition_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=use_attention)

    diffusion = ProbDiffusion(model, horizon, observation_dim, action_dim, n_timesteps=nr_diffusion_steps,
        loss_type='l2', clip_denoised=False, predict_epsilon=False,
        action_weight=1.0, loss_discount=1.0, loss_weights=None)

    ema_diffusion = copy.deepcopy(diffusion)

    if load:
        checkpoint = torch.load(save_pth_dir + '/ProbDiff_Model_best.pth')
        diffusion.model.load_state_dict(checkpoint['model'])
        ema_diffusion.model.load_state_dict(checkpoint['ema_model'])

    if train_value:
        value_model = ValueFunction(horizon, transition_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=use_attention,
                                    final_sigmoid=False)
        value_diffusion = ValueDiffusion(value_model, horizon, observation_dim, action_dim,
                                         n_timesteps=nr_diffusion_steps, loss_type='value_l2', clip_denoised=False,
                                         predict_epsilon=False, action_weight=1.0, loss_discount=1.0, loss_weights=None)
        ema_value_diffusion = copy.deepcopy(value_diffusion)
        if load_value:
            checkpoint = torch.load(save_pth_dir + '/ValueDiff_Model_best.pth')
            value_diffusion.model.load_state_dict(checkpoint['model'])
            ema_value_diffusion.model.load_state_dict(checkpoint['ema_model'])

    if train_cbfvalue:
        cbf_model = ValueFunction(horizon, transition_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=use_attention,
                                  out_dim=num_classes*horizon, final_sigmoid=True)
        cbf_diffusion = ValueDiffusion(cbf_model, horizon, observation_dim, action_dim, n_timesteps=nr_diffusion_steps,
                                       loss_type='cross_entropy', clip_denoised=False, predict_epsilon=False,
                                       action_weight=1.0, loss_discount=1.0, loss_weights=None)
        ema_cbf_diffusion = copy.deepcopy(cbf_diffusion)
        if load_cbfvalue:
            checkpoint = torch.load(save_pth_dir + '/CbfDiff_Model_best.pth')
            cbf_diffusion.model.load_state_dict(checkpoint['model'])
            ema_cbf_diffusion.model.load_state_dict(checkpoint['ema_model'])

    if torch.cuda.is_available():
        diffusion = diffusion.cuda()
        ema_diffusion = ema_diffusion.cuda()
        if train_value or load_value:
            value_diffusion = value_diffusion.cuda()
            ema_value_diffusion = ema_value_diffusion.cuda()
        if train_cbfvalue or load_cbfvalue:
            cbf_diffusion = cbf_diffusion.cuda()
            ema_cbf_diffusion = cbf_diffusion.cuda()
        gc.collect()

    # Use the adam optimizer
    optimizer = torch.optim.AdamW([
        {'params': diffusion.parameters()},
        ], lr=lr, weight_decay=reg_coef)

    if train_value:
        # Value optimizer
        value_optimizer = torch.optim.AdamW([
            {'params': value_diffusion.parameters()},
            ], lr=lr, weight_decay=reg_coef)

    if train_cbfvalue:
        # Cbf value optimizer
        cbf_optimizer = torch.optim.AdamW([
            {'params': cbf_diffusion.parameters()},
            ], lr=lr, weight_decay=reg_coef)

    counter = 0
    train_loader = ReplayBuffer(act_dim=act_dim, size=len(data), state_dim=state_dim)
    for d in data:
        train_loader.store(d[0].astype('float32'),
                           d[1].astype('float32'),
                           d[2],
                           d[3].astype('float32'),
                           d[4],
                           d[5])
        counter += 1

    print(counter)

    test_loader = ReplayBuffer(act_dim=act_dim, size=len(data_test), state_dim=state_dim)
    counter_t = 0
    for dt in data_test:
        test_loader.store(dt[0].astype('float32'),
                          dt[1].astype('float32'),
                          dt[2],
                          dt[3].astype('float32'),
                          dt[4],
                          dt[5])
        counter_t += 1
    print(counter_t)

    if training:
        best_loss_value = 1e6
        best_loss_dyn = 1e6
        best_loss_cbf = 1e6
        for epoch in range(0, max_epoch):
            diffusion.train()
            train(epoch=epoch, batch_size=batch_size, nr_data=counter, train_loader=train_loader, num_timesteps=horizon,
                  optimizer=optimizer, diffusion=diffusion, ema_diffusion=ema_diffusion)

            with torch.no_grad():
                test_loss_dyn = test_probdiffusion(epoch=epoch, batch_size=batch_size, nr_data=counter,
                                                   test_loader=test_loader, num_timesteps=horizon,
                                                   ema_diffusion=ema_diffusion)
                if test_loss_dyn < best_loss_dyn:
                    best_loss_dyn = test_loss_dyn
                    print("save best dyn model!")
                    torch.save({'model': diffusion.model.state_dict(), 'ema_model': ema_diffusion.model.state_dict()},
                               save_pth_dir + '/ProbDiff_Model_best.pth')
            if train_value:
                value_diffusion.train()
                train_valuefunction(epoch=epoch, batch_size=batch_size, nr_data=counter, train_loader=train_loader,
                                    num_timesteps=horizon, optimizer=value_optimizer, diffusion=value_diffusion,
                                    ema_diffusion=ema_value_diffusion)

                with torch.no_grad():
                    test_loss_value = test_valuefunction(epoch=epoch, batch_size=batch_size, nr_data=counter, test_loader=test_loader,
                                                         num_timesteps=horizon, ema_diffusion=ema_value_diffusion)
                    if test_loss_value < best_loss_value:
                        best_loss_value = test_loss_value
                        print("save best value model!")
                        torch.save({'model': value_diffusion.model.state_dict(), 'ema_model': ema_value_diffusion.model.state_dict()},
                                   save_pth_dir + '/ValueDiff_Model_best.pth')
            if train_cbfvalue:
                cbf_diffusion.train()
                train_cbfdiffusion(epoch=epoch, batch_size=batch_size, nr_data=counter, train_loader=train_loader,
                                   num_timesteps=horizon, optimizer=cbf_optimizer, diffusion=cbf_diffusion,
                                   ema_diffusion=ema_cbf_diffusion)

                with torch.no_grad():
                    test_loss_cbf = test_cbfdiffusion(epoch=epoch, batch_size=batch_size, nr_data=counter, train_loader=train_loader,
                                       num_timesteps=horizon, ema_diffusion=ema_cbf_diffusion)
                    if test_loss_cbf < best_loss_cbf:
                        best_loss_cbf = test_loss_cbf
                        print("save best cbf model!")
                        torch.save({'model': cbf_diffusion.model.state_dict(),
                                    'ema_model': ema_cbf_diffusion.model.state_dict()},
                                   save_pth_dir + '/CbfDiff_Model_best.pth')

            if epoch % plot_interval == 0:
                diffusion.eval()
                ema_diffusion.eval()
                with torch.no_grad():
                    plot_results(diffusion=diffusion, ema_diffusion=ema_diffusion, test_loader=test_loader,
                                 save_dir=save_pth_dir, num_timesteps=horizon, true_dyn=True, epoch=epoch)

        # save model after training is completed
        torch.save({'model': diffusion.model.state_dict(), 'ema_model': ema_diffusion.model.state_dict()},
                   save_pth_dir +'/ProbDiff_Model_latest.pth')
        if train_value:
            torch.save({'model': value_diffusion.model.state_dict(),
                        'ema_model': ema_value_diffusion.model.state_dict()}, save_pth_dir + '/ValueDiff_Model_latest.pth')
        if train_cbfvalue:
            torch.save({'model': cbf_diffusion.model.state_dict(),
                        'ema_model': ema_cbf_diffusion.model.state_dict()}, save_pth_dir + '/CbfDiff_Model_latest.pth')

    if plotting:
        diffusion.eval()
        ema_diffusion.eval()
        with torch.no_grad():
            plot_results(diffusion=diffusion, ema_diffusion=ema_diffusion, test_loader=test_loader,
                         save_dir=save_pth_dir, num_timesteps=horizon, true_dyn=True, epoch=max_epoch)

if __name__ == "__main__":

    main(exp=exp, mtype=mtype, training_dataset=training_dataset, testing_dataset=testing_dataset)
    print('Finished Training the DDPM models!')