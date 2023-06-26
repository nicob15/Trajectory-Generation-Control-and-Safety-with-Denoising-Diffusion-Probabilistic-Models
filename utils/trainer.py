from utils.losses import *
import numpy as np

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def reset_parameters(ema_model, model):
    ema_model.load_state_dict(model.state_dict())

def step_ema(ema_model, model, step, step_start_ema, ema):
    if step < step_start_ema:
        reset_parameters(ema_model, model)
        return
    ema.update_model_average(ema_model, model)

def train_ProbDiffusion(epoch, batch_size, nr_data, train_loader, optimizer, diffusion, ema_diffusion, num_timesteps=10,
                        ema_decay=0.995, update_ema_every=10, gradient_accumulate_every=2):

    ema_model = ema_diffusion
    ema = EMA(ema_decay)
    if epoch == 0:
        print("initialize ema model")
        reset_parameters(ema_model, diffusion)

    diffusion.train()

    train_loss = 0

    for i in range(int(nr_data/batch_size)):
        loss_t = 0
        for j in range(gradient_accumulate_every):
            #### sample batch of sequences ###
            data = train_loader.sample_sequence_batch(batch_size, num_timesteps)

            #### num_timesteps, batch_size, state/action_dim ###
            states = torch.from_numpy(data['states']).cuda()
            actions = torch.from_numpy(data['acts']).cuda()

            #### batch_size, num_timesteps, state/action_dim ###
            states = states.permute(1, 0, 2)
            actions = actions.permute(1, 0, 2)

            #### batch_size, num_timesteps, action_dim + state_dim ###
            traj = torch.cat([actions, states], dim=2)

            cond = {}

            loss, _ = diffusion.loss(traj, cond)
            loss = loss / gradient_accumulate_every
            loss_t += loss
            loss.backward()
        train_loss += loss_t.item()

        optimizer.step()
        optimizer.zero_grad()

        if epoch % update_ema_every == 0:
            step_ema(ema_model, diffusion, epoch, 10, ema)

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / nr_data))

def train_ValueDiffusion(epoch, batch_size, nr_data, train_loader, optimizer, diffusion, ema_diffusion, num_timesteps=10,
                        ema_decay=0.995, update_ema_every=10, gradient_accumulate_every=2, discount=0.997):

    ema_model = ema_diffusion
    ema = EMA(ema_decay)
    if epoch == 0:
        print("initialize ema value model")
        reset_parameters(ema_model, diffusion)

    diffusion.train()

    train_loss = 0

    discounts = (discount ** np.arange(200)[:, None]).astype('float32')

    for i in range(int(nr_data/batch_size)):
        loss_t = 0
        for j in range(gradient_accumulate_every):
            #### sample batch of sequences ###
            data = train_loader.sample_sequence_batch(batch_size, num_timesteps)

            #### num_timesteps, batch_size, state/action_dim ###
            states = torch.from_numpy(data['states']).cuda()
            actions = torch.from_numpy(data['acts']).cuda()
            rewards = torch.from_numpy(data['rews'].reshape(num_timesteps, batch_size, 1)).cuda()

            #### batch_size, num_timesteps, state/action_dim ###
            states = states.permute(1, 0, 2)
            actions = actions.permute(1, 0, 2)
            rewards = rewards.permute(1, 0, 2)

            #### compute target values ###
            discounts = discounts[:rewards.shape[1]]
            target_value = torch.from_numpy(discounts).cuda() * rewards
            target_value = torch.sum(target_value, dim=1).float().reshape(batch_size, 1)

            #### batch_size, num_timesteps, action_dim + state_dim ###
            traj = torch.cat([actions, states], dim=2)

            cond = {}

            loss, _ = diffusion.loss(traj, cond, target_value)
            loss = loss / gradient_accumulate_every
            loss_t += loss
            loss.backward()
        train_loss += loss_t.item()

        optimizer.step()
        optimizer.zero_grad()

        if epoch % update_ema_every == 0:
            step_ema(ema_model, diffusion, epoch, 10, ema)

    print('====> Epoch: {} Average value loss: {:.4f}'.format(epoch, train_loss / nr_data))

def train_CbfDiffusion(epoch, batch_size, nr_data, train_loader, optimizer, diffusion, ema_diffusion, num_timesteps=10,
                        ema_decay=0.995, update_ema_every=10, gradient_accumulate_every=2, discount=1.0):

    ema_model = ema_diffusion
    ema = EMA(ema_decay)
    if epoch == 0:
        print("initialize ema cbf value model")
        reset_parameters(ema_model, diffusion)

    diffusion.train()

    train_loss = 0

    discounts = (discount ** np.arange(200)[:, None]).astype('float32')

    for i in range(int(nr_data/batch_size)):
        loss_t = 0
        for j in range(gradient_accumulate_every):
            #### sample batch of sequences ###
            data = train_loader.sample_sequence_batch(batch_size, num_timesteps)

            #### num_timesteps, batch_size, state/action_dim ###
            states = torch.from_numpy(data['states']).cuda()
            actions = torch.from_numpy(data['acts']).cuda()
            cbf_values = torch.from_numpy(data['cbf_values'].reshape(num_timesteps, batch_size, 1)).cuda()

            #### batch_size, num_timesteps, state/action_dim ###
            states = states.permute(1, 0, 2)
            actions = actions.permute(1, 0, 2)
            cbf_values = cbf_values.permute(1, 0, 2)

            #### compute target values ###
            discounts = discounts[:cbf_values.shape[1]]
            target_cbfvalue = torch.from_numpy(discounts).cuda() * cbf_values

            #### batch_size, num_timesteps, action_dim + state_dim ###
            traj = torch.cat([actions, states], dim=2)

            cond = {}

            loss, _ = diffusion.loss(traj, cond, target_cbfvalue.reshape(batch_size, -1))
            loss = loss / gradient_accumulate_every
            loss_t += loss
            loss.backward()
        train_loss += loss_t.item()

        optimizer.step()
        optimizer.zero_grad()

        if epoch % update_ema_every == 0:
            step_ema(ema_model, diffusion, epoch, 10, ema)

    print('====> Epoch: {} Average cbf loss: {:.4f}'.format(epoch, train_loss / nr_data))

def test_ProbDiffusion(epoch, batch_size, nr_data, test_loader,ema_diffusion, num_timesteps=10,
                       gradient_accumulate_every=2):

    ema_diffusion.eval()

    test_loss = 0

    for i in range(int(nr_data/batch_size)):
        loss_t = 0
        for j in range(gradient_accumulate_every):
            #### sample batch of sequences ###
            data = test_loader.sample_sequence_batch(batch_size, num_timesteps)

            #### num_timesteps, batch_size, state/action_dim ###
            states = torch.from_numpy(data['states']).cuda()
            actions = torch.from_numpy(data['acts']).cuda()

            #### batch_size, num_timesteps, state/action_dim ###
            states = states.permute(1, 0, 2)
            actions = actions.permute(1, 0, 2)

            #### batch_size, num_timesteps, action_dim + state_dim ###
            traj = torch.cat([actions, states], dim=2)

            cond = {}

            loss, _ = ema_diffusion.loss(traj, cond)
            loss = loss / gradient_accumulate_every
            loss_t += loss

        test_loss += loss_t.item()

    print('====> Epoch: {} Average test loss (ema model): {:.4f}'.format(epoch, test_loss / nr_data))
    return (test_loss / nr_data)

def test_ValueDiffusion(epoch, batch_size, nr_data, test_loader, ema_diffusion, num_timesteps=10,
                        gradient_accumulate_every=2, discount=0.997):

    ema_diffusion.eval()

    test_loss = 0

    discounts = discount ** np.arange(200)[:, None]

    for i in range(int(nr_data/batch_size)):
        loss_t = 0
        for j in range(gradient_accumulate_every):
            #### sample batch of sequences ###
            data = test_loader.sample_sequence_batch(batch_size, num_timesteps)

            #### num_timesteps, batch_size, state/action_dim ###
            states = torch.from_numpy(data['states']).cuda()
            actions = torch.from_numpy(data['acts']).cuda()
            rewards = torch.from_numpy(data['rews'].reshape(num_timesteps, batch_size, 1)).cuda()

            #### batch_size, num_timesteps, state/action_dim ###
            states = states.permute(1, 0, 2)
            actions = actions.permute(1, 0, 2)
            rewards = rewards.permute(1, 0, 2)

            #### compute target values ###
            discounts = discounts[:rewards.shape[1]]
            target_value = torch.from_numpy(discounts).cuda() * rewards
            target_value = torch.sum(target_value, dim=1).float().reshape(batch_size, 1)

            #### batch_size, num_timesteps, action_dim + state_dim ###
            traj = torch.cat([actions, states], dim=2)

            cond = {}

            loss, _ = ema_diffusion.loss(traj, cond, target_value)
            loss = loss / gradient_accumulate_every
            loss_t += loss
        test_loss += loss_t.item()

    print('====> Epoch: {} Average test value loss (ema model): {:.4f}'.format(epoch, test_loss / nr_data))
    return (test_loss / nr_data)

def test_CbfDiffusion(epoch, batch_size, nr_data, train_loader, ema_diffusion, num_timesteps=10,
                      gradient_accumulate_every=2, discount=1.0):

    ema_diffusion.eval()

    test_loss = 0

    discounts = (discount ** np.arange(200)[:, None]).astype('float32')

    for i in range(int(nr_data/batch_size)):
        loss_t = 0
        for j in range(gradient_accumulate_every):
            #### sample batch of sequences ###
            data = train_loader.sample_sequence_batch(batch_size, num_timesteps)

            #### num_timesteps, batch_size, state/action_dim ###
            states = torch.from_numpy(data['states']).cuda()
            actions = torch.from_numpy(data['acts']).cuda()
            cbf_values = torch.from_numpy(data['cbf_values'].reshape(num_timesteps, batch_size, 1)).cuda()

            #### batch_size, num_timesteps, state/action_dim ###
            states = states.permute(1, 0, 2)
            actions = actions.permute(1, 0, 2)
            cbf_values = cbf_values.permute(1, 0, 2)

            #### compute target values ###
            discounts = discounts[:cbf_values.shape[1]]
            target_cbfvalue = torch.from_numpy(discounts).cuda() * cbf_values

            #### batch_size, num_timesteps, action_dim + state_dim ###
            traj = torch.cat([actions, states], dim=2)

            cond = {}

            loss, _ = ema_diffusion.loss(traj, cond, target_cbfvalue.reshape(batch_size, -1))
            loss = loss / gradient_accumulate_every
            loss_t += loss
        test_loss += loss_t.item()

    print('====> Epoch: {} Average test cbf loss: {:.4f}'.format(epoch, test_loss / nr_data))
    return (test_loss / nr_data)


def train_all(epoch, batch_size, nr_data, train_loader, optimizer, optimizer_value, optimizer_cbf, diffusion,
              ema_diffusion, value_diffusion, ema_value_diffusion, cbf_diffusion, ema_cbf_diffusion, num_timesteps=10,
              ema_decay=0.995, update_ema_every=10, gradient_accumulate_every=2, discount=0.997, cbf_discount=1.0):

    ema_model = ema_diffusion
    ema = EMA(ema_decay)
    if epoch == 0:
        print("initialize ema model")
        reset_parameters(ema_model, diffusion)

    ema_value_model = ema_value_diffusion
    ema_value = EMA(ema_decay)
    if epoch == 0:
        print("initialize ema model")
        reset_parameters(ema_value_model, value_diffusion)

    ema_cbf_model = ema_cbf_diffusion
    ema_cbf = EMA(ema_decay)
    if epoch == 0:
        print("initialize ema model")
        reset_parameters(ema_cbf_model, cbf_diffusion)

    diffusion.train()
    value_diffusion.train()
    cbf_diffusion.train()

    train_loss = 0
    train_loss_value = 0
    train_loss_cbf = 0

    discounts = discount ** np.arange(200)[:, None]
    cbf_discounts = cbf_discount ** np.arange(200)[:, None]

    for i in range(int(nr_data/batch_size)):
        loss_t = 0
        loss_value_t = 0
        loss_cbf_t = 0
        for j in range(gradient_accumulate_every):
            #### sample batch of sequences ###
            data = train_loader.sample_sequence_batch(batch_size, num_timesteps)

            #### num_timesteps, batch_size, state/action_dim ###
            states = torch.from_numpy(data['states']).cuda()
            actions = torch.from_numpy(data['acts']).cuda()
            rewards = torch.from_numpy(data['rews'].reshape(num_timesteps, batch_size, 1)).cuda()
            cbf_values = torch.from_numpy(data['cbf_values'].reshape(num_timesteps, batch_size, 1)).cuda()

            #### batch_size, num_timesteps, state/action_dim ###
            states = states.permute(1, 0, 2)
            actions = actions.permute(1, 0, 2)
            rewards = rewards.permute(1, 0, 2)
            cbf_values = cbf_values.permute(1, 0, 2)

            #### batch_size, num_timesteps, action_dim + state_dim ###
            traj = torch.cat([actions, states], dim=2)

            discounts = discounts[:rewards.shape[1]]
            target_value = torch.from_numpy(discounts).cuda() * rewards
            target_value = torch.sum(target_value, dim=1).float().reshape(batch_size, 1)

            cbf_discounts = cbf_discounts[:cbf_values.shape[1]]
            target_cbfvalue = torch.from_numpy(discounts).cuda() * cbf_values
            target_cbfvalue = target_cbfvalue.float().reshape(batch_size, -1)

            cond = {}

            loss, _ = diffusion.loss(traj, cond)
            loss = loss / gradient_accumulate_every
            loss_t += loss
            loss.backward()

            loss_value, _ = value_diffusion.loss(traj, cond, target_value)
            loss_value = loss_value / gradient_accumulate_every
            loss_value_t += loss_value
            loss_value.backward()

            loss_cbf, _ = cbf_diffusion.loss(traj, cond, target_cbfvalue)
            loss_cbf = loss_cbf / gradient_accumulate_every
            loss_cbf_t += loss_cbf
            loss_cbf.backward()

        train_loss += loss_t.item()
        train_loss_value += loss_value_t.item()
        train_loss_cbf += loss_cbf_t.item()

        optimizer.step()
        optimizer.zero_grad()

        optimizer_value.step()
        optimizer_value.zero_grad()

        optimizer_cbf.step()
        optimizer_cbf.zero_grad()

        if epoch % update_ema_every == 0:
            step_ema(ema_model, diffusion, epoch, 10, ema)
            step_ema(ema_value_model, value_diffusion, epoch, 10, ema_value)
            step_ema(ema_cbf_model, cbf_diffusion, epoch, 10, ema_cbf)

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / nr_data))
    print('====> Epoch: {} Average value loss: {:.4f}'.format(epoch, train_loss_value / nr_data))
    print('====> Epoch: {} Average cbf loss: {:.4f}'.format(epoch, train_loss_cbf / nr_data))


def test_all(epoch, batch_size, nr_data, test_loader, diffusion,
             ema_diffusion, value_diffusion, ema_value_diffusion, cbf_diffusion, ema_cbf_diffusion, num_timesteps=10,
            gradient_accumulate_every=2, discount=0.997, cbf_discount=1.0):

    diffusion.eval()
    value_diffusion.eval()
    cbf_diffusion.eval()

    ema_diffusion.eval()
    ema_value_diffusion.eval()
    ema_cbf_diffusion.eval()

    test_loss = 0
    test_loss_value = 0
    test_loss_cbf = 0

    discounts = discount ** np.arange(200)[:, None]
    cbf_discounts = cbf_discount ** np.arange(200)[:, None]

    for i in range(int(nr_data / batch_size)):
        loss_t = 0
        loss_value_t = 0
        loss_cbf_t = 0
        for j in range(gradient_accumulate_every):
            #### sample batch of sequences ###
            data = test_loader.sample_sequence_batch(batch_size, num_timesteps)

            #### num_timesteps, batch_size, state/action_dim ###
            states = torch.from_numpy(data['states']).cuda()
            actions = torch.from_numpy(data['acts']).cuda()
            rewards = torch.from_numpy(data['rews'].reshape(num_timesteps, batch_size, 1)).cuda()
            cbf_values = torch.from_numpy(data['cbf_values'].reshape(num_timesteps, batch_size, 1)).cuda()

            #### batch_size, num_timesteps, state/action_dim ###
            states = states.permute(1, 0, 2)
            actions = actions.permute(1, 0, 2)
            rewards = rewards.permute(1, 0, 2)
            cbf_values = cbf_values.permute(1, 0, 2)

            #### batch_size, num_timesteps, action_dim + state_dim ###
            traj = torch.cat([actions, states], dim=2)

            discounts = discounts[:rewards.shape[1]]
            target_value = torch.from_numpy(discounts).cuda() * rewards
            target_value = torch.sum(target_value, dim=1).float().reshape(batch_size, 1)

            cbf_discounts = cbf_discounts[:cbf_values.shape[1]]
            target_cbfvalue = torch.from_numpy(discounts).cuda() * cbf_values
            target_cbfvalue = target_cbfvalue.float().reshape(batch_size, -1)

            cond = {}

            loss, _ = ema_diffusion.loss(traj, cond)
            loss = loss / gradient_accumulate_every
            loss_t += loss

            loss_value, _ = ema_value_diffusion.loss(traj, cond, target_value)
            loss_value = loss_value / gradient_accumulate_every
            loss_value_t += loss_value

            loss_cbf, _ = ema_cbf_diffusion.loss(traj, cond, target_cbfvalue)
            loss_cbf = loss_cbf / gradient_accumulate_every
            loss_cbf_t += loss_cbf

        test_loss += loss_t.item()
        test_loss_value += loss_value_t.item()
        test_loss_cbf += loss_cbf_t.item()

    print('====> Epoch: {} Average test loss: {:.4f}'.format(epoch, test_loss / nr_data))
    print('====> Epoch: {} Average test value loss: {:.4f}'.format(epoch, test_loss_value / nr_data))
    print('====> Epoch: {} Average test cbf loss: {:.4f}'.format(epoch, test_loss_cbf / nr_data))

    return (test_loss / nr_data), (test_loss_value / nr_data), (test_loss_cbf / nr_data)
