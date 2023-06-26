import torch
import torch.nn as nn
import pdb
from diffuser.models.helpers import (
    extract,
    apply_conditioning,
)

import diffuser.utils as utils
import einops

class ValueGuide(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, cond, t):
        output = self.model(x, cond, t)
        return output.squeeze(dim=-1)

    def gradients(self, x, *args):
        x.requires_grad_()
        y = self(x, *args)
        grad = torch.autograd.grad([y.sum()], [x])[0]
        x.detach()
        return y, grad

class CbfGuide(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, cond, t):
        output = self.model(x, cond, t)
        return output.squeeze(dim=-1)

    def gradients(self, x, *args):
        x.requires_grad_()
        y = self(x, *args)
        n_classes = 2
        y2 = y.reshape(y.shape[0]*int(y.shape[1]/n_classes), -1)
        labels = torch.zeros((y2.shape[0],))
        log_probs = torch.nn.functional.log_softmax(y2, dim=-1)
        selected = log_probs[range(len(y2)), labels.view(-1).type(torch.long)]
        grad = torch.autograd.grad([selected.reshape(y.shape[0], -1).sum()], [x])[0]
        x.detach()
        # torch.nn.functional.softmax(y2, dim=-1).reshape(y.shape[0], -1)
        return y, grad

    # logits = classifier(x_in, t)
    # log_probs = F.log_softmax(logits, dim=-1)
    # selected = log_probs[range(len(logits)), y.view(-1)]
    # return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

@torch.no_grad()
def n_step_guided_p_sample(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=2, n_guide_steps=2, scale_grad_by_std=True): #0.01 #2 #2

    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    for _ in range(n_guide_steps):
        with torch.enable_grad():
            #y, grad = guide.gradients(x[:, :, 1:], cond, t)
            y, grad = guide.gradients(x, cond, t)

        if scale_grad_by_std:
            grad = model_var * grad

        grad[t < t_stopgrad] = 0

        x = x + scale * grad
        #x[:, :, 1:] = x[:, :, 1:] + scale * grad
        x = apply_conditioning(x, cond, model.action_dim)

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    return model_mean + model_std * noise, y

@torch.no_grad()
def n_step_doubleguided_p_sample(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=2, n_guide_steps=2, scale_grad_by_std=True, scale_cbf=5.0): #0.01 #2 #4

    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    for _ in range(n_guide_steps):
        with torch.enable_grad():
            y0, grad0 = guide[0].gradients(x, cond, t)
            y1, grad1 = guide[1].gradients(x, cond, t)

        if scale_grad_by_std:
            # scale_grad1 = torch.abs(grad0.sum()) / torch.abs(grad1.sum())
            # print("grad scale", scale_grad1)
            # print("t", t)
            grad0 = model_var * grad0
            #grad1 = model_var * scale_grad1 * grad1
            grad1 = model_var * grad1

        grad0[t < t_stopgrad] = 0
        grad1[t < t_stopgrad] = 0

        x = x + scale * (grad0) + scale_cbf * (grad1)
        x = apply_conditioning(x, cond, model.action_dim)

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    return model_mean + model_std * noise, y0, y1