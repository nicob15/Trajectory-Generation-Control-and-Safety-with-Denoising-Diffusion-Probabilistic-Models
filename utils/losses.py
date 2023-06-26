import torch.nn.functional as F
import torch
import torch.distributions as td

# return reconstruction error + KL divergence losses
def loss_function_VAE(recon_x, x, mu, log_var, beta=1):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + beta*KLD

def loss_bce(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size(0)
    return BCE

def kl_divergence(mu_1, var_1, mu_2, var_2, dim=1):
    p = td.Independent(td.Normal(mu_1, torch.sqrt(var_1)), dim)
    q = td.Independent(td.Normal(mu_2, torch.sqrt(var_2)), dim)
    div = td.kl_divergence(p, q)
    div = torch.max(div, div.new_full(div.size(), 3))
    return torch.mean(div)

def kl_divergence_balance(mu_1, var_1, mu_2, var_2, alpha=0.8, dim=1):
    p = td.Independent(td.Normal(mu_1, torch.sqrt(var_1)), dim)
    p_stop_grad = td.Independent(td.Normal(mu_1.detach(), torch.sqrt(var_1.detach())), dim)
    q = td.Independent(td.Normal(mu_2, torch.sqrt(var_2)), dim)
    q_stop_grad = td.Independent(td.Normal(mu_2.detach(), torch.sqrt(var_2.detach())), dim)
    div = alpha * td.kl_divergence(p_stop_grad, q) + (1 - alpha) * td.kl_divergence(p, q_stop_grad)
    div = torch.max(div, div.new_full(div.size(), 3))
    return torch.mean(div)

def loss_negloglikelihood(mu, target, std, dim=1):
    normal_dist = torch.distributions.Independent(torch.distributions.Normal(mu, std), dim)
    return -torch.mean(normal_dist.log_prob(target))

def loglikelihood_analitical_loss(mu, target, std):
    diff = ((mu - target) / std).pow(2)
    loss = 0.5 * diff + torch.log(std)
    return torch.mean(loss)
