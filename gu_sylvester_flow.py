"""
Variational Inference with Normalizing Flows
arXiv:1505.05770v6
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torchvision.transforms as T
from torchvision.utils import save_image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn.apionly as sns
plt.style.use('seaborn-paper')

import sys
import argparse
import time

from util import *
from gu import *

import torch.distributions.transforms.SylvesterFlow

parser = argparse.ArgumentParser()
# data
parser.add_argument('--gu_num', type=int, default=8, help='Components of GU clusters.')
parser.add_argument('--dataset', default='GU', help='Which dataset to use.')
parser.add_argument('--seed', type=int, default=1, help='Random seed to use.')
# model
parser.add_argument('--model', default='planar', choices=['planar', 'sylvester'], help='Which model to use: planar, sylvester.')
# model parameters
parser.add_argument('--n_blocks', type=int, default=5, help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
parser.add_argument('--hidden_size', type=int, default=64, help='Hidden layer size for MADE (and each MADE block in an MAF).')
parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each MADE.')
parser.add_argument('--no_learn_base', action='store_true', help='Whether to learn a mu-sigma affine transform of the base distribution.')

# training params
parser.add_argument('--batch_size', type=int, default=2048, help='Batch size in training.')
parser.add_argument('--niters', type=int, default=50000, help='Total iteration numbers in training.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay in Adam.')
parser.add_argument('--beta1', type=float, default=0.9, help='Beta 1 in Adam.')
parser.add_argument('--beta2', type=float, default=0.999, help='Beta 2 in Adam.')

parser.add_argument('--log_interval', type=int, default=1000, help='How often to show loss statistics and save models/samples.')

parser.add_argument('--clr', action='store_true', help='Use cyclic LR in training.')
parser.add_argument('--clr_size_up', type=int, default=2000, help='Size of up step in cyclic LR.')
parser.add_argument('--clr_scale', type=int, default=3, help='Scale of base lr in cyclic LR.')

parser.add_argument('--cuda', type=int, default=2, help='Number of CUDA to use if available.')
parser.add_argument('--eval_size', type=int, default=100000, help='Sample size in evaluation.')


# --------------------
# Flow
# --------------------

# class SylvesterTransform(nn.Module):
#     def __init__(self, hidden_size, init_sigma=0.01):
#         super().__init__()
#         self.u = nn.Parameter(torch.randn(hidden_size, 1).normal_(0, init_sigma))
#         self.w = nn.Parameter(torch.randn(hidden_size).normal_(0, init_sigma))
#         self.b = nn.Parameter(torch.randn(hidden_size).fill_(0))
#
#     def forward(self, x):
#         # allow for a single forward pass over all the transforms in the flows with a Sequential container
#         if isinstance(x, tuple):
#             z, sum_log_abs_det_jacobians = x
#         else:
#             z, sum_log_abs_det_jacobians = x, 0
#
#         # Support sufficient condition for invertibility.
#         u_hat = torch.tanh(self.u)
#         w_hat = torch.tanh(self.w)
#
#         f_z = z + (torch.tanh(z * w_hat + self.b) @ u_hat)
#         Sii = 1 - (torch.tanh(z * w_hat + self.b)**2).squeeze()
#         # print(Sii.shape)
#         det = 1. + (Sii * w_hat * u_hat.squeeze()).sum(1)
#         # print( torch.log(torch.abs(det) + 1e-6).shape)
#         # exit(1)
#         log_abs_det_jacobian = torch.log(torch.abs(det) + 1e-6).squeeze()
#         sum_log_abs_det_jacobians += log_abs_det_jacobian#.view(f_z.size(0), -1)
#         # print(sum_log_abs_det_jacobians.shape)
#         # exit(1)
#         # print(f_z.shape)
#         # print(z.shape)
#         # exit(1)
#         return f_z, sum_log_abs_det_jacobians

class SylvesterTransform(nn.Module):
    def __init__(self, hidden_size, init_sigma=0.01):
        super().__init__()
        torch.manual_seed(args.seed)
        self.v = nn.Parameter(torch.randn(hidden_size, 1).normal_(0, init_sigma))
        self.w = nn.Parameter(torch.randn(1, hidden_size).normal_(0, init_sigma))
        self.b = nn.Parameter(torch.randn(hidden_size).fill_(0))

    def forward(self, x):
        # allow for a single forward pass over all the transforms in the flows with a Sequential container
        """

        :param x: input batch: (batch_size, 1)
        :return: f_z: transformed batch: (batch_size, 1)
                 sum_log_abs_det_jacobian: (batch_size, )
        """
        if isinstance(x, tuple):
            z, sum_log_abs_det_jacobians = x
        else:
            z, sum_log_abs_det_jacobians = x, 0

        # Support sufficient condition for invertibility.
        v_hat = torch.tanh(self.v)
        w_hat = torch.tanh(self.w)

        # Compute transform
        f_z = z + torch.tanh(z @ w_hat + self.b) @ v_hat
        # Compute derivative term of tanh()
        Sii = 1 - torch.tanh(z @ w_hat + self.b)**2
        # Compute the Jacobian
        det = 1. + (w_hat.squeeze() * Sii * v_hat.squeeze()).sum(1)
        log_abs_det_jacobian = torch.log(torch.abs(det) + 1e-6).squeeze()
        sum_log_abs_det_jacobians += log_abs_det_jacobian
        return f_z, sum_log_abs_det_jacobians


class PlanarTransform(nn.Module):
    def __init__(self, init_sigma=0.01):
        super().__init__()
        self.u = nn.Parameter(torch.randn(1, 1).normal_(0, init_sigma))
        self.w = nn.Parameter(torch.randn(1, 1).normal_(0, init_sigma))
        self.b = nn.Parameter(torch.randn(1).fill_(0))

    def forward(self, x, normalize_u=True):
        # allow for a single forward pass over all the transforms in the flows with a Sequential container
        if isinstance(x, tuple):
            z, sum_log_abs_det_jacobians = x
        else:
            z, sum_log_abs_det_jacobians = x, 0

        # normalize u s.t. w @ u >= -1; sufficient condition for invertibility
        u_hat = self.u
        if normalize_u:
            wtu = (self.w @ self.u.t()).squeeze()
            m_wtu = - 1 + torch.log1p(wtu.exp())
            u_hat = self.u + (m_wtu - wtu) * self.w / (self.w @ self.w.t())

        # compute transform
        f_z = z + u_hat * torch.tanh(z @ self.w.t() + self.b)
        # compute log_abs_det_jacobian
        psi = (1 - torch.tanh(z @ self.w.t() + self.b)**2) @ self.w
        det = 1 + psi @ u_hat.t()
        log_abs_det_jacobian = torch.log(torch.abs(det) + 1e-6).squeeze()
        sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return f_z, sum_log_abs_det_jacobians


class AffineTransform(nn.Module):
    def __init__(self, learnable=False):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(1)).requires_grad_(learnable)
        self.logsigma = nn.Parameter(torch.zeros(1)).requires_grad_(learnable)

    def forward(self, x):
        # allow for a single forward pass over all the transforms in the flows with a Sequential container
        if isinstance(x, tuple):
            z, sum_log_abs_det_jacobians = x
        else:
            z, sum_log_abs_det_jacobians = x, 0

        f_z = self.mu + self.logsigma.exp() * z
        sum_log_abs_det_jacobians += self.logsigma.sum()
        return f_z, sum_log_abs_det_jacobians


class SylvesterFLow(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, **kwargs):
        super().__init__()

        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # construct model
        modules = []
        for i in range(n_blocks):
            if args.model == 'sylvester':
                modules += [SylvesterTransform(hidden_size, kwargs.get('init_sigma', 0.01))]
            else:
                modules += [PlanarTransform(kwargs.get('init_sigma', 0.01))]
        # modules += [AffineTransform(True)]
        self.net = nn.Sequential(*modules)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x):
        return self.net(x)

    def log_prob(self, x):
        u, sum_log_abs_det_jacobians = self.forward(x)
        return self.base_dist.log_prob(u) + sum_log_abs_det_jacobians.view(u.size(0), -1)


def train(model, dataloader, optimizer, scheduler, args):

    model.train()
    start = time.time()
    running_loss = 0.
    for i in range(1, args.niters + 1):
        x = dataloader.get_sample(args.batch_size)
        x = x.view(x.shape[0], -1).to(args.device)
        loss = - model.log_prob(x).mean()
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        if i % args.log_interval == 0:
            with torch.no_grad():
                # save model
                cur_state_path = os.path.join(model_path, str(i))
                torch.save(model, cur_state_path + '.pth')

                real = dataloader.get_sample(args.eval_size)

                logger.info(f'Iter {i} / {args.niters}, Time {round(time.time() - start, 4)},  '
                            f'w_distance_real: {w_distance_real}, loss {round(running_loss / args.log_interval, 5)}')

                # plot.
                plt.cla()
                fig = plt.figure(figsize=(FIG_W, FIG_H))
                ax = fig.add_subplot(111)
                ax.set_facecolor('whitesmoke')
                ax.grid(True, color='white', linewidth=2)

                ax.spines["top"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.get_xaxis().tick_bottom()

                real_sample = real.cpu().data.numpy().squeeze()
                x_min, x_max = min(real_sample), max(real_sample)
                range_width = x_max - x_min

                eval_width = range_width / args.eval_size
                eval_range = torch.arange(x_min, x_max, eval_width, device=args.device)
                eval_range = eval_range.unsqueeze(-1)
                prob = model.log_prob(eval_range).exp()
                eval_range = eval_range.cpu().data.numpy().squeeze()
                prob_learned = prob.cpu().data.numpy().squeeze()

                kde_num = 200
                kde_width = kde_num * range_width / args.eval_size
                sns.kdeplot(real_sample, bw=kde_width, label='Data', color='green', shade=True, linewidth=6)
                ax_r = ax.twinx()
                ax_r.plot(eval_range, prob_learned, label='Model', color='orange', linewidth=6)

                ax.set_title(f'True EM Distance: {w_distance_real}.', fontsize=FONTSIZE)
                ax.legend(loc=2, fontsize=FONTSIZE)
                ax.set_ylabel('Estimated Density by KDE', fontsize=FONTSIZE)
                ax.tick_params(axis='x', labelsize=FONTSIZE * 0.7)
                ax.tick_params(axis='y', labelsize=FONTSIZE * 0.5, direction='in')

                cur_img_path = os.path.join(image_path, str(i) + '.jpg')
                plt.tight_layout()

                plt.savefig(cur_img_path)
                plt.close()

                start = time.time()
                running_loss = 0


if __name__ == '__main__':

    w_distance_real = -1

    args = parser.parse_args()
    args.learn_base = not args.no_learn_base

    curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = os.path.split(curPath)[0]
    sys.path.append(rootPath)

    search_type = 'manual'
    experiment = f'gu{args.gu_num}/{args.model}_flow/{args.niters}'

    model_path = os.path.join(rootPath, search_type, 'models', experiment)
    image_path = os.path.join(rootPath, search_type, 'images', experiment)
    makedirs(model_path, image_path)
    log_path = model_path + '/logs'
    logger = get_logger(log_path)

    # setup device
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    if args.device.type == 'cuda': torch.cuda.manual_seed(args.seed)

    # load data
    if args.gu_num == 8:
        dataloader = GausUniffMixture(n_mixture=args.gu_num, mean_dist=10, sigma=2, unif_intsect=1.5, unif_ratio=1.,
                                      device=args.device, extend_dim=False)
    else:
        dataloader = GausUniffMixture(n_mixture=args.gu_num, mean_dist=5, sigma=0.1, unif_intsect=5, unif_ratio=3,
                                      device=args.device, extend_dim=False)
    args.input_size = 1
    args.input_dims = 1

    model = SylvesterFLow(args.n_blocks, args.input_size, args.hidden_size).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))
    if args.clr:
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr / args.clr_scale, max_lr=args.lr,
                                                      step_size_up=args.clr_size_up, cycle_momentum=False)
    else:
        scheduler = None

    logger.info('Start training...')
    train(model, dataloader, optimizer, scheduler, args)
    logger.info('Finish All...')

