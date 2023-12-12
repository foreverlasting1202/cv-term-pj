from .ray_sampler import sdf_to_sigma, fine_sample

from torch import autograd
import copy
import functools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import imp
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.net_utils import batchify_query
from lib.config import cfg

class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super(Embedder, self).__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(
                2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input: torch.Tensor):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out


def get_embedder(multires, input_dim=3):
    if multires < 0:
        return nn.Identity(), input_dim

    embed_kwargs = {
        "include_input": True,  # needs to be True for ray_bending to work properly
        "input_dim": input_dim,
        "max_freq_log2": multires - 1,
        "N_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim


class DenseLayer(nn.Linear):
    def __init__(self, input_dim: int, out_dim: int, *args, activation=None, **kwargs):
        super().__init__(input_dim, out_dim, *args, **kwargs)
        if activation is None:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = activation

    def forward(self, x):
        out = super().forward(x)
        out = self.activation(out)
        return out


class SDFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.D = cfg.model.sdf.net_depth
        self.W = cfg.model.net_width
        self.W_geo_feat = cfg.model.feature_width
        self.skips = cfg.model.sdf.skips
        embed_multires = cfg.model.sdf.fr_pos
        self.embed_fn, input_ch = get_embedder(embed_multires)

        surface_fc_layers = []
        # NOTE: as in IDR/NeuS, the network's has D+1 layers
        for l in range(self.D+1):
            # decide out_dim
            if l == self.D:
                if self.W_geo_feat > 0:
                    out_dim = 1 + self.W_geo_feat
                else:
                    out_dim = 1
            elif (l+1) in self.skips:
                out_dim = self.W - input_ch  # recude output dim before the skips layers, as in IDR / NeuS
            else:
                out_dim = self.W
                
            # decide in_dim
            if l == 0:
                in_dim = input_ch
            else:
                in_dim = self.W
            
            if l != self.D:
                layer = DenseLayer(in_dim, out_dim, activation=nn.Softplus(beta=100))
            else:
                layer = nn.Linear(in_dim, out_dim)

            # if true preform preform geometric initialization
            if cfg.model.sdf.geometric_init:
                #--------------
                # sphere init, as in SAL / IDR.
                #--------------
                if l == self.D:
                    nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                    nn.init.constant_(layer.bias, -cfg.model.sdf.radius_init) 
                elif embed_multires > 0 and l == 0:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.constant_(layer.weight[:, 3:], 0.0)   # let the initial weights for octaves to be 0.
                    torch.nn.init.normal_(layer.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif embed_multires > 0 and l in self.skips:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(layer.weight[:, -(input_ch - 3):], 0.0) # NOTE: this contrains the concat order to be  [h, x_embed]
                else:
                    nn.init.constant_(layer.bias, 0.0)
                    nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if cfg.model.sdf.weight_norm:
                layer = nn.utils.weight_norm(layer)

            surface_fc_layers.append(layer)

        self.surface_fc_layers = nn.ModuleList(surface_fc_layers)

    def forward(self, x: torch.Tensor, return_h = False):
        x = self.embed_fn(x)
        
        h = x
        for i in range(self.D):
            if i in self.skips:
                # NOTE: concat order can not change! there are special operations taken in intialization.
                h = torch.cat([h, x], dim=-1) / np.sqrt(2)
            h = self.surface_fc_layers[i](h)
        
        out = self.surface_fc_layers[-1](h)
        
        if self.W_geo_feat > 0:
            h = out[..., 1:]
            out = out[..., :1].squeeze(-1)
        else:
            out = out.squeeze(-1)
        
        out = -out  # make it suitable to inside-out scene

        if return_h:
            return out, h
        else:
            return out
    
    def forward_with_nablas(self,  x: torch.Tensor, has_grad_bypass: bool = None):
        has_grad = torch.is_grad_enabled() if has_grad_bypass is None else has_grad_bypass
        # force enabling grad for normal calculation
        with torch.enable_grad():
            x = x.requires_grad_(True)
            sdf, h = self.forward(x, return_h=True)
            nabla = autograd.grad(
                sdf,
                x,
                torch.ones_like(sdf, device=x.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True
            )[0]
        if not has_grad:
            sdf = sdf.detach()
            nabla = nabla.detach()
            h = h.detach()
        return sdf, nabla, h


class RadianceNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        input_ch_pts = 3
        input_ch_views = 3
        self.skips = cfg.model.radiance.skips
        self.D = cfg.model.radiance.net_depth
        self.W = cfg.model.net_width
        embed_multires = cfg.model.radiance.fr_pos
        embed_multires_view = cfg.model.radiance.fr_view
        self.embed_fn, input_ch_pts = get_embedder(embed_multires)
        self.embed_fn_view, input_ch_views = get_embedder(embed_multires_view)
        self.W_geo_feat = cfg.model.feature_width
        in_dim_0 = input_ch_pts + input_ch_views + 3 + self.W_geo_feat
        
        fc_layers = []
        # NOTE: as in IDR/NeuS, the network's has D+1 layers
        for l in range(self.D + 1):
            # decicde out_dim
            if l == self.D:
                out_dim = 3
            else:
                out_dim = self.W
            
            # decide in_dim
            if l == 0:
                in_dim = in_dim_0
            elif l in self.skips:
                in_dim = in_dim_0 + self.W
            else:
                in_dim = self.W
            
            if l != self.D:
                layer = DenseLayer(in_dim, out_dim, activation=nn.ReLU(inplace=True))
            else:
                layer = DenseLayer(in_dim, out_dim, activation=nn.Sigmoid())
            
            if cfg.model.radiance.weight_norm:
                layer = nn.utils.weight_norm(layer)

            fc_layers.append(layer)

        self.layers = nn.ModuleList(fc_layers)
    
    def forward(
        self, 
        x: torch.Tensor, 
        view_dirs: torch.Tensor, 
        normals: torch.Tensor, 
        geometry_feature: torch.Tensor
    ):
        # calculate radiance field
        x = self.embed_fn(x)
        view_dirs = self.embed_fn_view(view_dirs)
        radiance_input = torch.cat([x, view_dirs, normals, geometry_feature], dim=-1)
        
        h = radiance_input
        for i in range(self.D+1):
            if i in self.skips:
                h = torch.cat([h, radiance_input], dim=-1)
            h = self.layers[i](h)
        return h


class SemanticNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        input_ch_pts = 3
        self.skips = cfg.model.semantic.skips
        self.D = cfg.model.semantic.net_depth
        self.W = cfg.model.net_width
        embed_multires = cfg.model.semantic.fr_pos
        self.embed_fn, input_ch_pts = get_embedder(embed_multires)
        self.W_geo_feat = cfg.model.feature_width
        in_dim_0 = input_ch_pts + self.W_geo_feat
        
        fc_layers = []
        # NOTE: as in IDR/NeuS, the network's has D+1 layers
        for l in range(self.D + 1):
            # decicde out_dim
            if l == self.D:
                out_dim = 3
            else:
                out_dim = self.W
            
            # decide in_dim
            if l == 0:
                in_dim = in_dim_0
            elif l in self.skips:
                in_dim = in_dim_0 + self.W
            else:
                in_dim = self.W
            
            if l != self.D:
                layer = DenseLayer(in_dim, out_dim, activation=nn.ReLU(inplace=True))
            else:
                layer = DenseLayer(in_dim, out_dim, activation=nn.Sigmoid())
            
            if cfg.model.semantic.weight_norm:
                layer = nn.utils.weight_norm(layer)

            fc_layers.append(layer)

        self.layers = nn.ModuleList(fc_layers)
    
    def forward(
        self, 
        x: torch.Tensor, 
        geometry_feature: torch.Tensor):
        # calculate semantic field
        x = self.embed_fn(x)
        semantic_input = torch.cat([x, geometry_feature], dim=-1)
        
        h = semantic_input
        for i in range(self.D+1):
            if i in self.skips:
                h = torch.cat([h, semantic_input], dim=-1)
            h = self.layers[i](h)
        return h


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.speed_factor = cfg.model.speed_factor
        ln_beta_init = np.log(cfg.model.beta_init) / self.speed_factor
        self.ln_beta = nn.Parameter(data=torch.Tensor([ln_beta_init]), requires_grad=True)

        self.sdf_net = SDFNet()
        self.radiance_net = RadianceNet()
        self.semantic_net = SemanticNet()

    def forward_ab(self):
        beta = torch.exp(self.ln_beta * self.speed_factor)
        return 1./beta, beta

    def forward_surface(self, x: torch.Tensor):
        sdf = self.sdf_net.forward(x)
        return sdf        

    def forward_surface_with_nablas(self, x: torch.Tensor):
        sdf, nablas, h = self.sdf_net.forward_with_nablas(x)
        return sdf, nablas, h

    def forward(self, x:torch. Tensor, view_dirs: torch.Tensor):
        sdf, nablas, geometry_feature = self.forward_surface_with_nablas(x)
        radiances = self.radiance_net.forward(x, view_dirs, nablas, geometry_feature)
        semantics = self.semantic_net.forward(x, geometry_feature)
        return radiances, semantics, sdf, nablas
    
    def forward_semantic(self, x:torch. Tensor):
        sdf, nablas, geometry_feature = self.forward_surface_with_nablas(x)
        semantics = self.semantic_net.forward(x, geometry_feature)
        return semantics


def volume_render(
    rays_o, 
    rays_d,
    model: MLP,
    near=0.0,
    far=2.0,
    perturb = True,
    ):

    device = rays_o.device
    rayschunk = cfg.sample.rayschunk
    netchunk = cfg.sample.netchunk
    N_samples = cfg.sample.N_samples
    N_importance = cfg.sample.N_importance
    max_upsample_steps = cfg.sample.max_upsample_steps
    max_bisection_steps = cfg.sample.max_bisection_steps
    epsilon = cfg.sample.epsilon

    DIM_BATCHIFY = 1
    B = rays_d.shape[0]  # batch_size
    flat_vec_shape = [B, -1, 3]

    rays_o = torch.reshape(rays_o, flat_vec_shape).float()
    rays_d = torch.reshape(rays_d, flat_vec_shape).float()

    depth_ratio = rays_d.norm(dim=-1)
    rays_d = F.normalize(rays_d, dim=-1)
    
    batchify_query_ = functools.partial(batchify_query, chunk=netchunk, dim_batchify=DIM_BATCHIFY)

    def render_rayschunk(rays_o: torch.Tensor, rays_d: torch.Tensor):

        view_dirs = rays_d
        
        prefix_batch = [B]
        N_rays = rays_o.shape[-2]
        
        nears = near * torch.ones([*prefix_batch, N_rays, 1]).to(device)
        fars = far * torch.ones([*prefix_batch, N_rays, 1]).to(device)

        _t = torch.linspace(0, 1, N_samples).float().to(device)
        d_coarse = nears * (1 - _t) + fars * _t
        alpha, beta = model.forward_ab()
        with torch.no_grad():
            _t = torch.linspace(0, 1, N_samples*4).float().to(device)
            d_init = nears * (1 - _t) + fars * _t
            
            d_fine, beta_map, iter_usage = fine_sample(
                model.forward_surface, d_init, rays_o, rays_d, 
                alpha_net=alpha, beta_net=beta, far=fars, 
                eps=epsilon, max_iter=max_upsample_steps, max_bisection=max_bisection_steps, 
                final_N_importance=N_importance, perturb=perturb, 
                N_up=N_samples*4
            )

        d_all = torch.cat([d_coarse, d_fine], dim=-1)
        d_all, _ = torch.sort(d_all, dim=-1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * d_all[..., :, None]
        
        radiances, semantics, sdf, nablas = batchify_query_(model.forward, pts, view_dirs.unsqueeze(-2).expand_as(pts))
        sigma = sdf_to_sigma(sdf, alpha, beta)
            
        delta_i = d_all[..., 1:] - d_all[..., :-1]
        p_i = torch.exp(-F.relu_(sigma[..., :-1] * delta_i))

        tau_i = (1 - p_i + 1e-10) * (
            torch.cumprod(
                torch.cat(
                    [torch.ones([*p_i.shape[:-1], 1], device=device), p_i], dim=-1), 
                dim=-1)[..., :-1]
            )

        rgb_map = torch.sum(tau_i[..., None] * radiances[..., :-1, :], dim=-2)
        semantic_map = torch.sum(tau_i[..., None] * semantics[..., :-1, :], dim=-2)
        
        distance_map = torch.sum(tau_i / (tau_i.sum(-1, keepdim=True)+1e-10) * d_all[..., :-1], dim=-1)
        depth_map = distance_map / depth_ratio
        acc_map = torch.sum(tau_i, -1)

        ret_i = OrderedDict([
            ('rgb', rgb_map),
            ('semantic', semantic_map),
            ('distance', distance_map),
            ('depth', depth_map),
            ('mask_volume', acc_map)
        ])

        surface_points = rays_o + rays_d * distance_map[..., None]
        _, surface_normals, _ = model.sdf_net.forward_with_nablas(surface_points.detach())
        ret_i['surface_normals'] = surface_normals

        # normals_map = F.normalize(nablas, dim=-1)
        # N_pts = min(tau_i.shape[-1], normals_map.shape[-2])
        # normals_map = (normals_map[..., :N_pts, :] * tau_i[..., :N_pts, None]).sum(dim=-2)
        # ret_i['normals_volume'] = normals_map

        ret_i['sdf'] = sdf
        ret_i['nablas'] = nablas
        ret_i['radiance'] = radiances
        ret_i['alpha'] = 1.0 - p_i
        ret_i['p_i'] = p_i
        ret_i['visibility_weights'] = tau_i
        ret_i['d_vals'] = d_all
        ret_i['sigma'] = sigma
        ret_i['beta_map'] = beta_map
        ret_i['iter_usage'] = iter_usage

        return ret_i
        
    ret = {}
    for i in range(0, rays_o.shape[DIM_BATCHIFY], rayschunk):
        ret_i = render_rayschunk(rays_o[:, i:i+rayschunk], rays_d[:, i:i+rayschunk])
        for k, v in ret_i.items():
            if k not in ret:
                ret[k] = []
            ret[k].append(v)
    for k, v in ret.items():
        ret[k] = torch.cat(v, DIM_BATCHIFY)
    
    alpha, beta = model.forward_ab()
    alpha, beta = alpha.data, beta.data
    ret['scalars'] = {'alpha': alpha, 'beta': beta}

    return ret


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = MLP()
        
        self.theta = nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        # <cos(theta), sin(tehta), 0> is $\mathbf{n}_w$ in equation (9)
    
    def forward(self, batch):
        rays = batch['rays']
        rays_o, rays_d = rays[:, :, :3], rays[:, :, 3:6]
        rays_d[rays_d.abs() < 1e-6] = 1e-6

        if self.training:
            near = cfg.train_dataset.near
            far = cfg.train_dataset.far
            pertube = True
        else:
            near = cfg.test_dataset.near
            far = cfg.test_dataset.far
            pertube = False

        return volume_render(
            rays_o,
            rays_d,
            self.model,
            near = near,
            far=far,
            perturb=pertube
        )

    
def make_network(cfg):
    module = cfg.network_module
    path = cfg.network_path
    network = imp.load_source(module, path).Network()
    return network