import time
import datetime
import torch
import os
import imp
import open3d as o3d
from torch.nn.parallel import DistributedDataParallel
from lib.config import cfg
from lib.data_utils import to_cuda
from lib.mesh_utils import extract_mesh, refuse, transform
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg


class Trainer(object):
    def __init__(self, network):
        print('GPU ID: ', cfg.local_rank)
        device = torch.device('cuda:{}'.format(cfg.local_rank))
        network = network.to(device)
        if cfg.distributed:
            network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(network)
            network = DistributedDataParallel(
                network,
                device_ids=[cfg.local_rank],
                output_device=cfg.local_rank,
                # find_unused_parameters=True
           )
        self.network = network
        self.local_rank = cfg.local_rank
        self.device = device

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):
        for k in batch:
            if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                #batch[k] = [b.cuda() for b in batch[k]]
                batch[k] = [b.to(self.device) for b in batch[k]]
            elif isinstance(batch[k], dict):
                batch[k] = {key: self.to_cuda(batch[k][key]) for key in batch[k]}
            else:
                # batch[k] = batch[k].cuda()
                batch[k] = batch[k].to(self.device)
        return batch
    
    def get_loss_weights(self, epoch):
        loss_weights = dict()

        loss_weights['rgb'] = cfg.loss.rgb_weight

        loss_weights['depth'] = cfg.loss.depth_weight
        for decay_epoch in cfg.loss.depth_weight_decay_epochs:
            if epoch >= decay_epoch:
                loss_weights['depth'] *= cfg.loss.depth_weight_decay
        if epoch >= cfg.loss.depth_loss_clamp_epoch:
            loss_weights['depth_loss_clamp'] = cfg.loss.depth_loss_clamp
        
        loss_weights['joint_start'] = epoch >= cfg.loss.joint_start
        loss_weights['joint'] = cfg.loss.joint_weight

        loss_weights['ce_cls'] = torch.tensor([cfg.loss.non_plane_weight, 1.0, 1.0])
        loss_weights['ce_cls'] = to_cuda(loss_weights['ce_cls'])

        loss_weights['ce'] = cfg.loss.ce_weight
        for decay_epoch in cfg.loss.ce_weight_decay_epochs:
            if epoch >= decay_epoch:
                loss_weights['ce'] *= cfg.loss.ce_weight_decay
        
        loss_weights['eikonal'] = cfg.loss.eikonal_weight

        return loss_weights

    def train(self, epoch, data_loader, optimizer, recorder):
        max_iter = len(data_loader)
        self.network.train()
        end = time.time()

        loss_weights = self.get_loss_weights(epoch)

        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = iteration + 1

            batch = to_cuda(batch, self.device)
            batch['loss_weights'] = loss_weights
            output, loss, loss_stats, image_stats = self.network(batch)

            # training stage: loss; optimizer; scheduler
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
            optimizer.step()

            if cfg.local_rank > 0:
                continue

            # data recording stage: loss_stats, time, image_stats
            recorder.step += 1

            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % cfg.log_interval == 0 or iteration == (max_iter - 1):
                # print training state
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)

                # record loss_stats and image_dict
                recorder.update_image_stats(image_stats)
                recorder.record('train')

    def val(self, epoch, save_mesh=True, evaluate_mesh=False, data_loader=None, evaluator=None, recorder=None):
        self.network.eval()
        torch.cuda.empty_cache()
        mesh = extract_mesh(self.network.net.model.sdf_net)
        if save_mesh and not evaluate_mesh:
            os.makedirs(f'{cfg.result_dir}/', exist_ok=True)
            mesh.export(f'{cfg.result_dir}/{epoch}.obj')
        if evaluate_mesh:
            assert data_loader is not None
            assert evaluator is not None
            mesh = refuse(mesh, data_loader)
            mesh = transform(mesh, cfg.test_dataset.scale, cfg.test_dataset.offset)
            mesh_gt = o3d.io.read_triangle_mesh(f'{cfg.test_dataset.data_root}/{cfg.test_dataset.scene}/gt.obj')
            evaluate_result = evaluator.evaluate(mesh, mesh_gt)
            print(evaluate_result)

            
def _wrapper_factory(cfg, network):
    module = cfg.trainer_module
    path = cfg.trainer_path
    network_wrapper = imp.load_source(module, path).NetworkWrapper(network)
    return network_wrapper


def make_trainer(cfg, network):
    network = _wrapper_factory(cfg, network)
    return Trainer(network)


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()
        self.net = net

    def forward(self, batch):
        output = self.net(batch)
        if not self.net.training:
            return output

        loss_weights = batch['loss_weights']
        loss = 0
        scalar_stats = {}

        rgb_loss = F.l1_loss(batch['rgb'], output['rgb'], reduction='none').mean() # Eq.5
        scalar_stats.update({'rgb_loss': rgb_loss})
        loss += loss_weights['rgb'] * rgb_loss

        depth_colmap_mask = batch['depth_colmap'] > 0
        if depth_colmap_mask.sum() > 0:
            depth_loss = F.l1_loss(output['depth'][depth_colmap_mask], batch['depth_colmap'][depth_colmap_mask], reduction='none') # Eq.7
            if 'depth_loss_clamp' in loss_weights:
                depth_loss = depth_loss.clamp(max=loss_weights['depth_loss_clamp'])
            depth_loss = depth_loss.mean()
            scalar_stats.update({'depth_loss': depth_loss})
            loss += loss_weights['depth'] * depth_loss

        semantic_deeplab = batch['semantic_deeplab']
        wall_mask = semantic_deeplab == 1
        floor_mask = semantic_deeplab == 2
        semantic_score_log = F.log_softmax(output['semantic'], dim=-1)
        semantic_score = torch.exp(semantic_score_log)

        surface_normals = output['surface_normals']
        surface_normals_normalized = F.normalize(surface_normals, dim=-1).clamp(-1., 1.)

        if loss_weights['joint_start']:
            bg_score, wall_score, floor_score = semantic_score.split(dim=-1, split_size=1)
            joint_loss = 0.

            if floor_mask.sum() > 0:
                floor_normals = surface_normals_normalized[floor_mask]
                floor_loss = (1 - floor_normals[..., 2]) # Eq.8
                joint_floor_loss = (floor_score[floor_mask][..., 0] * floor_loss).mean() # Eq.13
                joint_loss += joint_floor_loss
            pi = math.pi
            if wall_mask.sum() > 0:
                wall_normals = surface_normals_normalized[wall_mask]
                wall_loss_vertical = wall_normals[..., 2].abs()
                wall_loss = torch.zeros(wall_loss_vertical.shape).to('cuda:0')
#                 wall_loss = torch.full(wall_loss_vertical.shape, 1e9).to('cuda:0')
                for Theta in range(0, 360, 30):
                    theta = self.net.theta
                    th = torch.Tensor([Theta / (2 * pi)]).to('cuda:0')
                    new_x = wall_normals[..., 0] * torch.cos(th) - wall_normals[..., 1] * torch.sin(th)
                    new_y = wall_normals[..., 0] * torch.sin(th) + wall_normals[..., 1] * torch.cos(th)
                    cos = new_x * torch.cos(theta) + new_y * torch.sin(theta)
                    wall_loss_horizontal = torch.min(cos.abs(), torch.min((1 - cos).abs(), (1 + cos).abs())) # Eq.9
                    wall_loss += wall_loss_vertical + wall_loss_horizontal
                wall_loss /= 12
                joint_wall_loss = (wall_score[wall_mask][..., 0] * wall_loss).mean() # Eq.13
                joint_loss += joint_wall_loss
            
            if floor_mask.sum() > 0 or wall_mask.sum() > 0:
                scalar_stats.update({'joint_loss': joint_loss})
                loss += loss_weights['joint'] * joint_loss
            
        else: # Semantic score is unreliable in early training stage
            geo_loss = 0.

            if floor_mask.sum() > 0:
                floor_normals = surface_normals_normalized[floor_mask]
                floor_loss = (1 - floor_normals[..., 2]).mean()
                geo_loss += floor_loss
            
            if wall_mask.sum() > 0:
                wall_normals = surface_normals_normalized[wall_mask]
                wall_loss_vertical = wall_normals[..., 2].abs().mean()
                geo_loss += wall_loss_vertical

            if floor_mask.sum() > 0 or wall_mask.sum() > 0:
                scalar_stats.update({'geo_loss': geo_loss})
                loss += loss_weights['joint'] * geo_loss

        cross_entropy_loss = F.nll_loss(
            semantic_score_log.reshape(-1, 3),
            semantic_deeplab.reshape(-1).long(),
            weight=loss_weights['ce_cls']
        ) # Eq.14
        scalar_stats.update({'cross_entropy_loss': cross_entropy_loss})
        loss += loss_weights['ce'] * cross_entropy_loss

        nablas: torch.Tensor = output['nablas']
        _, _ind = output['visibility_weights'][..., :nablas.shape[-2]].max(dim=-1)
        nablas = torch.gather(nablas, dim=-2, index=_ind[..., None, None].repeat([*(len(nablas.shape)-1)*[1], 3]))
        eik_bounding_box = cfg.model.bounding_radius
        eikonal_points = torch.empty_like(nablas).uniform_(-eik_bounding_box, eik_bounding_box).to(nablas.device)
        _, nablas_eik, _ = self.net.model.sdf_net.forward_with_nablas(eikonal_points)
        nablas = torch.cat([nablas, nablas_eik], dim=-2)
        nablas_norm = torch.norm(nablas, dim=-1)
        eikonal_loss = F.mse_loss(nablas_norm, nablas_norm.new_ones(nablas_norm.shape), reduction='mean') # Eq.6
        scalar_stats.update({'eikonal_loss': eikonal_loss})
        loss += loss_weights['eikonal'] * eikonal_loss

        scalar_stats.update({'loss': loss})
        scalar_stats['beta'] = output['scalars']['beta']
        scalar_stats['theta'] = self.net.theta.data

        image_stats = {}

        return output, loss, scalar_stats, image_stats
