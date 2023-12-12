from lib.config import cfg, args
from lib import make_network, make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler, make_evaluator, make_data_loader
from lib.net_utils import load_model, save_model, load_network, load_pretrain
import torch.multiprocessing
import torch
import torch.distributed as dist
import os
torch.autograd.set_detect_anomaly(True)

if cfg.fix_random:
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(cfg, network):
    trainer = make_trainer(cfg, network)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)

    begin_epoch = load_model(
        network,
        optimizer,
        scheduler,
        recorder,
        cfg.trained_model_dir,
        resume=cfg.resume
    )
    if begin_epoch == 0 and cfg.pretrain != '':
        load_pretrain(network, cfg.pretrain)
        
    set_lr_scheduler(cfg, scheduler)

    train_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=cfg.distributed,
        max_iter=cfg.ep_iter
    )
    test_loader = make_data_loader(cfg, is_train=False)
    
    if begin_epoch < cfg.train.epoch:
        epoch = begin_epoch
        recorder.epoch = epoch
        if cfg.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)

        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()

        if cfg.local_rank == 0:
            save_model(network, optimizer, scheduler, recorder, cfg.trained_model_dir, epoch)

        if cfg.local_rank == 0:
            save_model(
                network,
                optimizer,
                scheduler,
                recorder,
                cfg.trained_model_dir,
                epoch,
                last=True
            )

        if cfg.local_rank == 0:
            trainer.val(epoch, test_loader)
            
    return network


def main():
    network = make_network(cfg)
    train(cfg, network)


if __name__ == "__main__":
    main()