from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import BatchSampler
import math
import torch
import torch.utils.data
import imp
import os
from torch.utils.data.dataloader import default_collate
import numpy as np
import time
from lib.config import cfg
from torch.utils.data import DataLoader, ConcatDataset
from prefetch_generator import BackgroundGenerator


_collators = {}


def make_collator(cfg, is_train):
    collator = cfg.train.collator if is_train else cfg.test.collator
    if collator in _collators:
        return _collators[collator]
    else:
        return default_collate


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

torch.multiprocessing.set_sharing_strategy('file_system')

def _dataset_factory(is_train, is_val):
    if is_val:
        module = cfg.val_dataset_module
        path = cfg.val_dataset_path
    elif is_train:
        module = cfg.train_dataset_module
        path = cfg.train_dataset_path
    else:
        module = cfg.test_dataset_module
        path = cfg.test_dataset_path
    dataset = imp.load_source(module, path).Dataset
    return dataset


def make_dataset(cfg, is_train=True):
    if is_train:
        args = cfg.train_dataset
        module = cfg.train_dataset_module
        path = cfg.train_dataset_path
    else:
        args = cfg.test_dataset
        module = cfg.test_dataset_module
        path = cfg.test_dataset_path
    dataset = imp.load_source(module, path).Dataset
    dataset = dataset(**args)
    return dataset


def make_data_sampler(dataset, shuffle, is_distributed):
    if is_distributed:
        return DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter, is_train):
    if is_train:
        batch_sampler = cfg.train.batch_sampler
        sampler_meta = cfg.train.sampler_meta
    else:
        batch_sampler = cfg.test.batch_sampler
        sampler_meta = cfg.test.sampler_meta

    if batch_sampler == 'default':
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last)
    elif batch_sampler == 'image_size':
        batch_sampler = ImageSizeBatchSampler(sampler, batch_size, drop_last, sampler_meta)

    if max_iter != -1:
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, max_iter)
    return batch_sampler


def worker_init_fn(worker_id):
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))


def make_data_loader(cfg, is_train=True, is_distributed=False, max_iter=-1):
    if is_train:
        batch_size = cfg.train.batch_size
        shuffle = cfg.train.shuffle
        drop_last = False
    else:
        batch_size = cfg.test.batch_size
        shuffle = True if is_distributed else False
        drop_last = False

    dataset = make_dataset(cfg, is_train)
    if is_train == False and cfg.test.val_dataset != '':
        val_dataset = make_dataset(cfg, cfg.test.val_dataset, is_train, True)
        dataset = ConcatDataset([dataset, val_dataset])
    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(cfg, sampler, batch_size,
                                            drop_last, max_iter, is_train)
    num_workers = cfg.train.num_workers
    collator = make_collator(cfg, is_train)
    data_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collator,
        worker_init_fn=worker_init_fn
    )

    return data_loader

class ImageSizeBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last, sampler_meta):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.strategy = sampler_meta.strategy
        self.hmin, self.wmin = sampler_meta.min_hw
        self.hmax, self.wmax = sampler_meta.max_hw
        self.divisor = 32
        if cfg.fix_random:
            np.random.seed(0)

    def generate_height_width(self):
        if self.strategy == 'origin':
            return -1, -1
        h = np.random.randint(self.hmin, self.hmax + 1)
        w = np.random.randint(self.wmin, self.wmax + 1)
        h = (h | (self.divisor - 1)) + 1
        w = (w | (self.divisor - 1)) + 1
        return h, w

    def __iter__(self):
        batch = []
        h, w = self.generate_height_width()
        for idx in self.sampler:
            batch.append((idx, h, w))
            if len(batch) == self.batch_size:
                h, w = self.generate_height_width()
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.sampler = self.batch_sampler.sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset+self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
