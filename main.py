import os
import numpy as np
import torch
import torchvision
import argparse

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# SimCLR
from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet, Custom_InfoNCE
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model

################ CUSTOM IMPORTS ################
from model import load_optimizer, save_model
from utils import yaml_config_hook
import sklearn
from sklearn import neighbors, cluster
import umap
import matplotlib.pyplot as plt   
################################################

################ CUSTOM IMPORTS ################
def initialize(
    loader: torch.utils.data.DataLoader,
    encoder: torch.nn.Sequential,
    projection: torch.nn.Sequential,
    classes: int = None,
    normalize: bool =False
):

    encoder.fc = torch.nn.Identity()
    it = iter(loader)
    samples = []
    length = 10 if len(it) > 10 else len(it)
    for _ in range(length):
        (x, _), _ = next(it)
        samples.append(torch.squeeze(x))
    samples = torch.cat(samples, dim=0).to("cpu")

    if isinstance(encoder,torch.nn.Sequential): encoder = encoder.cpu()
    else: encoder = encoder.to("cpu")
    
    embeddings = encoder.forward(samples)
    embeddings = projection.forward(embeddings)
    if normalize: embeddings = embeddings/torch.sum(torch.pow(embeddings,2),dim=-1).unsqueeze(-1).repeat([1,embeddings.shape[-1]])
    #max_value = torch.max(torch.abs(embeddings))

    #kernel=neighbors.KernelDensity(metric="l1",kernel="gaussian").fit(embeddings.detach().cpu().numpy())
    #init_clusters = kernel.sample(classes)
    init_clusters = sklearn.cluster.KMeans(classes).fit(embeddings.detach().cpu().numpy()).cluster_centers_
    return torch.transpose(torch.tensor(init_clusters), 1, 0).unsqueeze(0)
################################################



def train(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    for step, ((x_i, x_j), labels) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)

        ################ CUSTOM IMPORTS ################
        p_y_z_i, p_y_z_j, h_i, h_j, z_i, z_j = model(x_i, x_j)
        ################################################

        if args.custom: loss = criterion(p_y_z_i,p_y_z_j)
        else: loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if args.nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        if args.nr == 0:
            writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
            args.global_step += 1

        loss_epoch += loss.item()
    return loss_epoch


def main(gpu, args):
    rank = args.nr * args.gpus + gpu

    if args.nodes > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="unlabeled",
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
    else:
        raise NotImplementedError

    if args.nodes > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )

    # initialize ResNet
    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer
    projector = nn.Sequential(nn.Linear(n_features, n_features, bias=False),nn.ReLU(),nn.Linear(n_features, args.projection_dim, bias=False),)
    if args.custom: init_clusters=initialize(train_loader,encoder,projector,args.classes,args.normalize)
    # initialize model
    model = SimCLR(encoder, projector, n_features, args.custom, init_clusters if args.custom else None,args.classes if args.custom else None,learn_std=args.learn_std)
    
    if args.reload:
        model_fp = os.path.join(
            args.model_path, "checkpoint_{}.tar".format(args.epoch_num)
        )
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    model = model.to(args.device)

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)
    if args.custom: criterion = Custom_InfoNCE(args.batch_size)
    else:criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)

    # DDP / DP
    if args.dataparallel:
        model = convert_model(model)
        model = DataParallel(model)
    else:
        if args.nodes > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[gpu])

    model = model.to(args.device)

    writer = None
    if args.nr == 0:
        writer = SummaryWriter("/cluster/home/abizeul/test")

    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.learn_std and args.std_epochs >= epoch:
            for param in model.parameters():
                param.requires_grad = True
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(args, train_loader, model, criterion, optimizer, writer)

        if args.nr == 0 and scheduler:
            scheduler.step()

        if args.nr == 0 and epoch % args.save_every == 0:
            save_model(args, model, optimizer)

        if args.nr == 0:
            writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            print(
                f"-------------------> Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
            )
            args.current_epoch += 1

    ## end training
    save_model(args, model, optimizer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    print(config)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    print(args)

    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    if args.nodes > 1:
        print(
            f"Training with {args.nodes} nodes, waiting until all nodes join before starting training"
        )
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    else:
        main(0, args)
