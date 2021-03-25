import torch
import sys
import os
import random
import argparse
import apex
import numpy as np
import torch.backends.cudnn as cudnn
from models.inception_resnet_v1 import InceptionResnetV1
from models.mtcnn import MTCNN, fixed_image_standardization
from models.utils import training
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from apex import amp


def parse_opts():
    parser = argparse.ArgumentParser(description='facenet')
    parser.add_argument('--seed', type=int, default=123456, help='random seed')
    parser.add_argument('--amp_cfg', action='store_true', help='If true, use apex.')
    parser.add_argument('--opt_level', default='O0', type=str, help='set opt level.')
    parser.add_argument('--loss_scale_value', default=1024, type=int, help='set loss scale value.')
    parser.add_argument('--device_list', default='0,1,2,3,4,5,6,7', type=str, help='device id list')
    parser.add_argument('--batch_size', default=64, type=int, help='set batch_size')
    parser.add_argument('--epochs', default=20, type=int, help='set epochs')
    parser.add_argument('--workers', default=0, type=int, help='set workers')
    parser.add_argument('--data_dir', default="", type=str, help='set data_dir')
    args = parser.parse_args()
    return args


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_opts()
    seed_everything(args.seed)
    device_id = int(args.device_list.split(",")[0])
    device = 'npu:{}'.format(device_id)
    print('Running on device: {}'.format(device))
    torch.npu.set_device(device)
    dataset = datasets.ImageFolder(args.data_dir, transform=None)

    resnet = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=len(dataset.class_to_idx)
    ).to(device)

    optimizer = apex.optimizers.NpuFusedAdam(resnet.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer, [5, 10])
    if args.amp_cfg:
        resnet, optimizer = amp.initialize(resnet, optimizer, opt_level=args.opt_level,
                                           loss_scale=args.loss_scale_value)

    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    dataset = datasets.ImageFolder(args.data_dir, transform=trans)
    img_inds = np.arange(len(dataset))

    np.random.shuffle(img_inds)
    train_inds = img_inds[:int(0.8 * len(img_inds))]
    val_inds = img_inds[int(0.8 * len(img_inds)):]

    train_loader = DataLoader(
        dataset,
        num_workers=args.workers,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(train_inds)
    )
    val_loader = DataLoader(
        dataset,
        num_workers=args.workers,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(val_inds)
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = {
        'fps': training.BatchTimer(),
        'acc': training.accuracy
    }

    writer = SummaryWriter()
    writer.iteration, writer.interval = 0, 10

    print('\n\nInitial')
    print('-' * 10)

    resnet.eval()
    training.pass_epoch(
        args.amp_cfg, resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    for epoch in range(args.epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, args.epochs))
        print('-' * 10)

        resnet.train()
        training.pass_epoch(
            args.amp_cfg, resnet, loss_fn, train_loader, optimizer, scheduler,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

        resnet.eval()
        training.pass_epoch(
            args.amp_cfg, resnet, loss_fn, val_loader,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

    writer.close()


if __name__ == "__main__":
    main()












