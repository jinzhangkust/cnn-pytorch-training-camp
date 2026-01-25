"""
Author: Dr. Jin Zhang
E-mail: j.zhang.vision@gmail.com
Created on 2022.02.24
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision.utils import save_image

import os
import time
import math
import argparse
from tqdm import tqdm

from dataset import Data4MetricLearn
from models.DEFIE import SSNet50
from util import AverageMeter

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=100, help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=600, help='number of epochs to train (default: 10)')
parser.add_argument('--load-epoch', type=int, default=1, help='number of epochs to train (default: 10)')
parser.add_argument('--learning-rate', type=float, default=1e-4, help='number of epochs to train (default: 10)')
parser.add_argument('--num-workers', type=int, default=4, help='number of epochs to train (default: 10)')
parser.add_argument('--label-smoothing', type=int, default=0.9, help='One-sided label smoothing')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--checkpoint_interval', type=int, default=10, help='interval between model checkpoints')
parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
parser.add_argument('--name', type=str, default='DAE', help='It decides where to store samples and models')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(comment="DAE")


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


full_data = Data4MetricLearn()
train_size = int(0.7 * len(full_data))
val_size = int(0.2 * len(full_data))
test_size = len(full_data) - train_size - val_size
train_data, val_data, test_data = torch.utils.data.random_split(full_data, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(56))
train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.num_workers)
val_loader = DataLoader(val_data, args.batch_size, shuffle=False, num_workers=args.num_workers)
test_loader = DataLoader(test_data, args.batch_size, shuffle=False, num_workers=args.num_workers)


def denormalize4img(x_hat):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
    std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
    x = x_hat * std + mean
    return x


class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        self.encoder = SSNet50()
        self.projector = nn.Sequential(nn.Upsample(scale_factor=2),
                                       nn.ZeroPad2d((1, 0, 1, 0)),
                                       nn.Conv2d(1280, 512, 4, padding=1))

    def forward(self, x):
        x = self.encoder(x)
        x = x.unsqueeze(2).unsqueeze(3)
        x = self.projector(x)
        return x

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
                  nn.InstanceNorm2d(out_size),
                  nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, use_eql=True):
        super(Decoder, self).__init__()
        self.down1 = UNetUp(512, 512, dropout=0.5)
        self.down2 = UNetUp(512, 512, dropout=0.5)
        self.down3 = UNetUp(512, 256, dropout=0.5)
        self.down4 = UNetUp(256, 128, dropout=0.5)
        self.down5 = UNetUp(128, 64)
        self.down6 = UNetUp(64, 32)
        self.down7 = nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False)
        self.adj = nn.Upsample(size=(300, 300))
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        return self.adj(d7)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = ResNetEncoder()
        self.decoder = Decoder()

    def forward(self, x):
        codes = self.encoder(x)
        out = self.decoder(codes)
        return torch.tanh(out)


model = Autoencoder()
if args.load_epoch != 1:
    model.load_state_dict(torch.load('./save/{}/DAE_{}.pth'.format(args.name, args.load_epoch)))
model = model.to(device)
consistency_loss = torch.nn.SmoothL1Loss()  # torch.nn.MSELoss()
consistency_loss = consistency_loss.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)


for epoch in range(args.load_epoch, args.epochs + 1):
    model.train()
    iter_data_time = time.time()
    recon_loss_ave = AverageMeter()
    for ii, (img, _, _) in tqdm(enumerate(train_loader)):
        img = img.to(device)
        mean1 = torch.tensor([0.5561, 0.5706, 0.5491]).unsqueeze(1).unsqueeze(1).cuda()
        std1 = torch.tensor([0.1833, 0.1916, 0.2061]).unsqueeze(1).unsqueeze(1).cuda()
        img = img * std1 + mean1
        mean2 = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(1).cuda()
        std2 = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(1).cuda()
        img = (img - mean2) / std2



        noise = torch.randn_like(img) * 0.1
        # noise = torch.FloatTensor(numpy.random.normal(0, 1, (img.size()))).to(device)
        # print(f"size of img: {img.size()}    size of noise: {noise.size()}")

        recon = model(img + noise)
        # print(f"size of img: {img.size()}    size of recon: {recon.size()}")
        loss = consistency_loss(recon, img)  # 保证real在后，第二个参数需保证requires_grad==False

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        recon_loss_ave.update(loss.cpu().data, img.size(0))

    for _, (img, _, _) in enumerate(val_loader):
        model.eval()
        with torch.no_grad():
            img = img.to(device)
            recon = model(img)
            # print(f"maximum_recon = {torch.max(recon)}    minimum_recon = {torch.min(recon)}")
        break

    grid = torchvision.utils.make_grid(denormalize4img(recon.cpu().data))
    writer.add_image("reconstruction", grid, epoch)
    grid = torchvision.utils.make_grid(img.cpu().data)
    writer.add_image("real", grid, epoch)
    #pic = denormalize4img(img.cpu().data)
    #save_image(pic, './dc_img/{}/{}_reals_image.png'.format(args.name, epoch), nrow=6, normalize=True)

    writer.add_scalar("recon_loss", recon_loss_ave.avg, epoch)

    if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0 and epoch > args.load_epoch:
        if not os.path.exists('./save/{}'.format(args.name)):
            os.mkdir('./save/{}'.format(args.name))
        torch.save(model.state_dict(), './save/{}/DAE_{}.pth'.format(args.name, epoch))
        pic = denormalize4img(img.cpu().data)
        save_image(pic, './dc_img/{}/{}_real_image.png'.format(args.name, epoch), nrow=6, normalize=True)
        pic = denormalize4img(recon.cpu().data)
        save_image(pic, './dc_img/{}/{}_recon_image.png'.format(args.name, epoch), nrow=6, normalize=True)




