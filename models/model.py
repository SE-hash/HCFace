import argparse
import time

import tqdm
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from .fr import FR
from .fas import FAS
from common.ops import load_network
import os
import torch
from torch import nn
import numpy as np

class model(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.fr = FR(opt)
        self.fr.set_loader()
        self.fr.set_model()
        self.fas = FAS(opt)
        self.fas.set_loader()
        self.fas.set_model()

    @staticmethod
    def parser():
        parser = argparse.ArgumentParser()

        parser.add_argument("--train_fr", help='train_fr', action='store_true')
        parser.add_argument("--train_fas", help='train_fas', action='store_true')

        # BACKBONE, HEAD
        parser.add_argument("--backbone_name", help='backbone', type=str,default='ir50')
        parser.add_argument("--head_s", help='s of cosface or arcface', type=float, default=64)
        parser.add_argument("--head_m", help='m of cosface or arcface', type=float, default=0.35)

        # OPTIMIZED
        parser.add_argument("--weight_decay", help='weight-decay', type=float, default=5e-4)
        parser.add_argument("--momentum", help='momentum', type=float, default=0.9)

        # LOSS
        parser.add_argument("--fr_age_loss_weight", help='age loss weight', type=float, default=0.001)
        parser.add_argument("--fr_da_loss_weight", help='cross age domain adaption loss weight', type=float,
                            default=0.002)
        parser.add_argument("--age_group", help='age_group', default=7, type=int)

        # LR
        parser.add_argument("--gamma", help='learning-rate gamma', type=float, default=0.1)
        parser.add_argument("--milestone", help='milestones', type=int, nargs='*', default=[20000, 23000])
        parser.add_argument("--warmup", help='learning rate warmup epoch', type=int, default=1000)
        parser.add_argument("--learning_rate", help='learning-rate', type=float, default=0.1)

        # TRAINING
        parser.add_argument("--dataset_name", "-d", help='dataset name', default='casia', type=str)
        parser.add_argument("--image_size", help='input image size', default=112, type=int)
        parser.add_argument("--num_iter", help='total epochs', type=int, default=36000)
        parser.add_argument("--restore_iter", help='restore_iter', default=0, type=int)
        parser.add_argument("--batch_size", help='batch-size', default=64, type=int)
        parser.add_argument("--val_interval", help='val dataset interval iteration', type=int, default=1000)
        parser.add_argument("--save_interval", help='save model interval iteration', type=int, default=2000)
        parser.add_argument("--save_model_path", help='path to save model', type=str,
                            default='your/path/')

        parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
        parser.add_argument("--num_worker", help='dataloader num-worker', default=32, type=int)
        parser.add_argument("--local_rank", help='local process rank, not need to be set.', default=0, type=int)

        parser.add_argument("--amp", help='amp', action='store_true')
        parser.add_argument("--experiment_name", default=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        # FAS
        parser.add_argument("--d_lr", help='learning-rate', type=float, default=1e-4)
        parser.add_argument("--g_lr", help='learning-rate', type=float, default=1e-4)
        parser.add_argument("--fas_gan_loss_weight", help='gan_loss_weight', type=float)
        parser.add_argument("--fas_id_loss_weight", help='id_loss_weight', type=float)
        parser.add_argument("--fas_age_loss_weight", help='age_loss_weight', type=float)
        parser.add_argument("--id_pretrained_path", help='id_pretrained_path', type=str)
        parser.add_argument("--age_pretrained_path", help='age_pretrained_path', type=str)

        return parser

    def fit(self):
        opt = self.opt

        # training routine
        for n_iter in tqdm.trange(opt.restore_iter + 1, opt.num_iter + 1, disable=(opt.local_rank != 0)):
            # img, label, age, gender
            fr_inputs = self.fr.prefetcher.next()
            if opt.train_fr:
                self.fr.train(fr_inputs, n_iter)
            if opt.train_fas:
                fas_inputs = self.fas.preftcher.next()

                _fas_inputs = [self.fr.backbone.module, self.fr.estimation_network,
                               fr_inputs[0], fas_inputs[0], fr_inputs[1], fas_inputs[1]]
                self.fas.train(_fas_inputs, n_iter)
            if n_iter % opt.val_interval == 0:
                if opt.train_fr:
                    self.fr.validate(n_iter)
                if opt.train_fas:
                    self.fas.validate(n_iter, self.fr.backbone.module, self.fr.estimation_network, fr_inputs[0],
                                      fr_inputs[1])

            if n_iter % opt.save_interval == 0:
                if opt.train_fr:
                    torch.save(self.state_dict(),
                               "your/path/model_{}_iter_{}".format('FR', n_iter))
                if opt.train_fas:
                    torch.save(self.state_dict(),
                               "your/path/model_{}_iter_{}".format('FAS', n_iter))

    def test(self, source_img):
        opt = self.opt
        self.fas.generator.eval()
        self.fas.discriminator.eval()
        backbone = self.fr.backbone
        age_estimation = self.fr.estimation_network
        sum_id_loss, sum_age_loss, sum_g_loss = 0, 0, 0
        num_batches = 0 
        images = []
        with torch.no_grad():
            backbone.eval()

            input_img = Image.open(source_img).convert("RGB")
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize([opt.image_size, opt.image_size]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, ], std=[0.5, ])
                ])
            input_img = transform(input_img).unsqueeze(0)
            images.append(input_img.cuda())
            images = torch.cat(images, dim=0)

            bs = images.size(0)
            target_labels = torch.arange(1).cuda().unsqueeze(1).repeat(bs, 1).flatten()
            repeat_images = images.repeat(1, 7, 1, 1, 1).view(-1, 3, 112, 112)

            x_1, x_2, x_3, x_4, x_5, x_id, x_age = backbone(images, return_shortcuts=True)

            outputs = images
            for i in range(self.opt.age_group):
                target_labels[0] = i
                temp = self.fas.generator(images, x_1, x_2, x_3, x_4, x_5, x_id, x_age, condition=target_labels)
                outputs = torch.cat([outputs, temp], dim=0)
            pil_img = to_pil_image(make_grid(outputs) * 0.5 + 0.5)
            pil_img.save('output_image1.png')
            x_age, x_group = self.fr.estimation_network(x_age)
            print(f'x_age:{x_age}, x_group:{x_group}')
            print(x_age.size())
            print(x_age[0])
