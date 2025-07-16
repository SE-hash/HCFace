import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch import nn
import os
from common.networks import AgingModule, PatchDiscriminator
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from common.sampler import RandomSampler
from common.data_prefetcher import DataPrefetcher
from common.ops import convert_to_ddp, LoggerX
from . import BasicTask
from common.dataset import AgingDataset, TrainImageDataset

class FAS(nn.Module):

    def __init__(self, opt) -> None:
        self.opt = opt
        super().__init__()
        self.logger = LoggerX(save_root='../output')

    def set_loader(self):
        opt = self.opt

        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize([opt.image_size, opt.image_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, ], std=[0.5, ])
            ])
        train_dataset = TrainImageDataset(opt.dataset_name, train_transform)

        weights = None
        sampler = RandomSampler(train_dataset, batch_size=opt.batch_size,
                                num_iter=opt.num_iter, restore_iter=opt.restore_iter, weights=weights)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, sampler=sampler, pin_memory=True,
            num_workers=opt.num_worker, drop_last=True
        )
        self.prefetcher = DataPrefetcher(train_loader)

    def set_model(self):
        opt = self.opt

        generator = AgingModule(age_group=opt.age_group)
        discriminator = PatchDiscriminator(opt.age_group, norm_layer='sn', repeat_num=4)

        d_optim = torch.optim.Adam(discriminator.parameters(), opt.d_lr, betas=(0.5, 0.99))
        g_optim = torch.optim.Adam(generator.parameters(), opt.g_lr, betas=(0.5, 0.99))

        generator, discriminator = convert_to_ddp(generator, discriminator)

        scaler = amp.GradScaler()
        self.generator = generator
        self.discriminator = discriminator
        self.d_optim = d_optim
        self.g_optim = g_optim
        self.scaler = scaler

        self.logger.modules = [generator, discriminator, d_optim, g_optim, scaler]
        # if opt.restore_iter > 0:
        #     self.logger.load_checkpoints(opt.restore_iter)

    def train(self, inputs, n_iter):
        opt = self.opt
        backbone, age_estimation, source_img, target_img, source_label, target_label = inputs
        backbone.eval()
        self.generator.train()
        self.discriminator.train()
        self.d_optim.zero_grad()
        with torch.no_grad():
            with amp.autocast(enabled=opt.amp):
                x_1, x_2, x_3, x_4, x_5, x_id, x_age = backbone(source_img, return_shortcuts=True)
            x_1, x_2, x_3, x_4, x_5, x_id, x_age = \
                x_1.float(), x_2.float(), x_3.float(), x_4.float(), x_5.float(), x_id.float(), x_age.float()
        g_source = self.generator(source_img, x_1, x_2, x_3, x_4, x_5, x_id, x_age, condition=target_label)
        d1_logit = self.discriminator(target_img, target_label)
        d3_logit = self.discriminator(g_source.detach().float(), target_label)
        d_loss = 0.5 * (torch.mean((d1_logit - 1) ** 2) + torch.mean(d3_logit ** 2))
        d_loss.backward()
        self.d_optim.step()
        with amp.autocast(enabled=opt.amp):
            _, g_x_id, g_x_age = backbone(g_source, return_age=True)
        g_x_id, g_x_age = g_x_id.float(), g_x_age.float()
        self.g_optim.zero_grad()
        g_logit = self.discriminator(g_source, target_label)
        g_loss = 0.5 * torch.mean((g_logit - 1) ** 2)
        fas_id_loss = F.mse_loss(x_id, g_x_id)
        fas_age_loss = F.cross_entropy(age_estimation(g_x_age)[1], target_label)
        total_loss = g_loss * opt.fas_gan_loss_weight \
                     + fas_id_loss * opt.fas_id_loss_weight \
                     + fas_age_loss * opt.fas_age_loss_weight
        if opt.amp:
            total_loss = self.scaler.scale(total_loss)
        total_loss.backward()
        if opt.amp:
            self.scaler.step(self.g_optim)
            self.scaler.update()
        else:
            self.g_optim.step()
        lr_g = self.g_optim.param_groups[0]['lr']
        lr_d = self.d_optim.param_groups[0]['lr']
        self.logger.msg([d1_logit, d3_logit, g_logit, fas_id_loss, fas_age_loss, lr_g, lr_d], n_iter)
        self.writer.add_scalar('Train/ID_Loss', fas_id_loss, n_iter)
        self.writer.add_scalar('Train/G_Loss', g_loss, n_iter)
        self.writer.add_scalar('Train/Age_Loss', fas_age_loss, n_iter)
        self.writer.add_scalar('Train/Total_Loss', total_loss, n_iter)

    def validate(self, n_iter, backbone, age_estimation, source_img, source_label):
        opt = self.opt
        self.generator.eval()
        self.discriminator.eval()
        sum_id_loss, sum_age_loss, sum_g_loss = 0, 0, 0
        num_batches = 0
        with torch.no_grad():
            print(len(self.eval_loader))
            for target_img, target_label in self.eval_loader:
                backbone.eval()
                target_img = target_img.cuda()
                target_label = target_label.cuda()
                x_1, x_2, x_3, x_4, x_5, x_id, x_age = backbone(source_img, return_shortcuts=True)

                g_source = self.generator(source_img, x_1, x_2, x_3, x_4, x_5, x_id, x_age, condition=target_label)
                g_logit = self.discriminator(g_source, target_label)
                _, g_x_id, g_x_age = backbone(g_source, return_age=True)

                g_loss = 0.5 * torch.mean((g_logit - 1) ** 2)

                fas_id_loss = F.mse_loss(x_id, g_x_id)
                fas_age_loss = F.cross_entropy(age_estimation(g_x_age)[1], target_label)

                sum_id_loss += fas_id_loss
                sum_g_loss += g_loss
                sum_age_loss += fas_age_loss
                total_loss = sum_g_loss * opt.fas_gan_loss_weight \
                             + sum_id_loss * opt.fas_id_loss_weight \
                             + sum_age_loss * opt.fas_age_loss_weight

                num_batches += 1
                print(
                    'validate:{},fas_id_loss:{},g_loss:{},fas_age_loss:{},total_loss:{}'.format(num_batches,
                                                                                                fas_id_loss, g_loss,
                                                                                                fas_age_loss,
                                                                                                total_loss), n_iter)
            avg_id_loss = sum_id_loss / num_batches
            avg_g_loss = sum_g_loss / num_batches
            avg_age_loss = sum_age_loss / num_batches
            avg_total_loss = total_loss / num_batches

            print(
                f'Average ID Loss: {avg_id_loss:.4f}, Average G Loss: {avg_g_loss:.4f}, Average Age Loss: {avg_age_loss:.4f}')
            lr_g = self.g_optim.param_groups[0]['lr']
            lr_d = self.d_optim.param_groups[0]['lr']
            self.writer.add_scalar('Validation/Avg_ID_Loss', avg_id_loss, n_iter)
            self.writer.add_scalar('Validation/Avg_G_Loss', avg_g_loss, n_iter)
            self.writer.add_scalar('Validation/Avg_Age_Loss', avg_age_loss, n_iter)
            self.writer.add_scalar('Validation/Avg_total_loss', avg_total_loss, n_iter)
            self.logger.msg([avg_id_loss, avg_g_loss, avg_age_loss, lr_g, lr_d], n_iter)
