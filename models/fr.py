import time

import torch.distributed as dist
import torch
from torch.utils.data import random_split, SubsetRandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import torch.cuda.amp as amp
import os
from common.sampler import RandomSampler
from common.data_prefetcher import DataPrefetcher
from common.ops import convert_to_ddp, get_dex_age, age2group, apply_weight_decay, reduce_loss, LoggerX
from common.grl import GradientReverseLayer
from . import BasicTask
from torch import nn
from backbone.aifr import backbone_dict, AgeEstimationModule
from head.cosface import CosFace
from common.dataset import TrainImageDataset, EvaluationImageDataset, BaseImageDataset, TestImageDataset

'''
    face recognition module
'''


class FR(nn.Module):
    def __init__(self, opt):
        self.opt = opt
        super().__init__()
        self.logger = LoggerX(save_root='your/path')

    def set_loader(self):
        opt = self.opt

        if not os.path.exists(f'your/path/{opt.experiment_name}'):
            os.mkdir(f'your/path/{opt.experiment_name}')
        self.writer = SummaryWriter(f'your/path/{opt.experiment_name}')
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize([opt.image_size, opt.image_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, ], std=[0.5, ])
            ])
        dataset = BaseImageDataset(opt.dataset_name, transform)
        dataset_size = len(dataset)
        train_split_point = int(dataset_size * 0.9)
        eval_split_point = int(dataset_size * 0.99)

        indices = list(range(dataset_size))
        np.random.shuffle(indices)

        train_indices = indices[:train_split_point]
        eval_indices = indices[train_split_point:eval_split_point]
        test_indices = indices[eval_split_point:]

        eval_sampler = SubsetRandomSampler(eval_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_dataset = TrainImageDataset(opt.dataset_name, transform)
        eval_dataset = EvaluationImageDataset(opt.dataset_name, transform)
        test_dataset = TestImageDataset(opt.dataset_name, transform)

        train_sampler = RandomSampler(train_dataset, batch_size=opt.batch_size,
                                      num_iter=opt.num_iter, restore_iter=opt.restore_iter)

        self.train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=opt.batch_size,
                                       pin_memory=True, num_workers=opt.num_worker, drop_last=True, shuffle=False)
        self.eval_loader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=opt.batch_size,
                                      pin_memory=True, num_workers=opt.num_worker, drop_last=False, shuffle=False)
        self.test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=opt.batch_size,
                                      pin_memory=True, num_workers=opt.num_worker, drop_last=False, shuffle=False)
        self.prefetcher = DataPrefetcher(self.train_loader)

    def set_model(self):
        opt = self.opt
        backbone = backbone_dict[opt.backbone_name](input_size=opt.image_size)
        head = CosFace(in_features=512, out_features=len(self.prefetcher.__loader__.dataset.classes),
                       s=opt.head_s, m=opt.head_m)

        estimation_network = AgeEstimationModule(input_size=opt.image_size, age_group=opt.age_group)

        da_discriminator = AgeEstimationModule(input_size=opt.image_size, age_group=opt.age_group)

        optimizer = torch.optim.SGD(list(backbone.parameters()) + \
                                    list(head.parameters()) + \
                                    list(estimation_network.parameters()) + \
                                    list(da_discriminator.parameters()),
                                    lr=opt.learning_rate, momentum=opt.momentum)

        backbone, head, estimation_network, da_discriminator = convert_to_ddp(backbone, head, estimation_network,
                                                                              da_discriminator)
        scaler = amp.GradScaler()
        self.optimizer = optimizer
        self.backbone = backbone
        self.head = head
        self.estimation_network = estimation_network
        self.da_discriminator = da_discriminator
        self.grl = GradientReverseLayer()
        self.scaler = scaler

        self.logger.modules = [optimizer, backbone, head, estimation_network, da_discriminator, scaler]
        # if opt.restore_iter > 0:
        #     self.logger.load_checkpoints(opt.restore_iter)

    def adjust_learning_rate(self, step):
        assert step > 0, 'batch index should large than 0'
        opt = self.opt
        if step > opt.warmup:
            lr = opt.learning_rate * (opt.gamma ** np.sum(np.array(opt.milestone) < step))
        else:
            lr = step * opt.learning_rate / opt.warmup
        lr = max(1e-4, lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def compute_age_loss(self, x_age, x_group, ages):
        opt = self.opt
        age_loss = F.mse_loss(get_dex_age(x_age), ages) + \
                   F.cross_entropy(x_group, age2group(ages, age_group=opt.age_group).long())
        return age_loss

    def forward_da(self, x_id, ages):
        x_age, x_group = self.da_discriminator(self.grl(x_id))
        loss = self.compute_age_loss(x_age, x_group, ages)
        return loss

    def train(self, inputs, n_iter):
        opt = self.opt

        images, labels, ages, genders = inputs
        self.backbone.train()
        self.head.train()
        self.da_discriminator.train()
        self.estimation_network.train()

        if opt.amp:
            with amp.autocast():
                embedding, x_id, x_age = self.backbone(images, return_age=True)
            embedding = embedding.float()
            x_id = x_id.float()
            x_age = x_age.float()
        else:
            embedding, x_id, x_age = self.backbone(images, return_age=True)

        ######## Train Face Recognition
        id_loss = F.cross_entropy(self.head(embedding, labels), labels)
        x_age, x_group = self.estimation_network(x_age)
        age_loss = self.compute_age_loss(x_age, x_group, ages)
        da_loss = self.forward_da(x_id, ages)
        loss = id_loss + \
               age_loss * opt.fr_age_loss_weight + \
               da_loss * opt.fr_da_loss_weight

        total_loss = loss
        if opt.amp:
            total_loss = self.scaler.scale(loss)
        self.optimizer.zero_grad()
        total_loss.backward()
        apply_weight_decay(self.backbone, self.head, self.estimation_network,
                           weight_decay_factor=opt.weight_decay, wo_bn=True)
        # self.adjust_learning_rate(n_iter)
        if opt.amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        id_loss, da_loss, age_loss = reduce_loss(id_loss, da_loss, age_loss)
        loss = id_loss + \
               age_loss * opt.fr_age_loss_weight + \
               da_loss * opt.fr_da_loss_weight
        lr = self.optimizer.param_groups[0]['lr']
        self.logger.msg([id_loss, da_loss, age_loss, lr], n_iter)
        self.writer.add_scalar('Train/ID_Loss', id_loss, n_iter)
        self.writer.add_scalar('Train/DA_Loss', da_loss, n_iter)
        self.writer.add_scalar('Train/Age_Loss', age_loss, n_iter)
        self.writer.add_scalar('Train/Total_Loss', loss, n_iter)

    def validate(self, n_iter):
        opt = self.opt
        self.backbone.eval()
        self.head.eval()
        self.da_discriminator.eval()
        self.estimation_network.eval()


        sum_id_loss, sum_da_loss, sum_age_loss = 0, 0, 0
        num_batches = 0 
        with torch.no_grad():
            for images, labels, ages in self.eval_loader:
                images = images.cuda()
                labels = labels.cuda()
                ages = ages.cuda()
                if opt.amp:
                    with amp.autocast():
                        embedding, x_id, x_age = self.backbone(images, return_age=True)
                    embedding = embedding.float()
                    x_id = x_id.float()
                    x_age = x_age.float()
                else:
                    embedding, x_id, x_age = self.backbone(images, return_age=True)
                id_loss = F.cross_entropy(self.head(embedding, labels), labels)
                x_age, x_group = self.estimation_network(x_age)
                age_loss = self.compute_age_loss(x_age, x_group, ages)
                da_loss = self.forward_da(x_id, ages)
                id_loss, da_loss, age_loss = reduce_loss(id_loss, da_loss, age_loss)
                lr = self.optimizer.param_groups[0]['lr']

                sum_id_loss += id_loss
                sum_da_loss += da_loss
                sum_age_loss += age_loss

                num_batches += 1

                # if num_batches % 10 ==0:
                # self.logger.msg([id_loss, da_loss, age_loss, lr], n_iter)

            avg_id_loss = sum_id_loss / num_batches
            avg_da_loss = sum_da_loss / num_batches
            avg_age_loss = sum_age_loss / num_batches
            print(
                f'Average ID Loss: {avg_id_loss:.4f}, Average DA Loss: {avg_da_loss:.4f}, Average Age Loss: {avg_age_loss:.4f}')

            self.writer.add_scalar('Validation/Avg_ID_Loss', avg_id_loss, n_iter)
            self.writer.add_scalar('Validation/Avg_DA_Loss', avg_da_loss, n_iter)
            self.writer.add_scalar('Validation/Avg_Age_Loss', avg_age_loss, n_iter)
            self.logger.msg([avg_id_loss, avg_da_loss, avg_age_loss, lr], n_iter)

    def test(self):
        opt = self.opt
        with torch.no_grad():
            for images, ages, labels in self.test_loader:
                images = images.cuda()
                ages = ages.cuda()
                labels = labels.cuda()
                if opt.amp:
                    with amp.autocast():
                        embedding, x_id, x_age = self.backbone(images, return_age=True)
                    x_age = x_age.float()
                    embedding = embedding.float()
                    x_id = x_id.float()
                else:
                    embedding, x_id, x_age = self.backbone(images, return_age=True)
                x_age = self.estimation_network(x_age)
                id_loss = F.cross_entropy(self.head(embedding, labels), labels)

                predicted_age = np.argmax(x_age[0].cpu(), axis=1)
                predicted_group = np.argmax(x_age[1].cpu(), axis=1)
                print(predicted_age == ages.cpu())
                print(np.mean(predicted_age.cpu().numpy() == ages.cpu().numpy()))

                group = [0, 11, 21, 31, 41, 51, 61, np.inf]
                age_group = np.digitize(ages.cpu(), group, right=True) - 1
                print(age_group == predicted_group.numpy())
                print(np.mean(age_group == predicted_group.numpy()))

    def get_age(self, imgs):
        for img in imgs:
            img.cuda()
            embedding, x_id, x_age = self.backbone(img, return_age=True)
            x_age, x_group = self.estimation_network(x_age)
            print(f'x_age:{x_age}, x_group:{x_group}')

    def save_model(self, save_path, n_iter):
        """
        Saves the model's parameters.

        Args:
            save_path (str): The directory path where the model parameters will be saved.
            n_iter (int): The current iteration, used to name the saved file.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        save_dict = {
            'backbone': self.backbone.state_dict(),
            'head': self.head.state_dict(),
            'estimation_network': self.estimation_network.state_dict(),
            'da_discriminator': self.da_discriminator.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict() if self.opt.amp else None
        }

        save_file_path = os.path.join(save_path, f'model_iter_{n_iter}.pth')
        torch.save(save_dict, save_file_path)
        print(f'Model saved successfully at {save_file_path}')

    def load_model(self, load_path, load_name=None):
        if load_name is None:
            checkpoint_path = self._find_latest_checkpoint(load_path)
        else:
            checkpoint_path = os.path.join(load_path, load_name)
        print(checkpoint_path)
        if checkpoint_path and os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.backbone.load_state_dict(checkpoint['backbone'])
            self.head.load_state_dict(checkpoint['head'])
            self.estimation_network.load_state_dict(checkpoint['estimation_network'])
            self.da_discriminator.load_state_dict(checkpoint['da_discriminator'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.opt.amp and 'scaler' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler'])
            print(f'Model loaded successfully from {checkpoint_path}')
        else:
            print('No saved models found at the specified path.')

    def _find_latest_checkpoint(self, directory):
        checkpoints = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pth')]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            return latest_checkpoint
        return None
