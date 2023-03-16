import os
import time
import scipy.stats
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
import torch.nn as nn
from ImageDataset import ImageDataset
from BaseCNN import BaseCNN
from mask import Mask
from MNL_Loss import Fidelity_Loss
from Transformers import AdaptiveResize
from tensorboardX import SummaryWriter
import prettytable as pt
import heapq
from shutil import copyfile


class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)
        self.config = config
        self.loss_count = 0
        self.train_transform = transforms.Compose([
            #transforms.RandomRotation(3),
            AdaptiveResize(512),
            transforms.RandomCrop(config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

        self.test_transform = transforms.Compose([
            AdaptiveResize(768),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

        self.writer = SummaryWriter(self.config.runs_path)
        self.model = nn.DataParallel(BaseCNN(config).cuda())
        self.model_name = type(self.model).__name__
        print(self.model)

        # loss function
        self.loss_fn = Fidelity_Loss().cuda()
        self.loss_mse = torch.nn.MSELoss().cuda()

        self.initial_lr = config.lr
        if self.initial_lr is None:
            lr = 0.0005
        else:
            lr = self.initial_lr

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr,
            weight_decay=5e-4)

        # some states
        self.start_epoch = 0
        self.start_step = 0
        self.train_loss = []
        self.train_q_loss = []
        self.train_std_loss = []
        self.test_results_srcc = {'live': [], 'csiq': [], 'tid': [], 'kadid': []}
        self.test_results_plcc = {'live': [], 'csiq': [], 'tid': [], 'kadid': []}
        self.ckpt_path = config.ckpt_path
        self.max_epochs = config.max_epochs
        self.epochs_per_eval = config.epochs_per_eval
        self.epochs_per_save = config.epochs_per_save

        # try load the model
        if config.resume or not config.train:
            ckpt = self._get_latest_checkpoint(path=config.ckpt_path)
            print('**********************************************************************************')
            print("ckpt:", ckpt)
            print('start from the pretrained model of Save Model')
            print('**********************************************************************************')
            self._load_checkpoint(ckpt=ckpt)

        self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                             last_epoch=self.start_epoch-1,
                                             step_size=config.decay_interval,
                                             gamma=config.decay_ratio)
        
    def fit(self):
        config = self.config
        self.train_batch_size = config.batch_size
        csv_file = os.path.join(config.trainset, config.kadid_LT, str(config.split), config.train_txt)
        self.train_data = ImageDataset(csv_file=csv_file,
                                   img_dir=config.trainset,
                                   transform=self.train_transform,
                                   test=False)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=self.train_batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=8)
        self.mask_model = Mask(self.model)
        self.mask_model.magnitudePruning(self.config.ratio)
        for epoch in range(self.start_epoch, self.max_epochs):
            _ = self._train_single_epoch(epoch)
            self.scheduler.step()
            
    def _train_single_epoch(self, epoch):
        # initialize logging system
        num_steps_per_epoch = len(self.train_loader)
        local_counter = epoch * num_steps_per_epoch + 1
        start_time = time.time()
        beta = 0.9
        for name, para in self.model.named_parameters():
            print('{} parameters requires_grad:{}'.format(name, para.requires_grad))

        running_loss = 0 if epoch == 0 else self.train_loss[-1]
        running_std_loss = 0 if epoch == 0 else self.train_std_loss[-1]
        running_q_loss = 0 if epoch == 0 else self.train_q_loss[-1]

        loss_corrected = 0.0
        std_loss_corrected = 0.0
        q_loss_corrected = 0.0
        running_duration = 0.0

        # start training
        print('Adam learning rate: {:.8f}'.format(self.optimizer.param_groups[0]['lr']))
        self.model.train()
        #self.scheduler.step()
        for step, sample_batched in enumerate(self.train_loader, 0):
            if step < self.start_step:
                continue
            x1, x2, g= sample_batched['I1'], sample_batched['I2'], sample_batched['y']
            x3, x4= sample_batched['I3'], sample_batched['I4']
            x1, x2 = Variable(x1).cuda(), Variable(x2).cuda()
            x3, x4 = Variable(x3).cuda(), Variable(x4).cuda()
            g = Variable(g).view(-1, 1).cuda()
            
            self.optimizer.zero_grad()
           
            self.model.module.backbone.set_prune_flag(False)
            y1, y1_var = self.model(x1)
            y2, y2_var = self.model(x2)
            y_diff = y1 - y2
            y_var = y1_var + y2_var + 1e-8
            p = 0.5 * (1 + torch.erf(y_diff / torch.sqrt(2 * y_var.detach())))
            self.q_loss = self.loss_fn(p, g.detach())

            self.model.module.backbone.set_prune_flag(True)

            y1_p, y1_var_p = self.model(x3)
            y2_p, y2_var_p = self.model(x4)
            y_diff_p = y1_p - y2_p
            y_var_p = y1_var_p + y2_var_p + 1e-8
            p_p = 0.5 * (1 + torch.erf(y_diff_p / torch.sqrt(2 * y_var_p.detach())))
            self.q_loss += self.loss_fn(p_p, g.detach())

            self.std_loss = self.loss_mse(y1, y1_p) + self.loss_mse(y2,y2_p)
            self.loss = self.q_loss + self.std_loss
            self.loss.backward()
            self.optimizer.step()

            # statistics
            running_loss = beta * running_loss + (1 - beta) * self.loss.data.item()
            loss_corrected = running_loss / (1 - beta ** local_counter)

            running_q_loss = beta * running_q_loss + (1 - beta) * self.q_loss.data.item()
            q_loss_corrected = running_q_loss / (1 - beta ** local_counter)

            running_std_loss = beta * running_std_loss + (1 - beta) * self.std_loss.data.item()
            std_loss_corrected = running_std_loss / (1 - beta ** local_counter)
           
            self.loss_count += 1
            if self.loss_count % 100 == 0:
                self.writer.add_scalars('data/Corrected_Loss', {'loss': loss_corrected}, self.loss_count)
                self.writer.add_scalars('data/Uncorrected_Loss', {'loss': self.loss.data.item()}, self.loss_count)
            
            current_time = time.time()
            duration = current_time - start_time
            running_duration = beta * running_duration + (1 - beta) * duration
            duration_corrected = running_duration / (1 - beta ** local_counter)
            examples_per_sec = self.train_batch_size / duration_corrected
            format_str = ('(E:%d, S:%d / %d) [Loss = %.4f Q Loss = %.4f Consistency Loss = %.4f] (%.1f samples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (epoch, step, num_steps_per_epoch, loss_corrected, q_loss_corrected, std_loss_corrected,
                                examples_per_sec, duration_corrected))

            local_counter += 1
            self.start_step = 0
            start_time = time.time()

        self.train_loss.append(loss_corrected)
        self.train_std_loss.append(std_loss_corrected)
        self.train_q_loss.append(q_loss_corrected)

        # evaluate after every other epoch
        srcc, plcc = self.eval()
        self.test_results_srcc['live'].append(srcc['live'])
        self.test_results_srcc['csiq'].append(srcc['csiq'])
        self.test_results_srcc['tid'].append(srcc['tid'])
        self.test_results_srcc['kadid'].append(srcc['kadid'])
       

        self.test_results_plcc['live'].append(plcc['live'])
        self.test_results_plcc['csiq'].append(plcc['csiq'])
        self.test_results_plcc['tid'].append(plcc['tid'])
        self.test_results_plcc['kadid'].append(plcc['kadid'])
        

        tb = pt.PrettyTable()
        tb.field_names = ["SRCC", "LIVE", "CSIQ", "TID2013", "KADID"]
        tb.add_row(['SRCC', srcc['live'], srcc['csiq'], srcc['tid'], srcc['kadid']])
        tb.add_row(['PLCC', plcc['live'], plcc['csiq'], plcc['tid'], plcc['kadid']])
        print(tb)
        f = open(os.path.join(self.config.result_path, r'results_{}.txt'.format(epoch)), 'w')
        f.write(str(tb))
        f.close()

        if (epoch+1) % self.epochs_per_save == 0:
            model_name = '{}-{:0>5d}.pt'.format(self.model_name, epoch)
            model_name = os.path.join(self.ckpt_path, model_name)
            self._save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'train_loss': self.train_loss,
                'train_std_loss': self.train_std_loss,
                'train_q_loss': self.train_q_loss,
                'test_results_srcc': self.test_results_srcc,
                'test_results_plcc': self.test_results_plcc,
            }, model_name)

        return self.loss.data.item()

    def _eval_single(self, loader):
        q_mos = []
        q_hat = []
        for step, sample_batched in enumerate(loader, 0):
            x, y = sample_batched['I'], sample_batched['mos']
            x = Variable(x)
            x = x.cuda()
            y_bar, _ = self.model(x)
            y_bar.cpu()
            q_mos.append(y.data.numpy())
            q_hat.append(y_bar.cpu().data.numpy())
        srcc = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
        plcc = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat)[0]
        return srcc, plcc

    def _testloader(self, csv_file, img_dir, bs=1):
        data = ImageDataset(csv_file = csv_file,
                           img_dir = img_dir,
                           transform=self.test_transform,
                           test=True)
        loader = DataLoader(data,
                           batch_size=bs,
                           shuffle=False,
                           pin_memory=True,
                           num_workers=8)
        return loader

    def eval(self):
        srcc = {}
        plcc = {}
        self.model.eval()
        self.model.module.backbone.set_prune_flag(False)
        config = self.config
        # testing set configuration
        live_test_loader = self._testloader(csv_file = os.path.join(config.live_set, 'splits2', str(config.split), 'live_test.txt'),
            img_dir = config.live_set)
        csiq_test_loader = self._testloader(csv_file = os.path.join(config.csiq_set, 'splits2', str(config.split), 'csiq_test.txt'),
            img_dir = config.csiq_set)
        tid_test_loader = self._testloader(csv_file = os.path.join(config.tid_set, 'splits2', str(config.split), 'tid_test.txt'),
            img_dir = config.tid_set)
        kadid_test_loader = self._testloader(csv_file = os.path.join(config.trainset, config.kadid_LT, str(config.split), 'kadid10k_test.txt'),
            img_dir = config.kadid_set)
        
        srcc['live'], plcc['live'] = self._eval_single(live_test_loader)
        srcc['csiq'], plcc['csiq'] = self._eval_single(csiq_test_loader)
        srcc['tid'], plcc['tid'] = self._eval_single(tid_test_loader)
        srcc['kadid'], plcc['kadid'] = self._eval_single(kadid_test_loader)

        return srcc, plcc

    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            self.start_epoch = checkpoint['epoch']+1
            self.train_loss = checkpoint['train_loss']
            self.train_std_loss = checkpoint['train_std_loss']
            self.train_q_loss = checkpoint['train_q_loss']
            self.test_results_srcc = checkpoint['test_results_srcc']
            self.test_results_plcc = checkpoint['test_results_plcc']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.initial_lr is not None:
                for param_group in self.optimizer.param_groups:
                    param_group['initial_lr'] = self.initial_lr
            print("[*] loaded checkpoint '{}' (epoch {})"
                  .format(ckpt, checkpoint['epoch']))
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt))

    @staticmethod
    def _get_latest_checkpoint(path):
        ckpts = os.listdir(path)
        ckpts = [ckpt for ckpt in ckpts if not os.path.isdir(os.path.join(path, ckpt))]
        all_times = sorted(ckpts, reverse=True)
        return os.path.join(path, all_times[0])

    # save checkpoint
    @staticmethod
    def _save_checkpoint(state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)

