import os
import time
import scipy.stats
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
import torch.nn as nn
from ImageDataset import ImageDataset
from BaseCNN import BaseCNN
from MNL_Loss import FidelityLoss
from Transformers import AdaptiveResize
import utils

class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.config = config
        self.loss_count = 0
        self.train_transform = transforms.Compose([
            AdaptiveResize(512),
            transforms.RandomRotation(3.0),
            transforms.RandomCrop(config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
            transforms.Lambda(lambda x : x + torch.randint_like(x, low=-2, high=3)/255.0),
        ])
        self.test_transform = transforms.Compose([
            AdaptiveResize(768),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
        self.valid_best= 0.0                                 
        # testing set configuration
        self.model = nn.DataParallel(BaseCNN(config, multFlag=True).cuda())
        self.model_name = type(self.model).__name__
        print(self.model)
        # loss function
        self.fid_fn = FidelityLoss().cuda()
        self.initial_lr = config.lr
        if self.initial_lr is None:
            lr = 0.0005
        else:
            lr = self.initial_lr

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            ], lr=lr, weight_decay=5e-4)

        self.train_loader = self._loader(csv_file = self.config.train_file, img_dir = self.config.trainset, \
                                         transform = self.train_transform, batch_size = self.config.batch_size, \
                                         drop_last = True)    
        # some states
        self.start_epoch = 0
        self.start_step = 0
        self.train_loss = []
        self.ckpt_path = config.ckpt_path
        self.ckpt_best_path = config.ckpt_best_path
        self.max_epochs = config.max_epochs
        self.epochs_per_eval = config.epochs_per_eval
        self.epochs_per_save = config.epochs_per_save

        # try load the model
        if config.resume or not config.train:
            ckpt = self._get_latest_checkpoint(path=config.ckpt_path)
            print("[*] ckpt:", ckpt)
            print('[*] start from the breakpoint')
            self._load_checkpoint(ckpt=ckpt)

        self.scheduler = lr_scheduler.StepLR(self.optimizer,
                            last_epoch=self.start_epoch-1,
                            step_size=config.decay_interval,
                            gamma=config.decay_ratio)

    def _loader(self, csv_file, img_dir, transform, test=False, batch_size=16, shuffle=True, pin_memory=True, num_workers=0, drop_last=False):
        data = ImageDataset(csv_file = csv_file,
                        img_dir = img_dir,
                        transform = transform,
                        test = test)
        train_loader = DataLoader(data,
                        batch_size = batch_size,
                        shuffle = shuffle,
                        pin_memory = pin_memory,
                        num_workers = num_workers,
                        drop_last = drop_last)
        return train_loader

    def fit(self):
        for epoch in range(self.start_epoch, self.max_epochs):
            _ = self._train_single_epoch(epoch)
            self.scheduler.step()
            #utils.predict_b(self.config)
        del self.model
        torch.cuda.empty_cache()

    def _train_single_batch(self, model, x1, x2, g=None):
        y1, y1_var = model(x1)
        y2, y2_var = model(x2)
        loss = self.fid_fn(y1, y1_var, y2, y2_var, g)
        return loss
       
    def _train_single_epoch(self, epoch):
        num_steps_per_epoch =  len(self.train_loader) 
        # initialize logging system
        local_counter = epoch * num_steps_per_epoch + 1
        start_time = time.time()
        beta = 0.9
        # training set configuration 
        running_loss = 0 if epoch == 0 else self.train_loss[-1][0]
        running_duration = 0.0

        # start training
        print('Adam learning rate: {:.8f}'.format(self.optimizer.param_groups[0]['lr']))
        self.model.train()
        
        for step, sample_batched in enumerate(self.train_loader, 0):
            if step < self.start_step:
                continue
            x1, x2, g = Variable(sample_batched['I1']).cuda(), Variable(sample_batched['I2']).cuda(),\
                        Variable(sample_batched['y']).view(-1,1).cuda()
            self.optimizer.zero_grad()
            self.loss = self._train_single_batch(self.model, x1=x1, x2=x2, g=g) 
            self.loss.backward()
            self.optimizer.step()
            # statistics
            running_loss = beta * running_loss + (1 - beta) * self.loss.data.item()
            loss_corrected = running_loss / (1 - beta ** local_counter)

            current_time = time.time()
            duration = current_time - start_time
            running_duration = beta * running_duration + (1 - beta) * duration
            duration_corrected = running_duration / (1 - beta ** local_counter)
            examples_per_sec = self.config.batch_size / duration_corrected
            format_str = ('(E:%d, S:%d / %d) [Loss = %.4f] (%.1f samples/sec; %.3f  sec/batch)')
            print(format_str % (epoch, step, num_steps_per_epoch, loss_corrected, examples_per_sec, duration_corrected))
            local_counter += 1
            self.start_step = 0
            start_time = time.time()
            self.train_loss.append([loss_corrected])
        # evaluate after every epoch
        valid_srcc = 0.0
        if epoch>=3:
            srcc, plcc = self._eval(self.model, epoch) # n is the number of heads
            valid_srcc = srcc["valid"]
            tb = utils.print_tb(srcc, plcc)
            print(tb)
            f = open(os.path.join(self.config.result_path, r'results_{}.txt'.format(epoch)), 'w')
            f.write(str(tb))
            f.close()
        if epoch == 2:
            model_name = '{}-{:0>5d}.pt'.format(self.model_name, epoch)
            model_name = os.path.join(self.ckpt_path, model_name)
            self._save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'train_loss': self.train_loss,
            }, os.path.join(self.ckpt_path, 'checkpoint.pt'))
       
        if self.valid_best < valid_srcc and epoch >=3:
            # save best path
            model_name = 'best_{}.pt'.format(self.config.split)
            model_name = os.path.join(self.ckpt_best_path, model_name)
            self._save_checkpoint({
                'state_dict': self.model.state_dict(),
            }, model_name)
            # save best result
            f = open(os.path.join(self.ckpt_best_path, r'best.txt'.format(epoch)), 'w')
            f.write(str(tb))
            f.close()
            # updata valid_best
            self.valid_best = valid_srcc

    def _eval_single(self, model, loader, epoch=None):
        srcc, plcc = {}, {}
        q_mos, q_ens = [], []
        for step, sample_batched in enumerate(loader, 0):
            x, y = Variable(sample_batched['I']).cuda(), sample_batched['mos']
            y_bar, _= model(x)
            for item in y.data.numpy().tolist():
                q_mos.append(item)
            for item in y_bar.cpu().data.numpy().tolist():
                q_ens.append(item)
        srcc = scipy.stats.mstats.spearmanr(x=q_mos, y=q_ens)[0]
        plcc = scipy.stats.mstats.pearsonr(x=q_mos, y=q_ens)[0]
        return srcc, plcc

        
    @torch.no_grad()
    def _eval(self, model, epoch):
        srcc, plcc = {}, {}
        test_model = nn.DataParallel(BaseCNN(self.config, multFlag=False).cuda())
        test_model.load_state_dict(model.state_dict())
        test_model.eval()
        start_time = time.time()
        valid_loader = self._loader(csv_file = self.config.valid_file1, img_dir = self.config.test_path, batch_size=1, 
                                    transform = self.test_transform, test = True, shuffle = False, pin_memory = True, num_workers = 8)
        test_loader = self._loader(csv_file = self.config.test_file1, img_dir = self.config.trainset_path, batch_size=1, 
                                    transform = self.test_transform, test = True, shuffle = False, pin_memory = True, num_workers = 8)
        test1_loader = self._loader(csv_file = self.config.test_file2, img_dir = self.config.test_path, batch_size=1,
                                    transform = self.test_transform, test = True, shuffle = False, pin_memory = True, num_workers = 8)
        srcc['valid'], plcc['valid'] = self._eval_single(test_model, valid_loader)
        srcc['test'], plcc['test'] = self._eval_single(test_model, test_loader)
        srcc['test1'], plcc['test1'] = self._eval_single(test_model, test1_loader)

        print('[*] testing time: {}'.format(time.time()-start_time))
        return srcc, plcc

    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            self.start_epoch = checkpoint['epoch']+1
            self.train_loss = checkpoint['train_loss']
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