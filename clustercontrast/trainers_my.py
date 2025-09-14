from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
from torch.cuda.amp import autocast, GradScaler

class ClusterContrastTrainer(object):
    def __init__(self, encoder, memory_all=None, memory_split=None):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory_all = memory_all
        self.memory_split = memory_split
        self.scaler = GradScaler()

    def train(self, epoch, data_loader, optimizer, lr_scheduler, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            optimizer.zero_grad()

            # process inputs
            inputs, labels = self._parse_data(inputs)

            # forward
            with autocast():
                x_all, x_splits = self._forward(inputs, epoch)
                loss_all = self.memory_all(x_all, labels)

                loss_split = 0

                if len(self.memory_split) == self.encoder.module.num_parts:
                    for j in range(self.encoder.module.num_parts):
                        loss_split += self.memory_split[j](x_splits[:,j,:], labels)

                loss = loss_all + loss_split / self.encoder.module.num_parts

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f'Epoch: [{epoch}][{i + 1}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                    f'LR {current_lr:.2e}'
                )

        lr_scheduler.step()

    def _parse_data(self, inputs):
        img, pid, cid, name = inputs
        return img.cuda(), pid.cuda()

    def _forward(self, inputs, epoch):
        return self.encoder(inputs, epoch)

