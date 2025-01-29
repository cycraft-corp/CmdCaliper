import time

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap

from .utils import rank0_print, calculate_param_nums, AverageMeter, save_checkpoint

class Trainer:
    def __init__(
        self, model, optimizer, criterion, training_args, 
        train_dataloader, eval_dataloader=None, lr_scheduler=None
    ):
        self.model = model
        self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.training_args = training_args

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.do_eval = False
        if self.eval_dataloader is not None:
            self.do_eval = True

        self.start_epoch = 0
        
    def evaluation(self, ep: int, it: int):        
        self.model.eval()
        losses = AverageMeter()

        with torch.no_grad():
            torch.cuda.synchronize()
            for i, (batch_data, auxiliary_data) in enumerate(self.eval_dataloader):
                
                torch.cuda.synchronize()

                N = len(batch_data)

                output = self.model(**batch_data)

                torch.cuda.synchronize()
                loss = self.criterion(output, auxiliary_data)
                torch.cuda.synchronize()

                if np.isnan(loss.item()):
                    rank0_print('Hit nan loss. Skip record!')
                else:
                    losses.update(loss.item(), N)

            rank0_print(f'Epoch: {ep + 1}/{self.training_args.epochs}.'
                        f' Iteration: {it + 1}/{len(self.train_dataloader)}.'
                        f' Eval loss: {losses.get_avg()}.')

        self.model.train()
        return losses.get_avg()


    def train(self):
        rank0_print(f'The total param num of the model: {calculate_param_nums(self.model)}')

        self.model = wrap(self.model)
        self.model.train()

        rank0_print('Start to train!!!!!')

        losses = AverageMeter()
        for param_name, param in vars(self.training_args).items():
            rank0_print(f'Param Name -- {param_name}: {param}')


        steps = 0
        best_eval_loss = float("inf")
        self.optimizer.zero_grad()

        for e in range(self.start_epoch, self.training_args.epochs):
            start_time = time.time()
            losses.reset()

            for batch_idx, (batch_data, auxiliary_data) in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()


                N = len(batch_data)
                output = self.model(**batch_data)

                loss = self.criterion(output, auxiliary_data)
                if np.isnan(loss.item()):
                    rank0_print('Hit nan loss. Skip record!')
                else:
                    losses.update(loss.item(), N)

                loss.backward()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                torch.cuda.synchronize()

                steps += 1
                self.optimizer.step()

                if (steps % self.training_args.log_interval == 0) or (batch_idx + 1 == len(self.train_dataloader)):

                    rank0_print(f'Epoch: {e + 1}/{self.training_args.epochs}.'
                                f' LR: {self.optimizer.param_groups[0]["lr"]:.9f}'
                                f' Iteration: {batch_idx + 1}/{len(self.train_dataloader)}.'
                                f' Train loss: {losses.get_avg()}.'
                                f' Time: {time.time() - start_time:.2f}.')

                    torch.cuda.synchronize()

                if (steps % self.training_args.log_interval == 0) or (batch_idx + 1 == len(self.train_dataloader)):
                    start_time = time.time()

                if self.do_eval:
                    eval_loss = None
                    if steps % self.training_args.eval_interval == 0:
                        eval_loss = self.evaluation(e, batch_idx)

                    if eval_loss is not None:
                        torch.cuda.synchronize()

                        if eval_loss < best_eval_loss:
                            rank0_print(f"Achieve new lowest eval loss: {eval_loss} ! Save checkpoint")
                            best_eval_loss = eval_loss
                            torch.cuda.synchronize()

                            save_checkpoint(
                                self.training_args.path_to_checkpoint_dir,
                                self.model, self.optimizer, self.lr_scheduler, e
                            )

