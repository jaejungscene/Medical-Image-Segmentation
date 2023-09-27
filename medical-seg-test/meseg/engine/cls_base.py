import os

import tqdm
import pandas as pd
import torch
import time, datetime
import wandb
from meseg.utils import compute_metrics, Metric, reduce_mean

from meseg.dataset import btcv_v2_inference
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.data import decollate_batch


@torch.inference_mode()
def test(valid_dataloader, valid_dataset, model, args):
    # 1. create metric
    data_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Data:')
    batch_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Batch:')

    # 2. start validate
    model.eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    total_iter = len(valid_dataloader)
    predictions = list()
    labels = list()
    start_time = time.time()
    args.log(f'start validation of {args.model_name}...')

    for batch_idx, batch in enumerate(valid_dataloader):
        x, y = (batch["image"].to(args.device), batch["label"].to(args.device))

        if args.channels_last:
            x = x.to(memory_format=torch.channels_last)

        data_m.update(time.time() - start_time)

        with torch.cuda.amp.autocast(args.amp):
            y_hat = model(x)

        predictions.append(y_hat)
        labels.append(y)

        if batch_idx and args.print_freq and batch_idx % args.print_freq == 0:
            num_digits = len(str(total_iter))
            args.log(f"{args.mode.upper()}: [{batch_idx:>{num_digits}}/{total_iter}] {batch_m} {data_m}")

        batch_m.update(time.time() - start_time)
        start_time = time.time()

    # 3. move prediction & label to cpu and normalize prediction to probability.
    predictions = torch.concat(predictions, dim=0).detach().float().cpu()
    labels = torch.concat(labels, dim=0).detach().cpu()
    if args.task == 'binary':
        predictions = torch.sigmoid(predictions)
    else:
        predictions = torch.softmax(predictions, dim=-1)

    # 4. save inference result or compute metrics
    if args.mode == 'test' and args.dataset_type in ['isic2018', 'isic2019']:
        metrics = []
        save_path = os.path.join(args.log_dir, f"{args.model_name}.csv")
        df = {"image":valid_dataset.id_list}
        df.update({c: predictions[:, i].tolist() for i, c in enumerate(valid_dataset.classes)})
        pd.DataFrame(df).to_csv(save_path, index=False)
        args.log(f'saved prediction to {save_path}')

    else:
        metrics = [x.detach().float().cpu().item() for x in compute_metrics(predictions, labels, args)]
        space = 12
        num_metric = 1 + len(metrics)
        args.log('-'*space*num_metric)
        args.log(("{:>12}"*num_metric).format('Stage', *args.metric_names))
        args.log('-'*space*num_metric)
        args.log(f"{f'{args.mode.upper()}':>{space}}" + "".join([f"{m:{space}.4f}" for m in metrics]))
        args.log('-'*space*num_metric)

    return predictions, labels, metrics


@torch.inference_mode()
def validate(valid_dataloader, model, criterion, args, epoch, train_loss=None):
    # 1. create metric
    loss_m = Metric(reduce_every_n_step=args.print_freq, header='Loss:')
    data_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Data:')
    batch_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Batch:')

    # 2. start validate
    model.eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    total_iter = len(valid_dataloader)
    predictions = list()
    labels = list()
    start_time = time.time()
    for batch_idx, (x, y) in enumerate(valid_dataloader):
        batch_size = x.size(0)
        x, y = x.to(args.device), y.to(args.device)

        if args.channels_last:
            x = x.to(memory_format=torch.channels_last)

        data_m.update(time.time() - start_time)

        with torch.cuda.amp.autocast(args.amp):
            y_hat = model(x)
            loss = criterion(y_hat, y)

        predictions.append(y_hat)
        labels.append(y)
        loss_m.update(loss, batch_size)

        if batch_idx and args.print_freq and batch_idx % args.print_freq == 0:
            num_digits = len(str(total_iter))
            args.log(f"VALID({epoch:03}): [{batch_idx:>{num_digits}}/{total_iter}] {batch_m} {data_m} {loss_m}")

        batch_m.update(time.time() - start_time)
        start_time = time.time()

    # 3. calculate metric
    loss = loss_m.compute()
    # avg_dice_score = dice_metric.aggregate().item()
    metrics = compute_metrics(predictions, labels, args)
    # metrics = [avg_dice_score]
    metric_dict = {m: v for m, v in zip(args.metric_names, metrics)}
    metric_dict.update({'train_loss': train_loss})

    # 4. print metric
    space = 12
    num_metric = 2+ len(metrics)
    args.log('-'*space*num_metric)
    args.log(("{:>12}"*num_metric).format('Stage', 'Loss', *args.metric_names))
    args.log('-'*space*num_metric)
    args.log(f"{f'VALID({epoch})':>{space}}" + "".join([f"{m:{space}.4f}" for m in [loss]+metrics]))
    args.log('-'*space*num_metric)

    if args.use_wandb:
        args.log(metric_dict, metric=True) # metric=True is wandb log

    if args.save_weight and args.is_rank_zero and args.best < metric_dict[args.save_metric]:
        args.best = metric_dict[args.save_metric]
        torch.save(model.state_dict(), args.best_weight_path)


def train_one_epoch_with_valid(
        train_dataloader, valid_dataloader, model, optimizer, criterion, args,
        scheduler=None, scaler=None, epoch=None, global_step=None
    ):

    # 1. create metric
    data_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Data:')
    batch_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Batch:')
    loss_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Loss:')

    # 2. start validate
    model.train()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    total_iter = len(train_dataloader)
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(train_dataloader):
        batch_size = x.size(0)
        x, y = x.to(args.device), y.to(args.device)

        if args.channels_last:
            x = x.to(memory_format=torch.channels_last)

        data_m.update(time.time() - start_time)

        with torch.cuda.amp.autocast(args.amp):
            y_hat = model(x)
            loss = criterion(y_hat, y)

        if args.distributed:
            loss = reduce_mean(loss, args.world_size)
            
        if args.amp:
            scaler(loss, optimizer, model.parameters(), scheduler, args.grad_norm, batch_idx % args.grad_accum == 0)
        else:
            # loss.backward()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # directly schedule learning rate
            optimizer.param_groups[0]['lr'] = args.lr * (1.0-global_step/(total_iter*args.epoch))**0.9
            if args.grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            # if batch_idx % args.grad_accum == 0:
                # optimizer.step()
                # optimizer.zero_grad()
                # if scheduler:
                #     scheduler.step()
                # print("x.shape:", x.shape)
                # print("learning rate: ", optimizer.param_groups[0]['lr'])
                # print("optim:", optimizer)
                # print("weight decay:", args.weight_decay)
                # print("momentum:", args.momentum)
                # print("criterion:")
                # print("iter_num:", global_step)
                # print("max_iterations:", total_iter*args.epoch)
                # print("base_lr:", args.lr)
                # print("-----------------")
        loss_m.update(loss, batch_size)
        if global_step%args.print_freq==0: # batch_idx%args.print_freq==0     args.print_freq
            num_digits = len(str(total_iter))
            args.log(f"TRAIN(e{epoch:03}|iter{global_step:}): [{batch_idx:>{num_digits}}/{total_iter}] {batch_m} {data_m} {loss_m}")

        batch_m.update(time.time() - start_time)

        if args.valid_freq>0 and batch_idx % args.valid_freq == 0:
            validate(valid_dataloader, model, criterion, args, epoch=epoch, train_loss=loss_m.compute())

        start_time = time.time()
        global_step += 1

    # 3. calculate metric
    duration = str(datetime.timedelta(seconds=batch_m.sum)).split('.')[0]
    data = str(datetime.timedelta(seconds=data_m.sum)).split('.')[0]
    f_b_o = str(datetime.timedelta(seconds=batch_m.sum - data_m.sum)).split('.')[0]
    loss = loss_m.compute()

    # 4. print metric
    space = 12
    num_metric = 4 + 1
    args.log('-'*space*num_metric)
    args.log(("{:>12}"*num_metric).format('Stage', 'Batch', 'Data', 'F+B+O', 'Loss'))
    args.log('-'*space*num_metric)
    args.log(f"{'TRAIN('+str(epoch)+')':>{space}}{duration:>{space}}{data:>{space}}{f_b_o:>{space}}{loss:{space}.4f}")
    args.log('-'*space*num_metric)

    if args.valid_freq==None  or  args.valid_freq==0:
        validate(valid_dataloader, model, criterion, args, epoch=epoch, train_loss=loss)
    elif args.valid_freq<0:
        if epoch+1 == args.epoch:
            args.log("="*100)
            btcv_v2_inference(valid_dataloader, args, model)
            args.log("="*100)
        elif args.use_wandb:
            wandb.log({"avg tarin loss":loss})
        torch.save(model.state_dict(), args.best_weight_path)
    args.log("*"*100)
    return global_step