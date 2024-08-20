#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
sys.path.append("../")

import os
import re
import random

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard.writer import SummaryWriter

### import from multilabel module
from datasets.voc import load_voc_dataset, VOC_NUM_CLASSES, VOC_ATTRIBUTE_NAMES
from models import ProbFeatureEmbedModel
from losses import load_loss_function
from utils import load_checkpoint, save_checkpoint, save_args
from utils.train_eval import training, evaluation
from utils.learning_rate import set_learning_rate

### import from same directory
from parse import argument_parser


CHECKPOINT_STEP = 10
RESULT_DIR_TRAIN = "result_%s_train"
RESULT_DIR_VAL   = "result_%s_val"
RESULT_DIR_TEST  = "result_%s_test"


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def main():
    args = argument_parser()

    ### GPU (device) settings -----------------------------
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    print("use cuda:", use_cuda)
    assert use_cuda, "This training script needs CUDA (GPU)."

    ### set random seed -----------------------------------
    fix_random_seed(args.seed)

    ### dataset -------------------------------------------
    print("load dataset")
    _, _, train_loader, val_loader = load_voc_dataset(args.data_root, '2012', args.batch_size, args.num_workers)

    ### network model -------------------------------------
    model = ProbFeatureEmbedModel(
        model_name=args.model,
        num_classes=VOC_NUM_CLASSES,
        embed_dim=args.embed_dim,
        pretrained=args.pretrained
    )

    ### optimizer -----------------------------------------
    _param_group = set_learning_rate(model, args.lr, backbone_lr=args.b_lr)
    optimizer = torch.optim.SGD(_param_group, momentum=args.momentum, weight_decay=args.wd, nesterov=args.use_nesterov)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    ### loss function -------------------------------------
    criterion = load_loss_function(args.loss, args=args)

    ### CPU or GPU (Data Parallel) ------------------------
    model = nn.DataParallel(model).cuda()
    criterion = criterion.cuda()
    cudnn.benchmark = True

    # -------------------------------------------------------------------------
    # MODE: evaluation
    # -------------------------------------------------------------------------
    # resume ----------------------------------------------
    if args.resume is not None:
        print("Load checkpoint for resuming a training ...")
        print("    checkpoint:", os.path.join(args.logdir, args.resume))
        model, optimizer, scheduler, _, _, _ = load_checkpoint(os.path.join(args.logdir, args.resume), model, optimizer, scheduler)
        resume_postfix = re.match(r'checkpoint-([a-zA-Z0-9_]+)\.pt', args.resume).group(1)

    if args.mode == 'test':
        print("mode: evaluation\n")

        ### validation data ---------------------
        print("evaluate test data ...")
        evaluation(
            model=model, data_loader=val_loader,
            attribute_names=VOC_ATTRIBUTE_NAMES, n_sampling=args.n_sampling, writer=None,
            result_dir_path=os.path.join(args.logdir, RESULT_DIR_VAL % resume_postfix)
        )

        print("evaluation; done.\n")
        exit(0)

    # -------------------------------------------------------------------------
    # MODE: training
    # -------------------------------------------------------------------------
    # tensorboard -----------------------------------------
    writer = SummaryWriter(log_dir=args.logdir)
    log_dir = writer.file_writer.get_logdir()
    save_args(os.path.join(log_dir, 'args.json'), args)

    ### logging variables ---------------------------------
    initial_epoch, iteration = 1, 0
    best_score = 0.0
    loss_sum, loss_sum_class, loss_sum_info = 0.0, 0.0, 0.0

    for epoch in range(initial_epoch, args.epochs + 1):

        ### train
        print("epoch:", epoch)
        iteration, loss_sum, loss_sum_class, loss_sum_info = training(model, train_loader, optimizer, criterion, args.beta, writer, iteration, loss_sum, loss_sum_class, loss_sum_info)

        ### validation
        print("evaluation ...")
        score = evaluation(
            model, val_loader, VOC_ATTRIBUTE_NAMES, n_sampling=0,
            writer=writer, epoch=epoch, result_dir_path=None
        )

         ### update learning rate
        scheduler.step()

        ### save model
        print("save model ...")
        ### 1. best validation accuracy
        if best_score < score['average_precision']['mean']:
            best_score = score['average_precision']['mean']
            save_checkpoint(os.path.join(log_dir, "checkpoint-best.pt"), model, optimizer, scheduler, best_score, epoch, iteration)

        ### 2. at regular intervals
        if epoch % CHECKPOINT_STEP == 0:
            save_checkpoint(os.path.join(log_dir, "checkpoint-%04d.pt" % epoch), model, optimizer, scheduler, best_score, epoch, iteration)

        print("epoch:", epoch, "; done.\n")

    ### save final model & close tensorboard writer -------
    print("save final model")
    save_checkpoint(os.path.join(log_dir, "checkpoint-final.pt"), model, optimizer, scheduler, best_score, epoch, iteration)
    writer.close()
    print("training; done.")


if __name__ == '__main__':
    main()
