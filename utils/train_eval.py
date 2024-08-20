#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
import torch
from torch.cuda.amp import GradScaler, autocast
from .metrics import MultilabelMetrics


LOG_STEP = 100


def training(model, data_loader, optimizer, criterion, beta, writer, iteration, loss_sum, loss_sum_class, loss_sum_info):
    _use_amp = True
    scaler = GradScaler(enabled=_use_amp)

    model.train()
    for image, label in data_loader:
        iteration += 1
        image, label = image.cuda(), label.to(torch.float32).cuda()
        model.zero_grad()

        with autocast(enabled=_use_amp):
            output, mu, sigma, _ = model(image)
            loss_class = criterion(output, label)
            loss_info  = - 0.5 * (1 + 2*sigma.log() - mu.pow(2) - sigma.pow(2)).sum(1).mean()
            loss = loss_class + beta * loss_info

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item()
        loss_sum_class += loss_class.item()
        loss_sum_info += loss_info.item()

        if iteration < LOG_STEP:
            print("iteration: %06d loss: %0.6f (class: %0.6f, info: %0.6f)" % (iteration, loss.data, loss_class.data, loss_info.data))


        if iteration % LOG_STEP == 0:
            print("iteration: %06d loss: %0.6f (class: %0.6f, info: %0.6f)" % (iteration, loss_sum / LOG_STEP, loss_sum_class / LOG_STEP, loss_sum_info / LOG_STEP))
            writer.add_scalar("00_loss/loss", loss_sum / LOG_STEP, iteration)
            writer.add_scalar("00_loss/class", loss_sum_class / LOG_STEP, iteration)
            writer.add_scalar("00_loss/info", loss_sum_info / LOG_STEP, iteration)
            loss_sum, loss_sum_class, loss_sum_info = 0.0, 0.0, 0.0

    return iteration, loss_sum, loss_sum_class, loss_sum_info


def evaluation(model, data_loader, attribute_names, n_sampling=0, writer=None, epoch=None, result_dir_path=None):
    metrics = MultilabelMetrics(num_attributes=len(attribute_names), attr_name_list=attribute_names)

    model.eval()
    with torch.no_grad():
        for image, label in data_loader:
            image = image.cuda()
            if n_sampling < 1:
                output, _, _, _ = model(image, deterministic=True)
            else:
                _output_stack = []
                for _ in range(n_sampling):
                    _out_tmp, _, _, _ = model(image)
                    _output_stack.append(_out_tmp)
                output = torch.mean(torch.stack(_output_stack), dim=0)

            # binarize output
            label = label.data
            metrics.stack(true_label=label, pred_score=torch.sigmoid(output))

    ### compute & print accuracy
    score = metrics.get_score()
    print("\nScore ------------------")
    print("    mean Precision:", score['precision']['mean'])
    print("    mean Recall   :", score['recall']['mean'])
    print("    mean F1-score :", score['f1-score']['mean'])
    print("    mean AP       :", score['average_precision']['mean'], "\n")

    ### write TensorBoard
    if writer is not None and epoch is not None:
        _keys = list(score['precision'].keys())
        for k in _keys:
            if _keys == 'mean': continue
            writer.add_scalar('02_avg_precision/%s' % str(k), score['average_precision'][k], epoch)
            writer.add_scalar('03_f1-score/%s' % str(k),      score['f1-score'][k], epoch)
            writer.add_scalar('05_precision/%s' % str(k),     score['precision'][k], epoch)
            writer.add_scalar('06_recall/%s' % str(k),        score['recall'][k], epoch)

        writer.add_scalar('01_mean_score/precision',     score['precision']['mean'], epoch)
        writer.add_scalar('01_mean_score/recall',        score['recall']['mean'], epoch)
        writer.add_scalar('01_mean_score/f1-score',      score['f1-score']['mean'], epoch)
        writer.add_scalar('01_mean_score/avg_precision', score['average_precision']['mean'], epoch)

    ### save as json
    if result_dir_path is not None:
        with open(os.path.join(result_dir_path, "score.json"), 'w') as f:
            json.dump(score, f, indent=4)

    return score
