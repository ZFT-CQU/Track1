import numpy as np
import torch
import torch.utils.data
from torchvision.ops import boxes as box_ops
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from tensorboardX import SummaryWriter

summary_writer = SummaryWriter('summaries')


def train_one_epoch(model, ema_model, optimizer, scheduler, data_loader, device, epoch):
    model.train()
    ema_model.eval()

    data_loader = tqdm(data_loader, desc='Train Epoch: [{}]'.format(epoch))

    for i, (images, targets, img_name) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        # sum of all losses
        losses = sum(loss for loss in loss_dict.values())
        # update regular model
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        # updata learning rate
        scheduler.step()
        # update ema_model
        ema_model.update(model)
        # set tqdm
        data_loader.set_postfix(
            sum_losses='%.4f' % losses.item(),
            loss_classifier='%.4f' %
            loss_dict['loss_classifier'].cpu().detach().numpy(),
            loss_box_reg='%.4f' %
            loss_dict['loss_box_reg'].cpu().detach().numpy(),
            loss_objectness='%.4f' %
            loss_dict['loss_objectness'].cpu().detach().numpy(),
            loss_rpn_box_reg='%.4f' %
            loss_dict['loss_rpn_box_reg'].cpu().detach().numpy(),
        )
        #
        summary_writer.add_scalar('train/sum_losses', losses.item(),
                                  (epoch * len(data_loader) + i))
        summary_writer.add_scalar('train/proposal_class_loss',
                                  loss_dict['loss_classifier'].item(),
                                  (epoch * len(data_loader) + i))
        summary_writer.add_scalar('train/proposal_transformer_loss',
                                  loss_dict['loss_box_reg'].item(),
                                  (epoch * len(data_loader) + i))
        summary_writer.add_scalar('train/anchor_objectness_loss',
                                  loss_dict['loss_objectness'].item(),
                                  (epoch * len(data_loader) + i))
        summary_writer.add_scalar('train/anchor_transformer_loss',
                                  loss_dict['loss_rpn_box_reg'].item(),
                                  (epoch * len(data_loader) + i))


def validate_one_epoch(model, data_loader, device, phase, epoch, model_name):
    model.eval()

    data_loader = tqdm(data_loader, desc='Test on ' + phase + ' Set')

    image_scores = []
    image_labels = []

    box_scores = []
    box_labels = []

    totalNumberOfImages = 0.0

    for i, (images, targets, img_name) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        totalNumberOfImages += len(images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # get preidctions
        with torch.no_grad():
            predictions = model(images)
        for idx in range(len(predictions)):
            # Non Max Supress
            keep_idx = box_ops.nms(predictions[idx]['boxes'],
                                   predictions[idx]['scores'],
                                   iou_threshold=0.3)
            pred_scores = predictions[idx]['scores'][keep_idx]
            pred_boxes = predictions[idx]['boxes'][keep_idx]
            gt_boxes = targets[idx]['boxes']
            #
            # image-level predictions
            if len(pred_scores) > 0:
                image_scores.append(np.max(pred_scores.cpu().numpy()))
            else:
                image_scores.append(0)
            # image-level annotations
            if targets[idx]['labels'].any() == 1:
                image_labels.append(1)
            else:
                image_labels.append(0)
            #
            # box-level predictions
            # calculate IOU
            if len(pred_boxes) == 0:
                continue
            elif (len(gt_boxes) == 0) and (len(pred_boxes) > 0):
                # all predicted boxes are false postives
                for score in pred_scores:
                    box_scores.append(score.cpu().numpy())
                    box_labels.append(0.0)
            elif (len(gt_boxes) > 0) and (len(pred_boxes) > 0):
                # calculate IOU
                TP_idx = []
                FP_idx = []
                for pred_idx in range(len(pred_boxes)):
                    pred_box = pred_boxes[pred_idx]
                    iou = np.zeros(len(gt_boxes))
                    for gt_idx in range(len(gt_boxes)):
                        gt_box = gt_boxes[gt_idx]
                        iou[gt_idx] = intersection_over_union(pred_box, gt_box)
                    # IOU > 0.2 is true positive
                    if iou.any() > 0.2:
                        TP_idx.append(pred_idx)
                    # IOU <= 0.2 is false positive
                    else:
                        FP_idx.append(pred_idx)
                    # retain the maximum score TP

                # deal with the overleaping boxes
                tmp = []
                for gt_box in targets[idx]['boxes']:
                    idx_max = None
                    score_max = 0
                    for i in TP_idx:
                        iou = intersection_over_union(gt_box, pred_boxes[i])
                        if (iou > 0.2) and (pred_scores[i].cpu().numpy() >
                                            score_max):
                            idx_max = i
                            score_max = pred_scores[i].cpu().numpy()
                    if idx_max is not None:
                        tmp.append(idx_max)
                for i in FP_idx:
                    box_scores.append(pred_scores[i].cpu().numpy())
                    box_labels.append(0.0)
                TP_idx = np.unique(tmp)
                for i in TP_idx:
                    box_scores.append(pred_scores[i].cpu().numpy())
                    box_labels.append(1.0)
    # calculate AUC
    auc = roc_auc_score(image_labels, image_scores)
    # calculate sensitivity
    numberOfDetectedLesions = sum(box_labels)
    totalNumberOfCandidates = len(box_scores)
    print('number of images: ', totalNumberOfImages)
    print('number of detected lesions: ', numberOfDetectedLesions)
    print('number of predicted boxes: ', totalNumberOfCandidates)
    sens_125 = 0.0
    sens_25 = 0.0
    sens_5 = 0.0
    if totalNumberOfCandidates != 0:
        fpr, tpr, thresholds = roc_curve(box_labels, box_scores, pos_label=1)
        # FROC
        fps = fpr * (totalNumberOfCandidates -
                     numberOfDetectedLesions) / totalNumberOfImages
        sens = tpr
        if fps.max() >= 0.125:
            for i in range(len(fps)):
                if fps[i] >= 0.125:
                    sens_125 = sens[i]
                    break
        if fps.max() >= 0.25:
            for i in range(len(fps)):
                if fps[i] >= 0.25:
                    sens_25 = sens[i]
                    break
        if fps.max() >= 0.5:
            for i in range(len(fps)):
                if fps[i] >= 0.5:
                    sens_5 = sens[i]
                    break
    print(
        "AUC:{:.4f}\t Sens125:{:.4f}\t Sens25:{:.4f}\t Sens5:{:.4f}\t Ranking_metric:{:.4f}\t"
        .format(auc, sens_125, sens_25, sens_5, (0.75 * auc + 0.25 * sens_25)))

    summary_writer.add_scalar(model_name+'_'+phase + '/AUC', auc, epoch)
    summary_writer.add_scalar(model_name+'_'+phase + '/Sens125', sens_125, epoch)
    summary_writer.add_scalar(model_name+'_'+phase + '/Sens25', sens_25, epoch)
    summary_writer.add_scalar(model_name+'_'+phase + '/Sens5', sens_5, epoch)
    summary_writer.add_scalar(model_name+'_'+phase + '/Ranking_metric',
                              (0.75 * auc + 0.25 * sens_25), epoch)
    return auc


def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou
