import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from utils.iotools import save_checkpoint
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
from loss.softmax_loss import CrossEntropyLabelSmooth
from model.cm import ClusterMemory
from utils.test_video_reid import test, _eval_format_logger

def do_train_stage2(cfg,
                    model,
                    center_criterion,
                    train_loader_stage1,
                    train_loader_stage2,
                    # val_loader,
                    query_loader,
                    gallery_loader,
                    optimizer,
                    optimizer_center,
                    scheduler,
                    loss_fn,
                    num_query,
                    local_rank, num_classes):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("TFCLIP.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    loss_meter_cm = AverageMeter()
    acc_meter = AverageMeter()
    acc_meter_id1 = AverageMeter()
    acc_meter_id2 = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    # scaler = amp.GradScaler()
    xent_frame = CrossEntropyLabelSmooth(num_classes=num_classes)
    criterion_shape_mse = nn.MSELoss()
    shape_mean_10 = x = torch.tensor([0.2056, 0.3356, -0.3507, 0.3561, 0.4175, 0.0309, 0.3048, 0.2361, 0.2091, 0.3121],
                                     dtype=torch.float32, device=local_rank)

    @torch.no_grad()
    def generate_cluster_features(labels, features, num_classes):
        import collections
        # features is a tensor of shape [N, feature_dim], labels is a numpy array of shape [N]
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if label == -1:
                continue
            if 0 <= labels[i] < num_classes:  # Ensure label is in valid range
                centers[labels[i]].append(features[i])  # features[i] is a tensor slice

        # Compute global mean for fallback initialization (for classes not in labels_list)
        if features.shape[0] > 0:
            global_mean = features.mean(0)  # features is already a tensor, just compute mean
        else:
            # Should not happen, but use zero vector as fallback
            global_mean = torch.zeros(features.shape[1], device=features.device, dtype=features.dtype)

        # Initialize all class centers (for classes not in labels_list, use global mean)
        all_centers = []
        for idx in range(num_classes):
            if idx in centers and len(centers[idx]) > 0:
                # Compute mean for classes that appear in labels_list
                all_centers.append(torch.stack(centers[idx], dim=0).mean(0))
            else:
                # For classes not in labels_list, use global mean (will be normalized later)
                all_centers.append(global_mean.clone())

        centers = torch.stack(all_centers, dim=0)
        return centers

    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    #######   1.CLIP-Memory module ####################

    best_performance = 0.0
    best_epoch = 1
    best_rank_1 = 0.0
    best_mAP =0.0
    for epoch in range(1, epochs + 1):
        print(f"=> Automatically generating CLIP-Memory Epoch{epoch}(might take a while, have a coffe)")
        image_features = []
        labels = []
        with torch.no_grad():
            # for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage1):
            for n_iter, batch_data in enumerate(train_loader_stage1):
                # Handle both original (3 values) and extended (8 values) dataset formats
                if len(batch_data) == 3:
                    img, vid, target_cam = batch_data
                    altitude = None
                    distance = None
                    angle = None
                else:
                    # Extended format: (img, vid, target_cam, altitude, distance, angle, aerial_distance, point_id)
                    img, vid, target_cam, altitude, distance, angle, aerial_distance, point_id = batch_data

                img = img.to(device)  # torch.Size([64, 4, 3, 256, 128])
                # img = img.permute(0, 2, 1, 3, 4)  # B, C, T, H, W -> B, T, C, H, W
                target = vid.to(device)  # torch.Size([64])
                target_view = None
                if len(img.size()) == 6:
                    # method = 'dense'
                    b, n, s, c, h, w = img.size()
                    assert (b == 1)
                    img = img.view(b * n, s, c, h, w)  # torch.Size([5, 8, 3, 256, 128])
                    # with amp.autocast(enabled=True):
                    image_feature = model(img, get_image=True)
                    image_feature = image_feature.view(-1, image_feature.size(1))
                    image_feature = torch.mean(image_feature, 0, keepdim=True)  # 1,512
                    for i, img_feat in zip(target, image_feature):
                        labels.append(i)
                        image_features.append(img_feat.cpu())
                else:
                    # with amp.autocast(enabled=True):
                    image_feature = model(img, get_image=True)
                    for i, img_feat in zip(target, image_feature):
                        labels.append(i)
                        image_features.append(img_feat.cpu())

            labels_list = torch.stack(labels, dim=0).cuda()  # N torch.Size([8256])
            image_features_list = torch.stack(image_features, dim=0).cuda()  # torch.Size([8256, 512])

        cluster_features = generate_cluster_features(labels_list.cpu().numpy(), image_features_list, num_classes)

        memory = ClusterMemory(1280, num_classes, temp=0.05, momentum=0.1, use_hard=False).cuda()
        memory.features = F.normalize(cluster_features.repeat(2, 1), dim=1).cuda()  # torch.Size([num_classes*2, 1280])

        # free temporary tensors used to build the memory bank to save GPU memory
        del labels_list, image_features_list, labels, image_features, cluster_features
        torch.cuda.empty_cache()

        #######   2.Mamba module ####################
        start_time = time.time()
        loss_meter.reset()
        loss_meter_cm.reset()
        acc_meter.reset()
        acc_meter_id1.reset()
        acc_meter_id2.reset()
        evaluator.reset()

        model.train()
        # for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage2):
        # for n_iter in range(500):
        #     img, vid, target_cam, target_view = train_loader_stage2.next()
        for n_iter, batch_data in enumerate(train_loader_stage2):
            # Handle both original (3 values) and extended (8 values) dataset formats
            if len(batch_data) == 3:
                img, vid, target_cam = batch_data
                altitude = None
                distance = None
                angle = None
            else:
                # Extended format: (img, vid, target_cam, altitude, distance, angle, aerial_distance, point_id)
                img, vid, target_cam, altitude, distance, angle, aerial_distance, point_id = batch_data

            optimizer.zero_grad()
            # optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else:
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            # with amp.autocast(enabled=True):
            # img = img.permute(0, 2, 1, 3, 4) # B, C, T, H, W -> B, T, C, H, W
            B, C, T, H, W = img.shape  # B=64, T=4.C=3 H=256,W=128
            score, feat, logits1, tsm = model(x = img, cam_label=target_cam, view_label=target_view)
            score1 = [score[i] for i in [0, 1, 3]]
            score2 = score[2]
            loss1 = loss_fn(score1, feat, target, target_cam, isprint=False)


            targetX = target.unsqueeze(1)  # 12,1   => [94 94 10 10 15 15 16 16 75 75 39 39]
            targetX = targetX.expand(B, T)
            # 12,8  => [ [94...94][94...94][10...10][10...10] ... [39...39] [39...39]]
            targetX = targetX.contiguous()
            targetX = targetX.view(B * T, -1)  # 96  => [94...94 10...10 15...15 16...16 75...75 39...39]
            targetX = targetX.squeeze(1)

            # # Ensure score2 has the correct shape [B*T, num_classes]
            # if score2.dim() == 3 and score2.shape[0] == B and score2.shape[1] == T:
            #     # Reshape from [B, T, num_classes] to [B*T, num_classes]
            #     score2 = score2.view(B * T, -1)
            # elif score2.shape[0] != B * T:
            #     # If shape doesn't match, try to reshape
            #     score2 = score2.view(B * T, -1)

            # # Validate targetX values are in valid range [0, num_classes-1]
            # if targetX.min() < 0 or targetX.max() >= num_classes:
            #     invalid_mask = (targetX < 0) | (targetX >= num_classes)
            #     if invalid_mask.any():
            #         logger.warning(f"Found {invalid_mask.sum().item()} invalid target values in targetX: min={targetX.min().item()}, max={targetX.max().item()}, num_classes={num_classes}")
            #         # Clamp invalid values to valid range
            #         targetX = targetX.clamp(0, num_classes - 1)

            loss_frame = xent_frame(score2, targetX)


            loss_p = memory(logits1, target)

            shape_pids = target.repeat_interleave(T)
            shape_id_loss = xent_frame(tsm[2].view(B*T, -1), shape_pids)
            # shape_mse = criterion_shape_mse(tsm[1], tsm[0])
            shape_mean_10s = shape_mean_10.view(1, 1, 10).expand(B, T, 10)
            shape_mse = criterion_shape_mse(tsm[0], shape_mean_10s)
            loss_tsm = 1.0 * shape_id_loss / T + 0.5 * shape_mse #1 8

            loss = loss1 + loss_p + loss_frame / T + loss_tsm


            # scaler.scale(loss).backward()
            # loss.backward()
            try:
                loss.backward()
            except:
                torch.save({
                    "model_state": model.state_dict(),
                    "img": img.detach().cpu(),
                    "target_cam": target_cam.detach().cpu(),
                    "target_view": target_view.detach().cpu(),
                }, "save_debug.pth")
                # import ipdb; ipdb.set_trace()
            optimizer.step()
            # scaler.step(optimizer)
            # scaler.update()


            acc1 = (logits1.max(1)[1] == target).float().mean()
            acc_id1 = (score[0].max(1)[1] == target).float().mean()
            acc_id2 = (score[1].max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            loss_meter_cm.update(loss_p.item(), img.shape[0])
            acc_meter.update(acc1, 1)
            acc_meter_id1.update(acc_id1, 1)
            acc_meter_id2.update(acc_id2, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                msg = "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Loss_p: {:.3f}, Acc_clip: {:.3f}, Acc_id1: {:.3f}, Acc_id2: {:.3f}, Base Lr: {:.2e}".format(
                    epoch, (n_iter + 1), len(train_loader_stage2), loss_meter.avg, loss_meter_cm.avg, acc_meter.avg,
                    acc_meter_id1.avg, acc_meter_id2.avg, scheduler.get_lr()[0])
                logger.info(msg)

        scheduler.step()

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            msg = "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(epoch, time_per_batch, B / time_per_batch)
            logger.info(msg)

        # save_epochs = [30,32,34,36,38,40,42,44,46,48,50]
        # if epoch in save_epochs:
        if epoch % checkpoint_period == 0:  # or epoch == 1
            msg = "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(epoch, time_per_batch, B / time_per_batch)
            logger.info(msg)

            ### save model dict
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

            ### Evaluation disabled during training (anonymized test set)
                # model.eval()
            from evaluate_all_cases import train_epoch_evaluate
            train_epoch_evaluate(model, cfg, epoch)

        # # Evaluation disabled during training (anonymized test set)
        # # Uncomment below to enable evaluation with proper ground truth
        # if epoch % eval_period == 0:
        #     if cfg.MODEL.DIST_TRAIN:
        #         assert 1 < 0
        #     else:
        #         use_gpu = True
        #         cmc, mAP, ranks = test(model, query_loader, gallery_loader, use_gpu, cfg)
        #         ptr = "mAP: {:.2%}".format(mAP)
        #         if cmc[0] > best_rank_1:
        #             best_rank_1 = cmc[0]
        #             torch.save(model.state_dict(),os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_rank1_best.pth'))
        #         if mAP > best_mAP:
        #             best_mAP = mAP
        #             torch.save(model.state_dict(),os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_mAP_best.pth'))
        #         for r in ranks:
        #             ptr += " | R-{:<3}: {:.2%}".format(r, cmc[r - 1])
        #         logger.info(ptr)

        # if epoch % eval_period == 0 and epoch >= 50:
        #     if cfg.MODEL.DIST_TRAIN:
        #         if dist.get_rank() == 0:
        #             model.eval()
        #             for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
        #                 with torch.no_grad():
        #                     img = img.to(device)
        #                     if cfg.MODEL.SIE_CAMERA:
        #                         camids = camids.to(device)
        #                     else:
        #                         camids = None
        #                     if cfg.MODEL.SIE_VIEW:
        #                         target_view = target_view.to(device)
        #                     else:
        #                         target_view = None
        #                     feat = model(img, cam_label=camids, view_label=target_view)
        #                     evaluator.update((feat, vid, camid))
        #             cmc, mAP, _, _, _, _, _ = evaluator.compute()
        #             logger.info("Validation Results - Epoch: {}".format(epoch))
        #             logger.info("mAP: {:.1%}".format(mAP))
        #             for r in [1, 5, 10, 20]:
        #                 logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        #             torch.cuda.empty_cache()
        #     else:
        #         model.eval()
        #         for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
        #             with torch.no_grad():
        #                 img = img.to(device)
        #                 if cfg.MODEL.SIE_CAMERA:
        #                     camids = camids.to(device)
        #                 else:
        #                     camids = None
        #                 if cfg.MODEL.SIE_VIEW:
        #                     target_view = target_view.to(device)
        #                 else:
        #                     target_view = None
        #                 feat = model(img, cam_label=camids, view_label=target_view)
        #                 evaluator.update((feat, vid, camid))
        #         cmc, mAP, _, _, _, _, _ = evaluator.compute()
        #         logger.info("Validation Results - Epoch: {}".format(epoch))
        #         logger.info("mAP: {:.1%}".format(mAP))
        #         for r in [1, 5, 10, 20]:
        #             logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        #         torch.cuda.empty_cache()
        #     prec1 = cmc[0] + mAP
        #     is_best = prec1 > best_performance
        #     best_performance = max(prec1, best_performance)
        #     if is_best:
        #         best_epoch = epoch
        #     save_checkpoint(model.state_dict(), is_best, os.path.join(cfg.OUTPUT_DIR, 'checkpoint_ep.pth.tar'))

    logger.info("==> Best Perform {:.1%}, achieved at epoch {}".format(best_performance, best_epoch))
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)


def do_inference_dense(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("TFCLIP.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        img = img.to(device)  # torch.Size([64, 4, 3, 256, 128])
        if len(img.size()) == 6:
            # method = 'dense'
            b, n, s, c, h, w = img.size()
            assert (b == 1)
            img = img.view(b * n, s, c, h, w)  # torch.Size([5, 8, 3, 256, 128])

        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else:
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view)
            feat = feat.view(-1, feat.size(1))
            feat = torch.mean(feat, 0, keepdim=True)  # 1,512
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)


    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]



def do_inference_rrs(cfg,
                     model,
                     val_loader,
                     num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            # model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        img = img.to(device)  # torch.Size([64, 4, 3, 256, 128])
        if len(img.size()) == 6:
            # method = 'dense'
            b, n, s, c, h, w = img.size()
            assert (b == 1)
            img = img.view(b * n, s, c, h, w)  # torch.Size([5, 8, 3, 256, 128])

        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else:
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view)
            # feat = feat.view(-1, feat.size(1))
            # feat = torch.mean(feat, 0, keepdim=True)  # 1,512
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
