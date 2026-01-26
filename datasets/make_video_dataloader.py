from datasets import data_manager
from torch.utils.data import DataLoader

from datasets.sampler import RandomIdentitySampler_Video
from dataset_transformer import temporal_transforms as TT, spatial_transforms as ST
from datasets.video_loader import VideoDataset, VideoDatasetInfer

import model.clip




def make_CLIMB_dataloader(cfg, all_iters=False):
    """
    PCL dataloader. It returns 3 dataloaders: training loader, cluster loader and validation loader.
    - For training loader, PK sampling is applied to select K instances from P classes.
    - For cluster loader, a plain loader is returned with validation augmentation but on
      training samples.
    - For validation loader, a validation loader is returned on test samples.

    Args:
    - dataset: dataset object.
    - all_iters: if `all_iters=True`, number training iteration is decided by `num_samples//batchsize`
    """
    # split_id = cfg.DATASETS.SPLIT
    # seq_srd = cfg.INPUT.SEQ_SRD
    # seq_len = cfg.INPUT.SEQ_LEN
    # num_workers = cfg.DATALOADER.NUM_WORKERS

    # Pass subset parameter if available in config
    subset = getattr(cfg.DATASETS, 'SUBSET', 'case1_aerial_to_ground') # case1_aerial_to_ground case2_ground_to_aerial case3_aerial_to_aerial
    # subset = 'case3_aerial_to_aerial' # case1_aerial_to_ground case2_ground_to_aerial case3_aerial_to_aerial
    dataset = data_manager.init_dataset(name=cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR, subset=subset)
    num_workers = cfg.DATALOADER.NUM_WORKERS

    spatial_transform_train_stage2 = ST.Compose([
        ST.Scale(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        ST.RandomHorizontalFlip(cfg.INPUT.PROB),
        ST.ToTensor(),
        ST.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
        ST.RandomErasing(probability=cfg.INPUT.RE_PROB)])
    spatial_transform_test = ST.Compose([
        ST.Scale(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        ST.ToTensor(),
        ST.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)])
    temporal_transform_train = TT.TemporalRestrictedCrop(size=cfg.DATALOADER.SEQ_LEN)
    temporal_transform_test= TT.TemporalRestrictedBeginCrop(size=cfg.DATALOADER.SEQ_LEN)

    train_loader_stage1 = DataLoader(
        VideoDataset(
            dataset.train,
            spatial_transform=spatial_transform_test,
            temporal_transform=temporal_transform_test),
        sampler=RandomIdentitySampler_Video(dataset.train, num_instances=cfg.DATALOADER.NUM_INSTANCE),
        batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH, num_workers=num_workers,
        pin_memory=True, drop_last=True)

    train_loader_stage2 = DataLoader(
        VideoDataset(
            dataset.train,
            spatial_transform=spatial_transform_train_stage2,
            temporal_transform=temporal_transform_train,
            clip_transform='ColorJitter'),
        sampler=RandomIdentitySampler_Video(dataset.train, num_instances=cfg.DATALOADER.NUM_INSTANCE),
        batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH, num_workers=num_workers,
        pin_memory=True, drop_last=True)

    queryloader_sampled_frames = DataLoader(
        VideoDataset(dataset.query, spatial_transform=spatial_transform_test,
                     temporal_transform=temporal_transform_test),
        batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False)

    galleryloader_sampled_frames = DataLoader(
        VideoDataset(dataset.gallery, spatial_transform=spatial_transform_test,
                     temporal_transform=temporal_transform_test),
        batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False)

    num_classes = dataset.num_train_pids
    num_query = dataset.num_query_pids
    camera_num = dataset.num_camera
    view_num = None

    return (train_loader_stage2,
            train_loader_stage1,
            queryloader_sampled_frames,
            galleryloader_sampled_frames,
            num_query,
            num_classes,
            camera_num,
            view_num)

def make_dataloader(cfg):
    # Pass subset parameter if available in config
    subset = getattr(cfg.DATASETS, 'SUBSET', 'case1_aerial_to_ground')
    dataset = data_manager.init_dataset(name=cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR, subset=subset)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    spatial_transform_train_stage2 = ST.Compose([
        ST.Scale(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        ST.RandomHorizontalFlip(cfg.INPUT.PROB),
        ST.ToTensor(),
        ST.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
        ST.RandomErasing(probability=cfg.INPUT.RE_PROB)])

    spatial_transform_train_stage1 = ST.Compose([
        ST.Scale(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        ST.ToTensor(),
        ST.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)])

    spatial_transform_test = ST.Compose([
        ST.Scale(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        ST.ToTensor(),
        ST.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)])

    temporal_transform_train = TT.TemporalRestrictedCrop(size=cfg.DATALOADER.SEQ_LEN)
    temporal_transform_test= TT.TemporalRestrictedBeginCrop(size=cfg.DATALOADER.SEQ_LEN)

    train_loader_stage1 = DataLoader(
        VideoDataset(
            dataset.train,
            spatial_transform=spatial_transform_train_stage2,
            temporal_transform=temporal_transform_train),
        sampler=RandomIdentitySampler_Video(dataset.train, num_instances=cfg.DATALOADER.NUM_INSTANCE),
        batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH, num_workers=num_workers,
        pin_memory=True, drop_last=True)

    train_loader_stage2 = DataLoader(
        VideoDataset(
            dataset.train,
            spatial_transform=spatial_transform_train_stage2,
            temporal_transform=temporal_transform_train),
        sampler=RandomIdentitySampler_Video(dataset.train, num_instances=cfg.DATALOADER.NUM_INSTANCE),
        batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH, num_workers=num_workers,
        pin_memory=True, drop_last=True)

    queryloader_sampled_frames = DataLoader(
        VideoDataset(dataset.query, spatial_transform=spatial_transform_test,
                     temporal_transform=temporal_transform_test),
        batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False)

    galleryloader_sampled_frames = DataLoader(
        VideoDataset(dataset.gallery, spatial_transform=spatial_transform_test,
                     temporal_transform=temporal_transform_test),
        batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False)

    queryloader_all_frames = DataLoader(
        VideoDatasetInfer(
            dataset.query, spatial_transform=spatial_transform_test, seq_len=cfg.DATALOADER.SEQ_LEN),
        batch_size=1, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True, drop_last=False)

    galleryloader_all_frames = DataLoader(
        VideoDatasetInfer(dataset.gallery, spatial_transform=spatial_transform_test, seq_len=cfg.DATALOADER.SEQ_LEN),
        batch_size=1, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True, drop_last=False)

    num_classes = dataset.num_train_pids
    num_query = dataset.num_query_pids
    camera_num = dataset.num_camera

    return (train_loader_stage2,
            train_loader_stage1,
            queryloader_sampled_frames,
            galleryloader_sampled_frames,
            queryloader_all_frames,
            galleryloader_all_frames,
            num_classes,
            num_query,
            camera_num)