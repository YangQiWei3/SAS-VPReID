class clip_color_jitter:
    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.3, color_p=0.5):
        self.color_jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)  # 只做 hue
        self.color_p = color_p  # RandomApply 的概率

    # def apply_clip_consistent_color_jitter(self, clip):
    def __call__(self, clip):
        # 50% 概率不抖
        if random.random() >= self.color_p:
            return clip

        # 关键：只采样一次 params，得到一个可调用的 transform
        fn = T.ColorJitter.get_params(
            self.color_jitter.brightness,
            self.color_jitter.contrast,
            self.color_jitter.saturation,
            self.color_jitter.hue,
        )
        # 对 clip 的每帧用同一个 fn
        return [fn(img) for img in clip]



class VideoDataset(data.Dataset):
    """Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """

    def __init__(self,
                 dataset,
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader,
                 clip_transform = ''):
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        if clip_transform is not None:
            self.clip_transforms = []
            if 'ColorJitter' in clip_transform:
                self.clip_transforms.append(clip_color_jitter(hue=0.3, color_p=0.5))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (clip, pid, camid) where pid is identity of the clip.
        """
        tracklet = self.dataset[index]
        # Handle different tracklet formats:
        # Old: (img_paths, pid, camid)
        # New: (img_paths, pid, camid, altitude, distance, angle, aerial_distance, point_id)
        if len(tracklet) >= 8:
            img_paths, pid, camid, altitude, distance, angle, aerial_distance, point_id = tracklet
        elif len(tracklet) >= 4:
            img_paths, pid, camid, altitude = tracklet
            distance = angle = aerial_distance = point_id = 0.0  # defaults
        else:
            img_paths, pid, camid = tracklet
            altitude = distance = angle = aerial_distance = point_id = 0.0  # defaults

        if self.temporal_transform is not None:
            img_paths = self.temporal_transform(img_paths)

        # print(img_paths)
        random.shuffle(img_paths)
        # print("=================")
        # print(img_paths)
        # assert 1
        #
        clip = self.loader(img_paths)

        for clip_transform in self.clip_transforms:
            clip = clip_transform(clip)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        # trans T x C x H x W to C x T x H x W
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        # print(clip)
        # assert 1 < 0

        # Return all geometric information for altitude conditioning
        return clip, pid, camid, altitude, distance, angle, aerial_distance, point_id