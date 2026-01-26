import torch
import torch.nn as nn
from torch.nn import functional as F

class CrossEntropyLabelSmooth(nn.Module):
    """
     Cross entropy loss with label smoothing regularizer + 安全检查版本.

     Args:
         num_classes (int): number of classes.
         epsilon (float): weight.
         use_gpu (bool): 已经不太需要, 保留只是为了兼容旧代码.
     """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, name=""):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.name = name  # 用来标识是哪个 head 的 loss
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        inputs : (batch_size, num_classes)  未经过 softmax 的 logits
        targets: (batch_size,)              int64 class indices
        """
        # ---- 真实 logits 维度 ----
        device = inputs.device
        actual_num_classes = inputs.size(1)

        targets_flat = targets.view(-1).to(device=device, dtype=torch.long)

        # ---- 越界检查 ----
        bad_low = (targets_flat < 0)
        bad_high = (targets_flat >= actual_num_classes)
        bad = bad_low | bad_high

        if bad.any():
            print(f"\n[ERROR in {self.name}] Invalid target detected!")
            print(f" inputs.shape          = {inputs.shape}")
            print(f" expected num_classes  = {self.num_classes}")
            print(f" actual logits dim     = {actual_num_classes}")
            print(f" targets shape         = {targets.shape}")
            print(f" targets min/max       = {targets_flat.min().item()}, {targets_flat.max().item()}")
            print(f" number of bad labels  = {bad.sum().item()}")
            bad_vals = targets_flat[bad][:20].tolist()
            print(f" bad values sample     = {bad_vals}")
            raise RuntimeError(f"[{self.name}] Found labels outside [0, {actual_num_classes - 1}]")

        # ---- 正常 label smoothing ----
        log_probs = self.logsoftmax(inputs)  # (B, C), 在同一个 device

        # one-hot, 注意这里不再 .cpu() / .cuda()，直接在同一 device 上
        targets_onehot = torch.zeros_like(log_probs).scatter_(
            1, targets_flat.unsqueeze(1), 1
        )

        # label smoothing
        targets_smoothed = (1.0 - self.epsilon) * targets_onehot + \
                           self.epsilon / float(actual_num_classes)

        loss = (-targets_smoothed * log_probs).mean(0).sum()
        return loss

# class CrossEntropyLabelSmooth(nn.Module):
#     """Cross entropy loss with label smoothing regularizer.
#
#     Reference:
#     Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
#     Equation: y = (1 - epsilon) * y + epsilon / K.
#
#     Args:
#         num_classes (int): number of classes.
#         epsilon (float): weight.
#     """
#
#     def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
#         super(CrossEntropyLabelSmooth, self).__init__()
#         self.num_classes = num_classes
#         self.epsilon = epsilon
#         self.use_gpu = use_gpu
#         self.logsoftmax = nn.LogSoftmax(dim=1)
#
#     def forward(self, inputs, targets):
#         """
#         Args:
#             inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
#             targets: ground truth labels with shape (num_classes)
#         """
#         log_probs = self.logsoftmax(inputs)
#         targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
#         if self.use_gpu: targets = targets.cuda()
#         targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
#         loss = (- targets * log_probs).mean(0).sum()
#         return loss
#
class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()