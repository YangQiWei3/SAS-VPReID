from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
import numpy as np
import collections
import random


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, indexes.tolist()):
            batch_centers[index].append(instance_feature)  # 找到这个ID对应的所有实例特征

        for index, features in batch_centers.items():
            distances = []
            for feature in features:  # 计算每个实例与质心之间的距离
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))  # 选择距离最小的，最不相似 即 最难的
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Mix_mean_hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        nums = len(ctx.features)//2
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()
            
        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, indexes.tolist()):
            batch_centers[index].append(instance_feature)  # 找到这个ID对应的所有实例特征

        ##### Mean
        # for index, features in batch_centers.items():
        #     feats = torch.stack(features, dim=0)
        #     features_mean = feats.mean(0)
        #     ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features_mean
        #     ctx.features[index] /= ctx.features[index].norm()
        ##### Hard
        for index, features in batch_centers.items():
            distances = []
            for feature in features:  # 计算每个实例与质心之间的距离
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            mean = torch.stack(features, dim=0).mean(0)  # 均值
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * mean
            ctx.features[index] /= ctx.features[index].norm()
            
            hard = np.argmin(np.array(distances))  #  余弦距离最近的，最不相似的  
            ctx.features[index+nums] = ctx.features[index+nums] * ctx.momentum + (1 - ctx.momentum) * features[hard]
            ctx.features[index+nums] /= ctx.features[index+nums].norm()

            # rand = random.choice(features)  # 随机选一个
        #### rand
#         for index, features in batch_centers.items():

#             features_mean = random.choice(features)

#             ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features_mean
#             ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None

def cm_mix(inputs, indexes, features, momentum=0.5):
    return CM_Mix_mean_hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features  # 2048
        self.num_samples = num_samples  # 79

        self.momentum = momentum  # 0.1
        self.temp = temp  # 0.05
        self.use_hard = use_hard  # false

        self.register_buffer('features', torch.zeros(num_samples*2, num_features))

    def forward(self, inputs, targets):

        inputs = F.normalize(inputs, dim=1).cuda()
        # if self.use_hard:
        #     outputs = cm_hard(inputs, targets, self.features, self.momentum)
        # else:
        #     outputs = cm(inputs, targets, self.features, self.momentum)
        outputs = cm_mix(inputs, targets, self.features, self.momentum)
        outputs /= self.temp
        
        mean, hard = torch.chunk(outputs, 2, dim=1)
        loss = 0.5 * (F.cross_entropy(hard, targets) + F.cross_entropy(mean, targets))

        return loss
