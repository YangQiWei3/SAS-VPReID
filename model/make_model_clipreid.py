import os.path

import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from collections import OrderedDict
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from .clip.model import QuickGELU, LayerNorm
# from .TAT import TemporalAttentionTransformer
from .Visual_Prompt import visual_prompt
from .vivim import MambaLayer

from .tsm.tsm import TSM
from .tsm.dsa import DSA
from .tsm.asa import ASA

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

model_paths={
    "ViT-B-16":['/yqw/checkpoints/torch/clip/ViT-B-16.pt'],
    "ViT-L-14":['/yqw/checkpoints/torch/clip/ViT-L-14.pt'],
    "ViT-L-14-336":['/yqw/checkpoints/torch/clip/ViT-L-14-336px.pt'],
}
from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    # model_path = clip._download(url)
    for pth in model_paths[backbone_name]:
        if os.path.exists(pth):
            model_path = pth
            break
        else:
            model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

def load_tsm_pretrained(model, ckpt_path='/yqw/checkpoints/torch/SMPL/tsm_model_wo.pth.tar'):
    ckpt = torch.load(ckpt_path)
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)
    return model

from torch.nn import init
class Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
        init.normal_(self.classifier.weight.data, std=0.001)
        init.constant_(self.classifier.bias.data, 0.0)

    def forward(self, x):
        y = self.classifier(x)

        return y

class CrossFramelAttentionBlock(nn.Module):  # 跨帧注意力
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, droppath=0., T=0, ):
        super().__init__()
        self.T = T

        self.message_fc = nn.Linear(d_model, d_model)  # 768 ->768  用一个FC得到信息token
        self.message_ln = LayerNorm(d_model)
        self.message_attn = nn.MultiheadAttention(d_model, n_head, )

        self.attn = nn.MultiheadAttention(d_model, n_head, )
        self.ln_1 = LayerNorm(d_model)

        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))  # mlp
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        l, bt, d = x.size()  # 197, 32, 768
        b = bt // self.T  # 4
        x = x.view(l, b, self.T, d)  # torch.Size([197, 4, 8, 768])
        # x_cls = [4, 8, 768]
        ######## 1.TMC  #####################
        msg_token = self.message_fc(x.mean(0))  # torch.Size([4, 8, 768])
        # msg_token = x.mean(0)  # torch.Size([4, 8, 768])
        msg_token = msg_token.view(b, self.T, 1, d)  # torch.Size([4, 8, 1, 768])

        msg_token = msg_token.permute(1, 2, 0, 3).view(self.T, b, d)  # torch.Size([8, 4, 768])
        msg_token = msg_token + self.drop_path(
            self.message_attn(self.message_ln(msg_token), self.message_ln(msg_token), self.message_ln(msg_token),
                              need_weights=False)[0])
        msg_token = msg_token.view(self.T, 1, b, d).permute(1, 2, 0, 3)  # torch.Size([1, 4, 8, 768])

        x = torch.cat([x, msg_token], dim=0)  # torch.Size([198, 4, 8, 768])
        ########  2.MD ###################
        x = x.view(l + 1, -1, d)  # torch.Size([198, 32, 768])
        x = x + self.drop_path(self.attention(self.ln_1(x)))  # torch.Size([198, 32, 768])
        # x = x[:l, :, :]  # torch.Size([197, 128, 768])
        x = x + self.drop_path(self.mlp(self.ln_2(x)))  # torch.Size([197, 32, 768])
        return x


class Temporal_Memory_Difusion(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, droppath=None,
                T=8):
        super().__init__()
        # self.use_checkpoint = use_checkpoint
        if droppath is None:
            droppath = [0.0 for i in range(layers)]
        self.width = width
        self.layers = layers

        self.resblocks = nn.Sequential(
            *[CrossFramelAttentionBlock(width, heads, attn_mask, droppath[i], T) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)



class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if 'ViT-B-16' in self.model_name:
            self.in_planes = 768
            self.in_planes_proj = 512
        elif 'ViT-L-14' in self.model_name:
            self.in_planes = 1024
            self.in_planes_proj = 768
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE    # 1

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        if 'ViT-B-16' in self.model_name:
            self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
            self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
            clip_model = load_clip_to_cpu('ViT-B-16', self.h_resolution, self.w_resolution, self.vision_stride_size)
        elif 'ViT-L-14-336' in self.model_name:
            self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 14) // cfg.MODEL.STRIDE_SIZE[0] + 1)
            self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 14) // cfg.MODEL.STRIDE_SIZE[1] + 1)
            clip_model = load_clip_to_cpu('ViT-L-14-336', self.h_resolution, self.w_resolution, self.vision_stride_size)
        elif 'ViT-L-14' in self.model_name:
            self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 14) // cfg.MODEL.STRIDE_SIZE[0] + 1)
            self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 14) // cfg.MODEL.STRIDE_SIZE[1] + 1)
            clip_model = load_clip_to_cpu('ViT-L-14', self.h_resolution, self.w_resolution, self.vision_stride_size)
        else:
            raise NotImplementedError
        # self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        # clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")
        self.image_encoder = clip_model.visual

        # Trick: freeze patch projection for improved stability
        # https://arxiv.org/pdf/2104.02057.pdf
        for _, v in self.image_encoder.conv1.named_parameters():
            v.requires_grad_(False)
        print('Freeze patch projection layer with shape {}'.format(self.image_encoder.conv1.weight.shape))
        freeze_layer_num = 0
        if freeze_layer_num > -1:
            for name, param in self.image_encoder.named_parameters():
                # print(name)
                # top layers always need to train
                if name.find("ln_post.") == 0 or name.find("proj") == 0:
                    continue  # need to train
                elif name.find("transformer.resblocks.") == 0:
                    layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                    if layer_num >= freeze_layer_num:
                        continue  # need to train
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False

        self.classifier = nn.Linear(self.in_planes_proj + self.in_planes, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.classifier2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier2.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)
        self.classifier_proj_temp = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_proj_temp.apply(weights_init_classifier)
        self.classifier_proj_temp2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_proj_temp2.apply(weights_init_classifier)
        self.classifier_proj_tsm = nn.Linear(cfg.MODEL.TSM_BETA_NUM, self.num_classes, bias=False)
        self.classifier_proj_tsm.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.bottleneck_proj_temp = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_proj_temp.bias.requires_grad_(False)
        self.bottleneck_proj_temp.apply(weights_init_kaiming)

        self.bottleneck_proj_temp2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_proj_temp2.bias.requires_grad_(False)
        self.bottleneck_proj_temp2.apply(weights_init_kaiming)

        self.bottleneck_proj_tsm = nn.BatchNorm1d(cfg.MODEL.TSM_BETA_NUM)
        self.bottleneck_proj_tsm.bias.requires_grad_(False)
        self.bottleneck_proj_tsm.apply(weights_init_kaiming)

        dataset_name = cfg.DATASETS.NAMES
        self.TMD_mamba = MambaLayer(dim=self.in_planes)
        self.Her_t_mamba1 = MambaLayer(dim=self.in_planes, nframe=2)
        self.Her_t_mamba2 = MambaLayer(dim=self.in_planes, nframe=4)
        # self.Her_t_mamba3 = MambaLayer(dim=768)
        self.norm3_mamba = nn.LayerNorm(self.in_planes)
        self.norm4_mamba = nn.LayerNorm(self.in_planes)
        self.alpha = nn.Parameter(torch.zeros(3))

        self.tsm = TSM(n_layers=2, hidden_size=1024, add_linear=True, use_residual=True)
        self.tsm = load_tsm_pretrained(self.tsm)
        self.pro_tsm = nn.Linear(self.in_planes, 2048, bias=False)
        self.shape_agg = ASA(rnn_size=1024,
                             input_size=1024,
                             num_shape_params=cfg.MODEL.TSM_BETA_NUM,
                             num_layers=2,
                             output_size=cfg.MODEL.TSM_BETA_NUM,
                             feature_pool='attention',
                             attention_size=1024,
                             attention_dropout=0.2,
                             attention_layers=3)
        self.shape_classifiers = nn.ModuleList(
            [Classifier(feature_dim=cfg.MODEL.TSM_BETA_NUM, num_classes=self.num_classes)
             for _ in range(cfg.DATALOADER.SEQ_LEN)])
        # self.SSP = ImageSpecificPrompt()
        # self.TAT = TemporalAttentionTransformer(T=cfg.INPUT.SEQ_LEN, embed_dim=512, layers=1)
        # width=768  layers=12  heads=12, droppath=None, use_checkpoint, T=8
        # self.SAT = Transformer_SP(width=512, layers=1, heads=8, droppath=None, T=cfg.INPUT.SEQ_LEN)
        # "meanP", "LSTM", "Transf", "Conv_1D", "Transf_cls"
        # self.temppool = visual_prompt(sim_head='meanP', T=cfg.INPUT.SEQ_LEN)
        # self.TMD = Temporal_Memory_Difusion(width=768, layers=1, heads=12, droppath=None, T=cfg.INPUT.SEQ_LEN)
        # self.ln_post = LayerNorm(512)


    def forward(self, x = None, get_image = False, cam_label= None, view_label=None):
        x = x.permute(0, 2, 1, 3, 4) # B, C, T, H, W -> B, T, C, H, W
        B, T, C, H, W = x.shape  # B=64, T=4.C=3 H=256,W=128

        if get_image == True:
            x = x.reshape(-1, C, H, W)  # 256,3,256,128

            image_features, image_features_proj = self.image_encoder(x)

            image_features = image_features[:, 0]
            image_features = image_features.view(B, T, -1)  # torch.Size([12, 8, 768])
            image_features = image_features.mean(1)

            img_feature_proj = image_features_proj[:,0]
            img_feature_proj = img_feature_proj.view(B, T, -1)  # torch.Size([64, 4, 512])
            img_feature_proj = img_feature_proj.mean(1)  # torch.Size([64, 512])


            feat = self.bottleneck(image_features)  # torch.Size([16, 768])
            feat_proj = self.bottleneck_proj(img_feature_proj)  # torch.Size([16, 512])
            out_feat = torch.cat([feat, feat_proj], dim=1)  # torch.Size([64, 1280])

            return out_feat

        x = x.reshape(-1, C, H, W)  # torch.Size([64, 3, 256, 128])

        if cam_label != None and view_label!=None:
            cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
        elif cam_label != None:  # 1
            cv_embed = self.sie_coe * self.cv_embed[cam_label]
        elif view_label!=None:
            cv_embed = self.sie_coe * self.cv_embed[view_label]
        else:
            cv_embed = None
        # cv_embed = cv_embed.repeat((1, B)).view(B, -1)  # torch.Size([64, 768])
        # cv_embed = cv_embed.repeat((1, T)).view(B * T, -1)  # torch.Size([64, 768])
        cv_embed = cv_embed.repeat((1, T)).view(B * T, -1)  # torch.Size([64, 768])
        # torch.Size([64, 129, 768])  torch.Size([64, 129, 768])  torch.Size([64, 129, 512])
        image_features, image_features_proj_raw = self.image_encoder(x, cv_embed)
        # image_features_SAT: torch.Size([129, 64, 768])

        ###################################################
        img_feature = image_features[:, 0]  # torch.Size([64, 768])
        img_feature_proj = image_features_proj_raw[:, 0]  # torch.Size([64, 512])

        ###################################################
        img_feature = img_feature.view(B, T, -1)  # torch.Size([16, 4, 768])
        img_feature_proj = img_feature_proj.view(B, T, -1)  # # torch.Size([16, 4, 512])
        # f_tp = self.temppool(img_feature_proj)  # b, 512
        img_feature = img_feature.mean(1)  # torch.Size([16, 768])
        img_feature_proj = img_feature_proj.mean(1)  # torch.Size([16, 512])
        ###################################################
        feats_for_mamba = image_features.detach()  # BT, hw,D = 64, 128, 768
        # BT, hw, D => BT, D, hw => B, T, D, hw => B, D, T, hw => B, D, T, h, w
        # feats_for_mamba = feats_for_mamba.permute(0, 2, 1)  # torch.Size([64, 768, 128])
        _, num_token, D = feats_for_mamba.shape
        feats_for_mamba = feats_for_mamba.reshape(B, T, num_token, D).contiguous()  #  torch.Size([8, 8, 129, 768])
        x_path1 = torch.zeros(feats_for_mamba.size()).cuda()
        x_path2 = torch.zeros(feats_for_mamba.size()).cuda()
        feats_for_mamba_total = feats_for_mamba.reshape(B, T * num_token, D)  # torch.Size([30, 1024, 768])
        mamba_output = self.TMD_mamba(feats_for_mamba_total)  # B, thw, D # torch.Size([8, 1024, 768])
        mamba_output_tap = mamba_output.reshape(B, T, num_token, D).contiguous().view(B * T, -1, D)  # B, D  torch.Size([64, 129, 768])
        ####
        tt = T // 4  # 8
        for j in range(T // tt):
            inputs = feats_for_mamba[:, j * tt: (j + 1) * tt, :, :]  # torch.Size([8, 2, 129, 768])
            inputs = inputs.reshape(B, tt*num_token, D).contiguous()
            x_div = self.Her_t_mamba1(inputs)  # torch.Size([8, 258, 768])
            x_div = x_div.reshape(B, tt, num_token, D).contiguous()  # torch.Size([8, 2, 129, 768])
            x_path1[:, j * tt: (j + 1) * tt,:, :] = x_div  # torch.Size([4, 8, 129 768])

        tt = T // 2  # 8
        for j in range(T // tt):
            inputs = feats_for_mamba[:, j * tt: (j + 1) * tt, :, :]  # torch.Size([8, 4, 129, 768])
            inputs = inputs.reshape(B, tt * num_token, D).contiguous()  # torch.Size([8, 516, 768])
            x_div = self.Her_t_mamba2(inputs)
            x_div = x_div.reshape(B, tt, num_token, D).contiguous()
            x_path2[:, j * tt: (j + 1) * tt, :, :] = x_div  # torch.Size([8, 8, 129, 768])

        ####
        cls_f_sp = mamba_output_tap.mean(1)  # torch.Size([64, 768])
        cls_f_sp = self.norm4_mamba(cls_f_sp)
        cls_f_sp_tap = cls_f_sp.view(B, T, -1)  # torch.Size([8, 8, 768])


        x_path1 = x_path1.mean(2).mean(1)  # torch.Size([8, 8, 768])
        x_path2 = x_path2.mean(2).mean(1)
        cls_f_tp = cls_f_sp_tap.mean(1)

        # cls_f_tp = x_path1 + x_path2 + cls_f_tp
        weights = torch.softmax(self.alpha, dim=0)
        cls_f_tp = (weights[0] * x_path1 + weights[1] * x_path2 + weights[2] * cls_f_tp)

        cls_f_tp = self.norm3_mamba(cls_f_tp)

        ###################################################
        feats_for_tsm = image_features.detach()  # BT, hw,D = 64, 128, 768
        _, num_token, D = feats_for_tsm.shape
        feats_for_tsm = feats_for_tsm.mean(1) # BT, D  torch.Size([64, 768])
        feats_for_tsm = self.pro_tsm(feats_for_tsm) # BT, D'  torch.Size([64, 2048])
        feats_for_tsm = feats_for_tsm.reshape(B, T, -1).contiguous()  # torch.Size([8, 8, 2048])

        betas, shape_1024s = [], []
        for i in range(feats_for_tsm.size(0)):
            beta, shape_1024 = self.tsm(feats_for_tsm[i, :, :].unsqueeze(0))
            betas.append(beta)
            shape_1024s.append(shape_1024)
        betas = torch.stack(betas, dim=0)
        shape_1024s = torch.stack(shape_1024s, dim=0) # 16 8 1024
        framewise_shapes, videowise_shapes = self.shape_agg(shape_1024s)  # 16 8 10   16 10

        ###################################################
        feat = self.bottleneck(img_feature)  # torch.Size([16, 768])
        feat_proj = self.bottleneck_proj(img_feature_proj)  # torch.Size([16, 512])
        feat_proj_frame = self.bottleneck_proj_temp(cls_f_sp)
        feat_proj_temp = self.bottleneck_proj_temp2(cls_f_tp)
        videowise_shapes_proj = self.bottleneck_proj_tsm(videowise_shapes)

        if self.training:
            out_feat = torch.cat([feat, feat_proj], dim=1)  # torch.Size([64, 1280])
            cls_score = self.classifier(out_feat)
            cls_score_proj_frame = self.classifier_proj_temp(feat_proj_frame)
            cls_score_proj_temp = self.classifier_proj_temp2(feat_proj_temp)
            cls_score_proj_shapes = self.classifier_proj_tsm(videowise_shapes_proj)
            framewise_shape_logits = self.optimize_shape_id_loss(framewise_shapes)

            return [cls_score, cls_score_proj_temp, cls_score_proj_frame, cls_score_proj_shapes], [img_feature, img_feature_proj, cls_f_tp, videowise_shapes], out_feat, (betas, framewise_shapes, framewise_shape_logits)

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj, videowise_shapes], dim=1)
            else:
                out_feat = torch.cat([feat, feat_proj, feat_proj_temp, videowise_shapes_proj], dim=1)  # torch.Size([64, 1280])
#                 out_feat = torch.cat([feat, feat_proj], dim=1)  # torch.Size([64, 1280])
                # out_feat = feat_proj_temp  # torch.Size([64, 1280])
                # out_feat = torch.cat([feat, feat_proj, cls_f_tp], dim=1)  # torch.Size([64, 1280])
                return out_feat
                # return f_tp

    def optimize_shape_id_loss(self, framewise_shapes): # get framewise shape logits and feed into ID loss
        batch_size, seq_len, _ = framewise_shapes.shape
        batch_framewise_logits_list = []
        for batch_idx in range(batch_size):
            framewise_logits_list = []
            for frame_idx in range(seq_len):
                frame_feature = framewise_shapes[batch_idx][frame_idx]
                logits = self.shape_classifiers[frame_idx](frame_feature)
                # Store the logits for the current frame
                framewise_logits_list.append(logits)
            batch_framewise_logits_list.append(framewise_logits_list)
        final = torch.stack([torch.stack(batch_framewise_logits_list[i]) for i in range(len(batch_framewise_logits_list))])
        return final

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


class MulitHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PromptGeneratorLayer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dropout=0.1,
    ):
        super().__init__()
        self.self_attn = MulitHeadAttention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = MulitHeadAttention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, visual):
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        q = self.norm2(x)
        x = x + self.cross_attn(q, visual, visual)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


class ImageSpecificPrompt(nn.Module):
    def __init__(self, layers=2, embed_dim=512, alpha=0.1, ):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)  # 512
        self.memory_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        self.decoder = nn.ModuleList([PromptGeneratorLayer(embed_dim, embed_dim // 64) for _ in range(layers)])  # 2层
        self.alpha = nn.Parameter(torch.ones(embed_dim) * alpha)  # torch.Size([512])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, text, visual):  # text: torch.Size([64, 150, 512]) visual: torch.Size([64, 129, 512])
        # B, N, C = visual.shape
        visual = self.memory_proj(visual)  # torch.Size([8, 129, 512])
        text = self.text_proj(text)  # torch.Size([8, 625, 512])
        # visual = self.norm(visual)  # torch.Size([4, 196, 512])  torch.Size([8, 129, 512])
        for layer in self.decoder:
            text = layer(text, visual)  # torch.Size([64, 150, 512])
        text = self.out_proj(text)
        return text
