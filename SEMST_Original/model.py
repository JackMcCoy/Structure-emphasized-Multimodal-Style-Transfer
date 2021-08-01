import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from normalisedVGG import NormalisedVGG
from VGGdecoder import Decoder
from feature_transfer import matching, labeled_whiten_and_color, calc_k
from utils import download_file_from_google_drive


def calc_mean_std(features):
    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1)
    return features_mean, features_std

def calc_mean_std_adain(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    #assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1)
    feat_var = feat_var.var(dim=2) + eps
    feat_std = feat_var.sqrt()
    feat_std = feat_std.view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1)
    feat_mean = feat_mean.mean(dim=2)
    feat_mean = feat_mean.reshape((N, C, 1, 1))
    return feat_mean, feat_std

def calc_content_loss(out_features, content_features):
    return F.mse_loss(out_features, content_features)


def calc_style_loss(out_middle_features, style_middle_features):
    loss = 0
    for c, s in zip(out_middle_features, style_middle_features):
        c_mean, c_std = calc_mean_std(c)
        s_mean, s_std = calc_mean_std(s)
        loss += F.mse_loss(c_mean, s_mean) + F.mse_loss(c_std, s_std)
    return loss

def calc_emd_loss(pred, target):
    """calculate emd loss.

    Args:
        pred (Tensor): of shape (N, C, H, W). Predicted tensor.
        target (Tensor): of shape (N, C, H, W). Ground truth tensor.
    """
    b, _, h, w = pred.size()
    pred = pred.view([b, -1, w * h])
    pred_norm = torch.sqrt((pred**2).sum(1).view([b, -1, 1]))
    pred = torch.transpose(pred, 2, 1)
    target_t = target.view([b, -1, w * h])
    target_norm = torch.sqrt((target**2).sum(1).view([b, 1, -1]))
    similarity = torch.bmm(pred, target_t) / pred_norm / target_norm
    dist = 1. - similarity
    return dist

def style_emd_loss(pred, target):
    """Forward Function.

    Args:
        pred (Tensor): of shape (N, C, H, W). Predicted tensor.
        target (Tensor): of shape (N, C, H, W). Ground truth tensor.
    """
    CX_M = calc_emd_loss(pred, target)
    m1,_ = CX_M.min(2)
    m2,_ = CX_M.min(1)
    m = torch.cat([m1.mean(), m2.mean()])
    loss_remd,_ = torch.max(m)
    return loss_remd

def content_relt_loss(pred, target):
    """Forward Function.

    Args:
        pred (Tensor): of shape (N, C, H, W). Predicted tensor.
        target (Tensor): of shape (N, C, H, W). Ground truth tensor.
    """
    dM = 1.
    Mx = calc_emd_loss(pred, pred)
    Mx = Mx / Mx.sum(1, keepdim=True)
    My = calc_emd_loss(target, target)
    My = My / My.sum(1, keepdim=True)
    loss_content = torch.abs(
        dM * (Mx - My)).mean() * pred.size()[2] * pred.size()[3]
    return loss_content

class VGGEncoder(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        vgg = NormalisedVGG(pretrained_path=pretrained_path).net
        self.block1 = vgg[: 4]
        self.block2 = vgg[4: 11]
        self.block3 = vgg[11: 18]
        self.block4 = vgg[18: 31]

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, images, output_last_feature=True):
        h1 = self.block1(images)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        h4 = self.block4(h3)
        if output_last_feature:
            return h4
        else:
            return [h1, h2, h3, h4]


class Model(nn.Module):
    def __init__(self,
                 alpha=1,
                 device='cpu',
                 use_kmeans_gpu=True,
                 pre_train=False):
        super().__init__()
        self.alpha = alpha
        self.device = device
        self.kmeans_device = device if use_kmeans_gpu else torch.device('cpu')
        if pre_train:
            if not os.path.exists('vgg_normalised_conv5_1.pth'):
                download_file_from_google_drive('1IAOFF5rDkVei035228Qp35hcTnliyMol',
                                                'vgg_normalised_conv5_1.pth')
            if not os.path.exists('decoder_relu4_1.pth'):
                download_file_from_google_drive('1kkoyNwRup9y5GT1mPbsZ_7WPQO9qB7ZZ',
                                                'decoder_relu4_1.pth')
            self.vgg_encoder = VGGEncoder('vgg_normalised_conv5_1.pth').to(device)
            self.decoder = Decoder(4, 'decoder_relu4_1.pth').to(device)
        else:
            self.vgg_encoder = VGGEncoder().to(device)
            self.decoder = Decoder(4).to(device)

    @staticmethod
    def calc_content_loss(out_features, content_features):
        return F.mse_loss(out_features, content_features)

    @staticmethod
    def calc_style_loss(out_middle_features, style_middle_features):
        loss = 0
        for c, s in zip(out_middle_features, style_middle_features):
            c_mean, c_std = calc_mean_std(c)
            s_mean, s_std = calc_mean_std(s)
            loss += F.mse_loss(c_mean, s_mean) + F.mse_loss(c_std, s_std)
        return loss

    @staticmethod
    def calc_emd_loss(pred, target):
        """calculate emd loss.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        b, _, h, w = pred.size()
        pred = pred.view([b, -1, w * h])
        pred_norm = torch.sqrt((pred ** 2).sum(1).view([b, -1, 1]))
        pred = torch.transpose(pred, 2, 1)
        target_t = target.view([b, -1, w * h])
        target_norm = torch.sqrt((target ** 2).sum(1).view([b, 1, -1]))
        similarity = torch.bmm(pred, target_t) / pred_norm / target_norm
        dist = 1. - similarity
        return dist

    @staticmethod
    def style_emd_loss(pred, target):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        CX_M = calc_emd_loss(pred, target)
        m1, _ = CX_M.min(2)
        m2, _ = CX_M.min(1)
        m = torch.cat([m1.mean(), m2.mean()])
        loss_remd, _ = torch.max(m)
        return loss_remd

    @staticmethod
    def content_relt_loss(pred, target):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        dM = 1.
        Mx = calc_emd_loss(pred, pred)
        Mx = Mx / Mx.sum(1, keepdim=True)
        My = calc_emd_loss(target, target)
        My = My / My.sum(1, keepdim=True)
        loss_content = torch.abs(
            dM * (Mx - My)).mean() * pred.size()[2] * pred.size()[3]
        return loss_content

    def generate(self,
                 content_image_path,
                 style_image_path,
                 content_image_tensor,
                 style_image_tensor,
                 alpha=None):
        alpha = self.alpha if alpha is None else alpha

        cs = []
        content_features = self.vgg_encoder(content_image_tensor.to(self.device),output_last_feature=False)
        style_features = self.vgg_encoder(style_image_tensor.to(self.device),output_last_features=False)

        for cp, sp, cf, sf in zip(content_image_path, style_image_path, [i[0] for i in content_features], [i[0] for i in style_features]):
            content_label = calc_k(cp, self.kmeans_device)
            style_label = calc_k(sp, self.kmeans_device)

            content_k = int(content_label.max().item() + 1)
            style_k = int(style_label.max().item() + 1)

            match = matching[(content_k, style_k)]
            # print(match)
            content_label = content_label.to(self.device)
            style_label = style_label.to(self.device)

            cs_feature = torch.zeros_like(cf)
            for i, j in match.items():
                cl = (content_label == i).unsqueeze(dim=0).expand_as(cf).to(torch.float)
                sl = torch.zeros_like(sf)
                for jj in j:
                    sl += (style_label == jj).unsqueeze(dim=0).expand_as(sf).to(torch.float)
                sl = sl.to(torch.bool)
                sub_sf = sf[sl].reshape(sf.shape[0], -1)
                cs_feature += labeled_whiten_and_color(cf, sub_sf, alpha, cl)

            cs.append(cs_feature.unsqueeze(dim=0))
        cs = torch.cat(cs, dim=0)
        out = self.decoder(cs,content_features,style_features)
        return out

    def forward(self,
                content_image_path,
                style_image_path,
                content_image_tensor,
                style_image_tensor,
                gamma=1):

        cs = []
        content_features = self.vgg_encoder(content_image_tensor.to(self.device),output_last_feature=False)
        style_features = self.vgg_encoder(style_image_tensor.to(self.device),output_last_feature=False)

        for cp, sp, cf, sf in zip(content_image_path, style_image_path, [i[0] for i in content_features], [i[0] for i in style_features]):
            content_label = calc_k(cp, self.kmeans_device)
            style_label = calc_k(sp, self.kmeans_device)
            content_k = int(content_label.max().item() + 1)
            style_k = int(style_label.max().item() + 1)

            match = matching[(content_k, style_k)]
            content_label = content_label.to(self.device)
            style_label = style_label.to(self.device)
            print(content_label)
            print(style_label)
            print(content_label.size())
            print(style_label.size())
            print(list(match.items())[0])
            cs_feature = torch.zeros_like(cf)
            for i, j in match.items():
                cl = (content_label == i).unsqueeze(dim=0).expand_as(cf).to(torch.float)
                sl = torch.zeros_like(sf)
                for jj in j:
                    sl += (style_label == jj).unsqueeze(dim=0).expand_as(sf).to(torch.float)
                sl = sl.to(torch.bool)
                sub_sf = sf[sl].reshape(sf.shape[0], -1)
                cs_feature += labeled_whiten_and_color(cf, sub_sf, self.alpha, cl)

            cs.append(cs_feature.unsqueeze(dim=0))

        cs = torch.cat(cs, dim=0)
        out = self.decoder(cs, content_features, style_features)

        out_features = self.vgg_encoder(out, output_last_feature=True)
        out_middle_features = self.vgg_encoder(out, output_last_feature=False)
        style_middle_features = self.vgg_encoder(style_image_tensor.to(self.device), output_last_feature=False)

        loss_c = self.calc_content_loss(out_features, content_features)
        loss_s = self.calc_style_loss(out_middle_features, style_middle_features)
        loss = loss_c + gamma * loss_s
        # print('loss: ', loss_c.item(), gamma*loss_s.item())

        return loss
