import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class Interpolate(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor)
        return x

class ResBlock(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs

def adain(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std_adain(style_feat)
    content_mean, content_std = calc_mean_std_adain(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    t = normalized_feat * style_std.expand(size) + style_mean.expand(size)
    return t

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

decoder_1 = nn.Sequential(
    ResBlock(nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
        nn.Conv2d(512, 512, (1 , 1)),
    )),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
)
decoder_2 = nn.Sequential(
    ResBlock(nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.Conv2d(256, 256, (1 , 1)),
    )),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),

)
decoder_3 = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),

)
decoder_4 = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

'''
vgg_decoder_relu5_1 = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
    Interpolate(2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, 3),
    nn.ReLU(),
    Interpolate(2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, 3),
    nn.ReLU(),
    Interpolate(2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, 3),
    nn.ReLU(),
    Interpolate(2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, 3)
    )
'''

class Decoder(nn.Module):
    def __init__(self, level, pretrained_path=None):
        super().__init__()

        self.decoder_1 = decoder_1
        self.decoder_2 = decoder_2
        self.decoder_3 = decoder_3
        self.decoder_4 = decoder_4

    def init_weights(self):
        self.decoder_1.apply(init_weights)
        self.decoder_2.apply(init_weights)
        self.decoder_3.apply(init_weights)
        self.decoder_4.apply(init_weights)

    def forward(self, cs,content_feat,style_feats):
        m = nn.Upsample(scale_factor=2, mode='nearest')
        t = adain(content_feat[-1], style_feats[-1])
        t = self.decoder_1(t)
        t = m(t)
        # t_2 = UPSCALE CONTENT FEAT!
        t += adain(content_feat[-2], style_feats[-2])
        t = self.decoder_2(t)
        t = m(t)
        t += adain(content_feat[-3], style_feats[-3])
        t = self.decoder_3(t)
        t = m(t)
        t = (t+cs)/2
        t = self.decoder_4(t)
        return t
