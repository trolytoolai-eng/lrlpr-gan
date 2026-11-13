# -*- coding: utf-8 -*-
"""
Perceptual loss VGG19 (sao chép từ losses/lossPack.py của repo LPSRGAN, bỏ phần TensorFlow và .cuda() cứng).
Loss OCR có gradient nằm trong vòng huấn luyện (train.py, hàm step_fn).
"""

import torch.nn as nn


class VGG19Perceptual(nn.Module):
    """
    So sánh ảnh SR và ảnh HR ở nhiều tầng đặc trưng của VGG19 (đã học sẵn trên ImageNet).
    Giúp ảnh sinh ra giống ảnh thật về hình dạng nét chữ, không chỉ giống từng điểm ảnh.
    """

    def __init__(self, pretrained=True, weights=(0.1, 0.1, 1.0, 1.0, 1.0)):
        super().__init__()
        from torchvision.models import vgg19
        try:
            from torchvision.models import VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1 if pretrained else None)
        except ImportError:  # torchvision cũ
            vgg = vgg19(pretrained=pretrained)
        feats = vgg.features.eval()
        cuts = [3, 8, 17, 26, 35]  # conv1_2, conv2_2, conv3_4, conv4_4, conv5_4 (như repo)
        blocks, last = [], 0
        for c in cuts:
            blocks.append(feats[last:c + 1])
            last = c + 1
        self.blocks = nn.ModuleList(blocks)
        for p in self.parameters():
            p.requires_grad = False
        self.weights = weights
        self.l1 = nn.L1Loss()

    def _features(self, x):
        x = (x - 0.45) / 0.225  # chuẩn hóa đơn giản giống repo
        outs = []
        for b in self.blocks:
            x = b(x)
            outs.append(x)
        return outs

    def forward(self, sr, hr):
        loss = 0.0
        for w, a, b in zip(self.weights, self._features(sr), self._features(hr)):
            loss = loss + w * self.l1(a, b)
        return loss
