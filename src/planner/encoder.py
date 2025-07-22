# -------------------------------------------------------------------------------------------------------------
# File: encoder.py
# Project: DAA*: Deep Angular A Star for Image-based Path Planning
# Contributors:
#     Zhiwei Xu <zwxu064@gmail.com>
#
# Copyright (c) 2025 Zhiwei Xu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# -------------------------------------------------------------------------------------------------------------

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

# =============================================================================================================

class EncoderBase(nn.Module):
    def __init__(
        self,
        input_dim,
        encoder_depth=4,
        const=None
    ):
        super().__init__()
        self.model = self.construct_encoder(input_dim, encoder_depth)

        if const is None:
            self.const = 1.0
        else:
            self.const = nn.Parameter(torch.ones(1) * const)

    def construct_encoder(
        self,
        input_dim,
        encoder_depth
    ):
        pass

    def forward(
        self,
        x,
        map_designs=None
    ):
        y = torch.sigmoid(self.model(x))

        if map_designs is not None:
            y = y * map_designs + torch.ones_like(y) * (1 - map_designs)

        return y * self.const

# =============================================================================================================

class Unet(EncoderBase):

    DECODER_CHANNELS = [256, 128, 64, 32, 16]

    def construct_encoder(
        self,
        input_dim,
        encoder_depth
    ):
        decoder_channels = self.DECODER_CHANNELS[:encoder_depth]

        return smp.Unet(
            encoder_name="vgg16_bn",
            encoder_weights=None,
            classes=1,
            in_channels=input_dim,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
        )

# =============================================================================================================

class CNN(EncoderBase):

    CHANNELS = [32, 64, 128, 256]

    def construct_encoder(
        self,
        input_dim,
        encoder_depth
    ):
        channels = [input_dim] + self.CHANNELS[:encoder_depth] + [1]
        blocks = []

        for i in range(len(channels) - 1):
            blocks.append(nn.Conv2d(channels[i], channels[i + 1], 3, 1, 1))
            blocks.append(nn.BatchNorm2d(channels[i + 1]))
            blocks.append(nn.ReLU())

        return nn.Sequential(*blocks[:-1])

# =============================================================================================================

class CNNDownSize(CNN):
    def construct_encoder(
        self,
        input_dim,
        encoder_depth
    ):
        channels = [input_dim] + self.CHANNELS[:encoder_depth] + [1]
        blocks = []

        for i in range(len(channels) - 1):
            blocks.append(nn.Conv2d(channels[i], channels[i + 1], 3, 1, 1))
            blocks.append(nn.BatchNorm2d(channels[i + 1]))
            blocks.append(nn.ReLU())
            blocks.append(nn.MaxPool2d((2, 2)))

        return nn.Sequential(*blocks[:-2])