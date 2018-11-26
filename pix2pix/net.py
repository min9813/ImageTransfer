#!/usr/bin/env python

from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L

# U-net https://arxiv.org/pdf/1611.07004v1.pdf


class CBRDropout(chainer.Chain):

    def __init__(self, ch0, ch1, activation=F.relu, sample="down"):
        w = chainer.initializers.Normal(0.02)
        self.activation = activation
        super(CBRDropout, self).__init__()
        with self.init_scope():
            if sample == "down":
                self.c = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
            else:
                self.c = L.Deconvolution2D(ch0, ch1, 4, 2, 1, initialW=w)
            self.bn = L.BatchNormalization(ch1)

    def __call__(self, x):
        h = self.bn(self.c(x))
        h = self.activation(F.dropout(h))

        return h


class CBR(chainer.Chain):

    def __init__(self, ch0, ch1, activation=F.relu, sample="down"):
        w = chainer.initializers.Normal(0.02)
        self.activation = activation
        super(CBR, self).__init__()
        with self.init_scope():
            if sample == "down":
                self.c = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
            else:
                self.c = L.Deconvolution2D(ch0, ch1, 4, 2, 1, initialW=w)
            self.bn = L.BatchNormalization(ch1)

    def __call__(self, x):
        h = self.bn(self.c(x))
        h = self.activation(h)

        return h


class CR(chainer.Chain):

    def __init__(self, ch0, ch1, activation=F.relu, sample="down"):
        w = chainer.initializers.Normal(0.02)
        self.activation = activation
        super(CR, self).__init__()
        with self.init_scope():
            if sample == "down":
                self.c = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
            else:
                self.c = L.Deconvolution2D(ch0, ch1, 4, 2, 1, initialW=w)

    def __call__(self, x):
        h = self.c(x)
        h = self.activation(h)

        return h


class Encoder(chainer.Chain):
    def __init__(self, in_channel):
        w = chainer.initializers.Normal(0.02)
        super(Encoder, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(in_channel, 64, 3, 1, 1, initialW=w)
            self.c1 = CBR(64, 128, activation=F.leaky_relu)
            self.c2 = CBR(128, 256, activation=F.leaky_relu)
            self.c3 = CBR(256, 512, activation=F.leaky_relu)
            self.c4 = CBR(512, 512, activation=F.leaky_relu)
            self.c5 = CBR(512, 512, activation=F.leaky_relu)
            self.c6 = CBR(512, 512, activation=F.leaky_relu)
            self.c7 = CBR(512, 512, activation=F.leaky_relu)

    def __call__(self, x):
        h0 = F.leaky_relu(self.c0(x))
        h1 = self.c1(h0)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h6 = self.c6(h5)
        h7 = self.c7(h6)

        return [h0, h1, h2, h3, h4, h5, h6, h7]


class Decoder(chainer.Chain):
    def __init__(self, out_channel):
        w = chainer.initializers.Normal(0.02)
        super(Decoder, self).__init__()
        with self.init_scope():
            self.c0 = CBRDropout(512, 512, sample="up", activation=F.relu)
            self.c1 = CBRDropout(1024, 512, sample="up", activation=F.relu)
            self.c2 = CBRDropout(1024, 512, sample="up", activation=F.relu)
            self.c3 = CBR(1024, 512, sample="up", activation=F.relu)
            self.c4 = CBR(1024, 256, sample="up", activation=F.relu)
            self.c5 = CBR(512, 128, sample="up", activation=F.relu)
            self.c6 = CBR(256, 64, sample="up", activation=F.relu)
            self.c7 = L.Convolution2D(128, out_channel, 3, 1, 1, initialW=w)

    def __call__(self, hs):
        h = self.c0(hs[-1])
        h = self.c1(F.concat([h, hs[-2]]))
        h = self.c2(F.concat([h, hs[-3]]))
        h = self.c3(F.concat([h, hs[-4]]))
        h = self.c4(F.concat([h, hs[-5]]))
        h = self.c5(F.concat([h, hs[-6]]))
        h = self.c6(F.concat([h, hs[-7]]))
        h = self.c7(F.concat([h, hs[-8]]))

        return h


class Generator(chainer.Chain):

    def __init__(self, in_channel, out_channel):
        super(Generator, self).__init__()
        with self.init_scope():
            self.encoder = Encoder(in_channel)
            self.decoder = Decoder(out_channel)

    def __call__(self, x):
        h = self.encoder(x)
        h = self.decoder(h)

        return h


class Discriminator(chainer.Chain):
    def __init__(self, in_channel, out_channel):
        w = chainer.initializers.Normal(0.02)
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.c0_0 = CR(in_channel, 32, sample="down",
                           activation=F.leaky_relu)
            self.c0_1 = CR(out_channel, 32, sample="down",
                           activation=F.leaky_relu)
            self.c1 = CR(64, 128, sample="down", activation=F.leaky_relu)
            self.c2 = CR(128, 256, sample="down", activation=F.leaky_relu)
            self.c3 = CR(256, 512, sample="down", activation=F.leaky_relu)
            self.c4 = L.Convolution2D(512, 1, 3, 1, 1, initialW=w)

    def __call__(self, x_0, x_1):
        h = F.concat([self.c0_0(x_0), self.c0_1(x_1)])
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        #h = F.average_pooling_2d(h, h.data.shape[2], 1, 0)
        return h
