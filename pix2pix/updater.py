#!/usr/bin/env python

import chainer
import chainer.functions as F
from chainer import Variable


class FacadeUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        models = kwargs.pop('models')
        lam = kwargs.pop('lam')
        self.gen = models["gen"]
        self.dis = models["dis"]
        self.lam1 = lam["lam1"]
        self.lam2 = lam["lam2"]
        super(FacadeUpdater, self).__init__(*args, **kwargs)

    def loss_enc(self, enc, x_out, t_out, y_out, lam1=100, lam2=1):
        batchsize, _, w, h = y_out.data.shape
        loss_rec = lam1 * (F.mean_absolute_error(x_out, t_out))
        loss_adv = lam2 * F.sum(F.softplus(-y_out)) / batchsize / w / h
        loss = loss_rec + loss_adv
        chainer.report({'loss': loss}, enc)
        return loss

    def loss_dec(self, dec, x_out, t_out, y_out, lam1=100, lam2=1):
        batchsize, _, w, h = y_out.data.shape
        loss_rec = lam1 * (F.mean_absolute_error(x_out, t_out))
        loss_adv = lam2 * F.sum(F.softplus(-y_out)) / batchsize / w / h
        loss = loss_rec + loss_adv
        chainer.report({'loss': loss}, dec)
        return loss

    def loss_dis(self, dis, y_in, y_out):
        batchsize, _, w, h = y_in.data.shape

        L1 = F.sum(F.softplus(-y_in)) / batchsize / w / h
        L2 = F.sum(F.softplus(y_out)) / batchsize / w / h
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        gen, dis = self.gen, self.dis
        xp = gen.xp
        lam1 = self.lam1
        lam2 = self.lam2

        batch = self.get_iterator('main').next()

        x_real, t_real = zip(*batch)
        x_real = Variable(xp.asarray(x_real, dtype=xp.float32))

        t_real = xp.asarray(t_real, dtype=xp.float32)

        t_fake = gen(x_real)

        y_fake = dis(x_real, t_fake)
        y_real = dis(x_real, t_real)

        loss_gen = lam1 * F.mean_absolute_error(t_fake, t_real)
        chainer.report({'gen/loss_l1': loss_gen})

        loss_gen += lam2 * \
            F.sigmoid_cross_entropy(y_fake, xp.ones_like(
                y_fake.data).astype(xp.int8))

        gen.cleargrads()
        loss_gen.backward()
        gen_optimizer.update()
        chainer.report({'gen/loss_total': loss_gen})

        x_real.unchain_backward()
        t_fake.unchain_backward()

        loss_dis = F.sigmoid_cross_entropy(
            y_fake, xp.zeros_like(y_fake.data).astype(xp.int8))
        loss_dis += F.sigmoid_cross_entropy(
            y_real, xp.ones_like(y_real.data).astype(xp.int8))

        dis.cleargrads()
        loss_dis.backward()
        dis_optimizer.update()

        chainer.report({'dis/loss': loss_dis})
