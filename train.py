#!usr/bin/python
# -*- coding: UTF-8 -*-

import chainer
import argparse
import os
import time
import common.dataset
import common.utils
from chainer import training
from chainer.training import extensions


DATASET_LIST = ["facades"]


def make_optimizer(model, alpha=0.0002, beta1=0.5):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
    optimizer.setup(model)
    return optimizer


def check_gpu(model, gpu_id):
    if gpu_id >= 0:
        chainer.cuda.get_device(gpu_id).use()
        for m in model:
            m.to_gpu(gpu_id)


def main():
    start = time.time()
    out_folder = os.path.join("./result", args.method, args.dataset)
    if os.path.isdir(out_folder):
        new_folder = "folder_{}".format(len(os.listdir(out_folder)))
    else:
        new_folder = "folder_0"
    out_folder = os.path.join(out_folder, new_folder)
    if args.no_debug:
        max_time = (args.max_iteration, "iteration")
    else:
        max_time = (1000, "iteration")
    if args.dataset == "facades":
        train = common.dataset.FacadeDataset()
        valid = common.dataset.FacadeValidDataset(data_range=(300, 379))
        train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
        valid_iter = chainer.iterators.SerialIterator(
            valid, args.batchsize, shuffle=False, repeat=False)
    updater_args = {"iterator": train_iter}
    if args.method == "pix2pix":
        from pix2pix.updater import FacadeUpdater as Updater
        import pix2pix.net
        gen = pix2pix.net.Generator(in_channel=12, out_channel=3)
        dis = pix2pix.net.Discriminator(in_channel=12, out_channel=3)
        check_gpu((gen, dis), args.gpu_id)
        gen_optimizer = make_optimizer(gen, alpha=args.lr, beta1=args.beta1)
        dis_optimizer = make_optimizer(dis)
        updater_args["lam"] = {"lam1": args.lam1, "lam2": args.lam2}

        plot_report = ["gen/loss_total", "dis/loss"]
        print_report = ["gen/loss_l1"] + plot_report
    else:
        raise NotImplementedError
    model = {"gen": gen, "dis": dis}
    optimizer = {"gen": gen_optimizer, "dis": dis_optimizer}
    updater_args["models"] = model
    updater_args["optimizer"] = optimizer
    updater = Updater(**updater_args)

    # epoch_interval = (1, 'epoch')
    save_snapshot_interval = (10000, "iteration")
    plot_interval = (100, "iteration")
    display_interval = (100, "iteration")

    trainer = training.Trainer(updater, stop_trigger=max_time, out=out_folder)
    common.utils.check_and_make_dir(out_folder)
    out_image_folder = os.path.join(out_folder, "preview")
    common.utils.check_and_make_dir(out_image_folder)

    if args.snapshot:
        trainer.extend(extensions.snapshot_object(
            gen, args.method + '_gen_iteration_{.updater.iteration}.npz'), trigger=save_snapshot_interval)
        trainer.extend(extensions.snapshot_object(
            dis, args.method + '_dis_iteration_{.updater.iteration}.npz'), trigger=save_snapshot_interval)
    trainer.extend(extensions.dump_graph(root_name="gen/loss_total", out_name="cg.dot"))
    trainer.extend(extensions.PlotReport(
        plot_report, x_key='iteration', file_name='loss.png', trigger=plot_interval))
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'elapsed_time'] + print_report), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(common.utils.out_generated_image(
        gen, out_image_folder, valid_iter), trigger=display_interval)

    print("finish setup training environment in {} s".format(time.time() - start))
    print("start training ...")
    # Run the training
    trainer.run()


parser = argparse.ArgumentParser(
    description="This file is used to train semi-supervised model")
parser.add_argument("-m", "--method",
                    help="method for semi supervised learning.",
                    default="pix2pix")
parser.add_argument("--dataset",
                    help="dataset to train",
                    default="facades",
                    choices=DATASET_LIST)
parser.add_argument("--batchsize",
                    help="batchsize used in test/valid data",
                    type=int,
                    default=10)
parser.add_argument("--lr", help="learning rate", type=float, default=0.0002)
parser.add_argument("--beta1",
                    help=" value of beta1 used in former training",
                    type=float,
                    default=0.5)
parser.add_argument("--lam1",
                    help="weight of l1 loss",
                    type=int,
                    default=100)
parser.add_argument("--lam2",
                    help="weight of conditional loss",
                    type=int,
                    default=1)
parser.add_argument("-ndb", "--no_debug",
                    help="flag if not debug, default is False", action="store_true")
parser.add_argument("--max_iteration",
                    help="max iteration", type=int, default=1000)
parser.add_argument(
    "--snapshot", help="falg to save snapshot", action="store_true")
parser.add_argument(
    "--gpu_id", help="id of gpu", type=int, default=0)

args = parser.parse_args()


if __name__ == "__main__":
    main()
