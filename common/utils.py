import os
import chainer
import numpy as np
import chainer.cuda
from PIL import Image


def check_and_make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def out_generated_image(gen, dst, test_iter, seed=0, from_gaussian=False):
    @chainer.training.make_extension()
    def make_image(trainer):
        data_length = len(test_iter.dataset)
        rows = int(np.sqrt(data_length))
        cols = int(np.sqrt(data_length))

        n_images = rows * cols

        xp = gen.xp
        # batchsize = updater.get_iterator("test").batch_size
        gen_image = []
        for batch in test_iter:
            # batch = updater.get_iterator("test").next()
            batch = xp.asarray(batch, dtype=xp.float32)
            t_fake = gen(batch)
            t_fake = chainer.cuda.to_cpu(t_fake.data)
            gen_image.append(t_fake)

        test_iter.current_position = 0
        test_iter.epoch = 0

        gen_image = np.concatenate(gen_image, axis=0)
        gen_image = gen_image[:n_images]

        def save_figure(x, file_name="image", mode=None):
            file_name += "_iteration:{:0>6}.png".format(
                trainer.updater.iteration)
            preview_path = os.path.join(dst, file_name)
            _, C, H, W = x.shape
            x = x.reshape((rows, cols, C, H, W))
            x = x.transpose(0, 3, 1, 4, 2)
            if C == 1:
                x = x.reshape((rows * H, cols * W))
            else:
                x = x.reshape((rows * H, cols * W, C))
            Image.fromarray(x, mode=mode).convert("RGB").save(preview_path)
        # gen output_activation_func is tanh (-1 ~ 1)
        gen_image = np.asarray(
            np.clip((gen_image * 0.5 + 0.5) * 255.0, 0, 255.), dtype=np.uint8)

        save_figure(gen_image, file_name="image")

    return make_image
