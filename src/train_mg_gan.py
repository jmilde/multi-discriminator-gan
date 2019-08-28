from tqdm import tqdm
from numpy.random import RandomState
import os
from PIL import Image
try:
    from util_np import np, batch_sample, partition
    from util_tf import pipe, tf, spread_image, batch2
    from util_io import pform
    from models.mg_gan import MG_GAN
except ImportError:
    from src.util_np import np, batch_sample,unison_shfl, partition
    from src.util_tf import pipe, tf, spread_image, batch2
    from src.util_io import pform
    from src.mg_gan import MG_GAN

def prod(iterable):
    from functools import reduce
    from operator import mul
    return reduce(mul, iterable, 1)

def resize_images(imgs, size=[32, 32]):
    # convert float type to integer
    resized_imgs = np.asarray([np.asarray(Image.fromarray(img).resize(size=size, resample=Image.ANTIALIAS))
                                       for i, img in enumerate(imgs.astype('uint8'))])

    return np.expand_dims(resized_imgs, -1)

def train(anomaly_class = 8, dataset="cifar", n_dis=1, epochs=25, dim_btlnk=32,
          batch_size=64, loss="mean", context_weight=1, dim_d=64, dim_g=64, extra_layers=0, gpu="0"):

    #set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    path_log = f"/cache/tensorboard-logdir/{dataset}"
    path_ckpt = "/project/multi-discriminator-gan/ckpt"
    path_data = "/project/multi-discriminator-gan/data"

    #reset graphs and fix seeds
    tf.reset_default_graph()
    if 'sess' in globals(): sess.close()
    rand = RandomState(0)
    tf.set_random_seed(0)

    #load data
    if dataset=="ucsd1":
        x_train = np.load("./data/ucsd1_train_x.npz")["arr_0"]/255
        y_train = np.load("./data/ucsd1_train_y.npz")["arr_0"]
        x_test = np.load("./data/ucsd1_test_x.npz")["arr_0"]/255
        y_test = np.load("./data/ucsd1_test_y.npz")["arr_0"]

    elif dataset=="uscd2":
        x_train = np.load("./data/ucsd2_train_x.npz")["arr_0"]
        y_train = np.load("./data/ucsd2_train_y.npz")["arr_0"]
        x_test = np.load("./data/ucsd2_test_x.npz")["arr_0"]
        y_test = np.load("./data/ucsd2_test_y.npz")["arr_0"]

    else:
        if dataset=="mnist":
            (train_images, train_labels),(test_images, test_labels) = tf.keras.datasets.mnist.load_data()
            train_images = resize_images(train_images)
            test_images = resize_images(test_images)
        else:
            (train_images, train_labels),(test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
            train_labels = np.reshape(train_labels, len(train_labels))
            test_labels = np.reshape(test_labels, len(test_labels))

        inlier = train_images[train_labels!=anomaly_class]
        #data_size = prod(inlier[0].sha
        x_train = inlier/255
        #x_train = np.reshape(inlier, (len(inlier), data_size))/255
        #y_train = train_labels[train_labels!=anomaly_class]
        y_train = np.zeros(len(x_train), dtype=np.int8) # dummy
        outlier = train_images[train_labels==anomaly_class]
        x_test = np.concatenate([outlier, test_images])/255
        #x_test = np.reshape(np.concatenate([outlier, test_images])
        #                    ,(len(outlier)+len(test_images), data_size))/255
        y_test= np.concatenate([train_labels[train_labels==anomaly_class], test_labels])
        y_test = [0 if y!=anomaly_class else 1 for y in y_test]
        x_test, y_test = unison_shfl(x_test, np.array(y_test))

    img_size_x= x_train[0].shape[0]
    img_size_y= x_train[0].shape[1]
    channel = x_train[0].shape[-1]
    trial = f"{dataset}_{loss}_dis{n_dis}_{anomaly_class}_w{context_weight}_btlnk{dim_btlnk}_d{dim_d}_g{dim_g}e{extra_layers}"




     # data pipeline
    batch_fn = lambda: batch2(x_train, y_train, batch_size)
    x, y = pipe(batch_fn, (tf.float32, tf.float32), prefetch=4)
    #z = tf.random_normal((batch_size, z_dim))

    # load graph
    mg_gan = MG_GAN.new(img_size_x, channel, dim_btlnk, dim_d, dim_g, n_dis, extra_layers=0)
    model = MG_GAN.build(mg_gan, x, y, context_weight, loss)


    # start session, initialize variables

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    wrtr = tf.summary.FileWriter(pform(path_log, trial))
    wrtr.add_graph(sess.graph)

    ### if load pretrained model
    # pretrain = "modelname"
    #saver.restore(sess, pform(path_ckpt, pretrain))
    ### else:
    auc_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='AUC')
    init = tf.group(tf.global_variables_initializer(), tf.variables_initializer(var_list=auc_vars))
    sess.run(init)


    #if "ucsd" in dataset:
    summary_test = tf.summary.merge([tf.summary.scalar('g_loss', model.g_loss)
                                         , tf.summary.scalar("lambda", model.lam)
                                         , tf.summary.scalar('d_loss_mean', tf.reduce_mean(model.d_loss))
                                         #, tf.summary.scalar('d_loss', model.d_loss)
                                         , tf.summary.scalar("AUC_gx", model.auc_gx)])
    if dataset=="ucsd1":
        summary_images = tf.summary.merge((tf.summary.image("gx", model.gx, max_outputs=8)
                                               , tf.summary.image("x", model.x, max_outputs=8)
                                               , tf.summary.image('gx400', spread_image(tf.concat([model.gx, model.x], axis=1), 8,2, img_size_x, img_size_y, channel))))
    else:
        summary_images = tf.summary.merge((tf.summary.image("gx", model.gx, max_outputs=8)
                                               , tf.summary.image('gx400', spread_image(model.gx[:400], 20,20, img_size_x, img_size_y, channel))
                                               , tf.summary.image("x", model.x, max_outputs=8)))


    if n_dis>1:
        d_wrtr = {i: tf.summary.FileWriter(pform(path_log, trial+f"d{i}"))
                  for i in range(n_dis)}
        summary_discr= {i: tf.summary.scalar('d_loss_multi', model.d_loss[i])
                        for i in range(n_dis)}

    def summ(step):
        fetches = model.g_loss, model.lam, tf.reduce_mean(model.d_loss), model.auc_gx
        results = map(np.mean, zip(*(
            sess.run(fetches,
                     {model['x']: x_test[i:j]
                      , model['y']: y_test[i:j]})
            for i, j in partition(len(x_test), batch_size, discard=False))))
        results = list(results)
        wrtr.add_summary(sess.run(summary_test, dict(zip(fetches, results))), step)

        if dataset=="ucsd1":
            # bike, skateboard, grasswalk, shopping cart, car, normal, normal, grass
            wrtr.add_summary(sess.run(summary_images, {model.x:x_test[[990, 1851, 2140, 2500, 2780, 2880, 3380, 3580]]}), step)
        else:
            wrtr.add_summary(sess.run(summary_images, {model.x:x_test}), step)
        wrtr.flush()

    def summ_discr(step):
        fetches = model.d_loss
        results = map(np.mean, zip(*(
            sess.run(fetches,
                     {model['x']: x_test[i:j]
                      , model['y']: y_test[i:j]})
            for i, j in partition(len(x_test), batch_size, discard=False))))
        results = list(results)
        if n_dis>1: # put all losses of the discriminators in one plot
            for i in range(n_dis):
                d_wrtr[i].add_summary(sess.run(summary_discr[i], dict(zip(fetches, results))), step)
                #d_wrtr[i].add_summary(sess.run(summary_discr[i], dict([(fetches[i], results[i])])), step)
                d_wrtr[i].flush()

    #def log(step
    #        , wrtr= wrtr
    #        , log = tf.summary.merge([tf.summary.scalar('g_loss', model.g_loss)
    #                                  , tf.summary.scalar('d_loss', tf.reduce_mean(model.d_loss))
    #                                  , tf.summary.scalar("lambda", model.lam)
    #                                  , tf.summary.image("gx", model.gx, max_outputs=5)
    #                                  , tf.summary.image('gx400', spread_image(model.gx[:400], 20,20, img_size, img_size, channel))
    #                                  #, tf.summary.scalar("AUC_dgx", model.auc_dgx)
    #                                  #, tf.summary.scalar("AUC_dx", model.auc_dx)
    #                                  , tf.summary.scalar("AUC_gx", model.auc_gx)])
    #        , y= y_test
    #        , x= x_test):
    #    wrtr.add_summary(sess.run(log, {model["x"]:x
    #                                    , model["y"]:y})
    #                     , step)
    #    wrtr.flush()


    steps_per_epoch = len(x_train)//batch_size-1
    for epoch in tqdm(range(epochs)):
        for i in range(steps_per_epoch):
            #sess.run(model["train_step"])
            sess.run(model['d_step'])
            sess.run(model['g_step'])
        # tensorboard writer
        #if "ucsd" in dataset:
        summ(sess.run(model["step"])//steps_per_epoch)
        #else:
        #    log(sess.run(model["step"])//steps_per_epoch)
        #if n_dis>1:
        #    summ_discr(sess.run(model["step"])//steps_per_epoch)

    saver.save(sess, pform(path_ckpt, trial), write_meta_graph=False)


if __name__ == "__main__":
    ###########################
    run="2u3" #"2u3" # "basic", "4u5"
    ###########################
    btlnk_dim = 32
    batch_size = 32
    extra_layers = 0
    w=1


    if run=="basic":
        gpu="0"
        for w in [1, 0.1, 5]:
            for method in ["mean"]: #:, "max", "softmax"]:  #"softmax_self_challenged"
                for n in [1]: # range(2,4):
                    for dim in [64]: #dim of encoder/decoder
                        for d in ["mnist", "cifar", "ucsd1"]:

                            if d=="cifar":
                                epoch=25
                                for i in range(0,10):
                                    train(i, d, n, epoch, btlnk_dim, batch_size,
                                          method, w, dim, dim, extra_layers, gpu=gpu)
                            if d=="ucsd1":
                                epoch=100
                                train(0, d, n, epoch, btlnk_dim, batch_size,
                                      method, w, dim, dim, extra_layers, gpu=gpu) #0=dummy

                            if d=="mnist":
                                epoch=15
                                for i in range(0,10):
                                    train(i, d, n, epoch, btlnk_dim, batch_size,
                                          method, w, dim, dim, extra_layers, gpu=gpu)


    elif run=="2u3":
        gpu="1"
        for w in [1, 0.1, 5]:
            for method in ["mean", "max", "softmax"]:  #"softmax_self_challenged"
                for n in range(2,4):
                    for dim in [64]: #dim of encoder/decoder
                        for d in ["ucsd1", "mnist", "cifar"]:

                            if d=="cifar":
                                epoch=25
                                for i in range(0,10):
                                    train(i, d, n, epoch, btlnk_dim, batch_size,
                                          method, w, dim, dim, extra_layers, gpu=gpu)
                            if d=="ucsd1":
                                epoch=100
                                train(0, d, n, epoch, btlnk_dim, batch_size,
                                      method, w, dim, dim, extra_layers, gpu=gpu) #0=dummy

                            if d=="mnist":
                                epoch=15
                                for i in range(0,10):
                                    train(i, d, n, epoch, btlnk_dim, batch_size,
                                          method, w, dim, dim, extra_layers, gpu=gpu)
    elif run=="4u5":
        gpu="2"
        for w in [1, 0.1, 5]:#, 0.5, 2]:
            for method in ["max", "mean", "softmax"]:  #"softmax_self_challenged"
                for n in range(4,6):
                    for dim in [64]: #dim of encoder/decoder
                        for d in ["ucsd1", "mnist", "cifar"]:
                            if d=="cifar":
                                epoch=25
                                for i in range(0,10):
                                    train(i, d, n, epoch, btlnk_dim, batch_size,
                                          method, w, dim, dim, extra_layers, gpu=gpu)
                            if d=="ucsd1":
                                epoch=100
                                train(0, d, n, epoch, btlnk_dim, batch_size,
                                      method, w, dim, dim, extra_layers, gpu=gpu) #0=dummy

                            if d=="mnist":
                                epoch=15
                                for i in range(0,10):
                                    train(i, d, n, epoch, btlnk_dim, batch_size,
                                          method, w, dim, dim, extra_layers, gpu=gpu)
