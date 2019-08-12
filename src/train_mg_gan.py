from tqdm import tqdm
from numpy.random import RandomState
import os

try:
    from util_np import np, batch_sample
    from util_tf import pipe, tf, spread_image, batch2
    from util_io import pform
    from models.mg_gan import MG_GAN
except ImportError:
    from src.util_np import np, batch_sample,unison_shfl
    from src.util_tf import pipe, tf, spread_image, batch2
    from src.util_io import pform
    from src.mg_gan import MG_GAN

def prod(iterable):
    from functools import reduce
    from operator import mul
    return reduce(mul, iterable, 1)

def train(anomaly_class = 8, dataset="cifar"):
    #set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    path_log = "/cache/tensorboard-logdir/mg"
    path_ckpt = "/project/multi-discriminator-gan/ckpt"
    path_data = "/project/multi-discriminator-gan/data"

    epochs = 400
    batch_size = 700
    dim_btlnk = 32
    dim_dense = 64
    n_dis = 5
    context_weight = 1

    #reset graphs and fix seeds
    tf.reset_default_graph()
    if 'sess' in globals(): sess.close()
    rand = RandomState(0)
    tf.set_random_seed(0)

    #load data
    if dataset="ucsd": #todo
        folders = os.listdir(datapath)
    else:
        if dataset=="mnist":
            (train_images, train_labels),(test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        else:
            (train_images, train_labels),(test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
            train_labels = np.reshape(train_labels, len(train_labels))
            test_labels = np.reshape(test_labels, len(test_labels))

        inlier = train_images[train_labels!=anomaly_class]
        data_size = prod(inlier[0].shape)
        img_size= inlier[0].shape[0]
        color_size = inlier[0].shape[-1]
        x_train = np.reshape(inlier, (len(inlier), data_size))/255
        #y_train = train_labels[train_labels!=anomaly_class]
        y_train = np.zeros(len(x_train), dtype=np.int8) # dummy
        outlier = train_images[train_labels==anomaly_class]
        x_test = np.reshape(np.concatenate([outlier, test_images])
                            ,(len(outlier)+len(test_images), data_size))/255
        y_test= np.concatenate([train_labels[train_labels==anomaly_class], test_labels])
        y_test = [0 if y!=anomaly_class else 1 for y in y_test]
        x_test, y_test = unison_shfl(x_test, np.array(y_test))


    dim_x = len(x_train[0])
    trial = f"{dataset}_mean_dis{n_dis}_{anomaly_class}_b{batch_size}_btlnk{dim_btlnk}_d{dim_dense}"




     # data pipeline
    batch_fn = lambda: batch2(x_train, y_train, batch_size)
    x, y = pipe(batch_fn, (tf.float32, tf.float32), prefetch=4)
    #z = tf.random_normal((batch_size, z_dim))

    # load graph
    mg_gan = MG_GAN.new(dim_x, dim_btlnk, dim_dense, n_dis)
    model = MG_GAN.build(mg_gan, x, y, context_weight)


    # start session, initialize variables

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    wrtr = tf.summary.FileWriter(pform(path_log, trial))
    #wrtr.add_graph(sess.graph)

    ### if load pretrained model
    # pretrain = "modelname"
    #saver.restore(sess, pform(path_ckpt, pretrain))
    ### else:
    auc_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='AUC')
    init = tf.group(tf.global_variables_initializer(), tf.variables_initializer(var_list=auc_vars))
    sess.run(init)

    def log(step
            , wrtr= wrtr
            , log = tf.summary.merge([tf.summary.scalar('g_loss', model.g_loss)
                                      , tf.summary.scalar('d_loss', model.d_loss)
                                      , tf.summary.image('gx400', spread_image(model.gx[:400], 20,20, img_size, img_size, color_size))
                                      #, tf.summary.scalar("AUC_dgx", model.auc_dgx)
                                      #, tf.summary.scalar("AUC_dx", model.auc_dx)
                                      , tf.summary.scalar("AUC_gx", model.auc_gx)])
            , y= y_test
            , x= x_test):
        wrtr.add_summary(sess.run(log, {model["x"]:x
                                        , model["y"]:y})
                         , step)
        wrtr.flush()


    steps_per_epoch = len(x_train)//batch_size-1
    for epoch in tqdm(range(epochs)):
        for i in range(steps_per_epoch):
            #sess.run(model["train_step"])
            sess.run(model['d_step'])
            sess.run(model['g_step'])
        # tensorboard writer
        log(sess.run(model["step"])//steps_per_epoch)

    saver.save(sess, pform(path_ckpt, trial), write_meta_graph=False)


if __name__ == "__main__":
    for i in range(0,10):
        train(i)
