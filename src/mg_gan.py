try:
    from util import Record, identity, comp, partial
    from util_tf import tf, scope, variable, placeholder, Linear, Affine, Normalize, spread_image
except ImportError:
    from src.util import Record, identity, comp, partial
    from src.util_tf import tf, scope, variable, placeholder, Linear, Affine, Normalize, spread_image
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Conv2D

class Decoder(Record):
    def __init__(self, dim_x, channel_x, dim_d, name="decoder"):
        assert dim_x % 16 == 0, "image size has to be a multiple of 16"
        with scope(name):
            cngf, tisize = dim_d // 2, 4 # first n_filter=4,
            while tisize != dim_x:
                cngf = cngf * 2
                tisize = tisize * 2

            self.conv0 = Conv2DTranspose(cngf, (4, 4), padding='valid', use_bias=False , name="conv0")
            self.bn0 = BatchNormalization(name="bn0")

            size_now, i = 4, 1
            self.conv, self.bn = {}, {}
            while size_now < dim_x // 2:
                self.conv[i]= Conv2DTranspose(cngf//2, (4, 4), strides=2,
                                               padding='same', use_bias=False, name=f"conv{i}" )
                self.bn[i] = BatchNormalization(name=f"bn{i}")
                cngf = cngf // 2
                size_now = size_now*2
                i+=1

            # final layer, expand the size with 2 and set channels to number of channels of x
            self.conv_out = Conv2DTranspose(channel_x, (4, 4),strides=2, padding='same', use_bias=False, name="conv_out" )

    def __call__(self, x):
        # z is input, and first deconvolution layer to size channel * 4 * 4
        x = tf.nn.relu(self.bn0(self.conv0(x)))

        # size increasing layers
        for i in sorted(self.conv.keys()):
            x = tf.nn.relu(self.bn[i](self.conv[i](x)))
        x = tf.nn.sigmoid(self.conv_out(x))
        return x



class Encoder(Record):
    def __init__(self, dim_x, channel_x, dim_g, name="encoder"):
        assert dim_x % 16 == 0, "image size has to be a multiple of 16"

        with scope(name):
            self.conv0 = Conv2D(dim_g, (4, 4), strides=2, padding='same', use_bias=False, name="conv0")
            size_now = dim_x // 2
            channel = dim_g
            i = 1
            self.conv, self.bn = {}, {}
            while size_now > 4:
                channel *= 2 # channel increases, size decreases
                self.conv[i] = Conv2D(channel, (4, 4), strides=2, padding='same', use_bias=False, name= f"conv{i}")
                self.bn[i] = BatchNormalization(name=f"bn{i}")
                size_now = size_now // 2
                i += 1

    def __call__(self, x):

        # initial layer
        x = tf.nn.leaky_relu(self.conv0(x))

        # in between layers
        for i in sorted(self.conv.keys()):
            x = tf.nn.leaky_relu(self.bn[i](self.conv[i](x)))


        return x



class Gen(Record):

    def __init__(self, dim_x, channel_x, dim_btlnk, dim_d, dim_g, name= 'generator'):
        self.name = name
        with scope(name):
            self.enc = Encoder(dim_x, channel_x, dim_g)
            #resize the layer to channel X 1 X 1
            self.conv_out = Conv2D(dim_btlnk, (4, 4), padding='valid', use_bias=False, name="conv_out")
            self.dec = Decoder(dim_x, channel_x, dim_d)


    def __call__(self, x, name= None):
        with scope(name or self.name):
            x = self.enc(x)
            # final layer
            x = self.conv_out(x)
            x = self.dec(x)

            return x



class Dis(Record):

    def __init__(self, dim_x, channel_x, dim_g, name= 'discriminator'):
        self.name = name
        with scope(name):
            self.enc = Encoder(dim_x, channel_x, dim_g)
            self.conv_out = Conv2D(1, (4, 4), padding='valid', use_bias=False, name="conv_out")

    def __call__(self, x, name= None):
        with scope(name or self.name):
            x = self.enc(x)
            x = self.conv_out(x)
            return tf.nn.sigmoid(x)




class MG_GAN(Record):
    @staticmethod
    def new(dim_x, channel_x, dim_btlnk, dim_d, dim_g,  n_dis):
        return MG_GAN(dim_x= dim_x
                      , channel_x= channel_x
                      , gen= Gen(dim_x, channel_x, dim_btlnk, dim_d, dim_g)
                      , dis= {n:Dis(dim_x, channel_x, dim_g, name=f"discriminator_{n}") for n in range(n_dis)})

    def build(self, x, y, weight, loss):
        with scope("x"):
            x = placeholder(tf.float32, [None, None, None, self.channel_x], x, "x")
        with scope("y"):
            y = placeholder(tf.float32, [None], y, "y")

        gx = self.gen(x)
        dx = {k:v(x) for k,v in self.dis.items()}
        dgx = {k:v(gx) for k,v in self.dis.items()}
        #dx, dgx = self.dis(x), self.dis(gx)

        with scope("loss"):
            d_loss_real, d_loss_fake = [], []
            for k in dx.keys():
                d_loss_real.append(tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dx[k])*0.9, logits=dx[k])))
                d_loss_fake.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(dgx[k]), logits=dgx[k])))

            if loss=="mean":
                d_loss = tf.reduce_mean(d_loss_real) + tf.reduce_mean(d_loss_fake)
            elif loss=="max":
                d_loss = tf.reduce_mean(d_loss_real) + tf.reduce_mean(d_loss_fake)
            elif loss=="softmax":
                d_loss = tf.reduce_mean(d_loss_real) + tf.reduce_mean(d_loss_fake)

            epsilon = 1e-10
            loss_rec = tf.reduce_mean(-tf.reduce_sum(x * tf.log(epsilon+gx) +
                                                     (1-x) * tf.log(epsilon+1-gx),  axis=1))
            loss_g_fake = [tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dgx_), logits=dgx_)) for dgx_ in dgx.values()]

            if loss=="mean":
                g_loss = weight* loss_rec + tf.reduce_mean(loss_g_fake)
            elif loss=="max": # max picks biggest loss = best discriminators feedback is used
                g_loss = weight* loss_rec + tf.reduce_max(loss_g_fake)
            elif loss="softmax":
                g_loss = weight* loss_rec + tf.reduce_mean(tf.nn.softmax(loss_g_fake)*loss_g_fake)



        with scope("AUC"):
            #_, auc_dgx = tf.metrics.auc(y, tf.nn.sigmoid(tf.reduce_mean(list(dgx.values()))))
            #_, auc_dx = tf.metrics.auc(y, tf.nn.sigmoid(tf.reduce_mean(list(dx.values()))))
            _, auc_gx = tf.metrics.auc(y, tf.reduce_mean((x-gx)**2, axis=(1,2,3)))

        g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
        d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")

        with scope('train_step'):
            step = tf.train.get_or_create_global_step()
            optimizer = tf.train.AdamOptimizer()
            d_step = optimizer.minimize(d_loss, step, var_list=d_vars)
            g_step = optimizer.minimize(g_loss, step, var_list=g_vars)


        return MG_GAN(self
                     , step=step
                     , x=x
                     , y=y
                     , gx=gx
                     #, auc_dgx=auc_dgx
                     , auc_gx=auc_gx
                     #, auc_dx=auc_dx
                     , g_step=g_step
                     , d_step=d_step
                     , g_loss=g_loss
                     , d_loss=d_loss)
