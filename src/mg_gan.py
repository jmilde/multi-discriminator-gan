try:
    from util import Record, identity, comp, partial
    from util_tf import tf, scope, variable, placeholder, Linear, Affine, Normalize, spread_image
except ImportError:
    from src.util import Record, identity, comp, partial
    from src.util_tf import tf, scope, variable, placeholder, Linear, Affine, Normalize, spread_image

class Gen(Record):

    def __init__(self, dim_x, dim_btlnk, name= 'generator'):
        self.name = name
        with scope(name):
            self.lin = Linear(dim_btlnk, dim_x, name= 'lin')
            self.nrm = Normalize(    dim_btlnk, name= 'nrm')
            self.lex = Linear(dim_x, dim_btlnk, name= 'lex')

    def __call__(self, x, name= None):
        with scope(name or self.name):
            #return tf.clip_by_value(self.lex(self.nrm(tf.nn.relu(self.lin(x)))), 0.0, 1.0)
            return tf.nn.sigmoid(self.lex(self.nrm(tf.nn.relu(self.lin(x)))))

class Dis(Record):

    def __init__(self, dim_x, dim_d, name= 'discriminator'):
        self.name = name
        with scope(name):
            self.lin = Linear(dim_d, dim_x, name= 'lin')
            self.nrm = Normalize(    dim_d, name= 'nrm')
            self.lin2 = Linear(dim_d, dim_d, name= 'lin2')
            self.nrm2 = Normalize(    dim_d, name= 'nrm2')
            self.lex = Linear(1, dim_d, name= 'lex')

    def __call__(self, x, name= None):
        with scope(name or self.name):
            x = self.nrm(tf.nn.leaky_relu(self.lin(x)))
            x = self.nrm2(tf.nn.leaky_relu(self.lin2(x)))
            return tf.clip_by_value(self.lex(x), 0.0, 1.0)


class MG_GAN(Record):

    @staticmethod
    def new(dim_x, dim_btlnk, dim_d, n_dis):
        return MG_GAN(dim_x= dim_x
                      , gen= Gen(dim_x, dim_btlnk)
                      , dis= {n:Dis(dim_x, dim_d, name=f"discriminator_{n}") for n in range(n_dis)})

    def build(self, x, y, weight):
        with scope("x"):
            x = placeholder(tf.float32, [None, self.dim_x], x, "x")
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

            # mean feedback
            d_loss = tf.reduce_mean(d_loss_real) + tf.reduce_mean(d_loss_fake)

            epsilon = 1e-10
            loss_rec = tf.reduce_mean(-tf.reduce_sum(x * tf.log(epsilon+gx) +
                                                     (1-x) * tf.log(epsilon+1-gx),  axis=1))
            loss_g_fake = [tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dgx_), logits=dgx_)) for dgx_ in dgx.values()]

            # mean feedback
            g_loss = weight* loss_rec + tf.reduce_mean(loss_g_fake)

        with scope("AUC"):
            _, auc_dgx = tf.metrics.auc(y, tf.nn.sigmoid(tf.reduce_mean(list(dgx.values()))))
            _, auc_dx = tf.metrics.auc(y, tf.nn.sigmoid(tf.reduce_mean(list(dx.values()))))
            _, auc_gx = tf.metrics.auc(y, tf.reduce_mean((x-gx)**2, axis=1))

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
                     , auc_dgx=auc_dgx
                     , auc_gx=auc_gx
                     , auc_dx=auc_dx
                     , g_step=g_step
                     , d_step=d_step
                     , g_loss=g_loss
                     , d_loss=d_loss)
