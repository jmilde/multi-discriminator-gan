import tensorflow as tf
try:
    from src.util_np import np, sample, batch_sample
except ImportError:
    from util_np import np, batch_sample, sample


def pipe(*args, prefetch=1, repeat=-1, name='pipe', **kwargs):
    """see `tf.data.Dataset.from_generator`."""
    with tf.variable_scope(name):
        return tf.data.Dataset.from_generator(*args, **kwargs) \
                              .repeat(repeat) \
                              .prefetch(prefetch) \
                              .make_one_shot_iterator() \
                              .get_next()


def batch3(x, y, size, oh_size, noise_size, seed=25):
    """batch function to use with pipe, takes to numpy labels as input"""
    b, l = [],[]
    for i in sample(len(x), seed):
        if size == len(b):
            z = np.random.normal(size=(size, noise_size))
            yield b, l, z
            b, l = [], []
        b.append(x[i])
        oh = np.zeros((oh_size))
        oh[y[i]]=1
        l.append(oh)

def batch2(x, y, size, noise_size=0, seed=25):
    """batch function to use with pipe, takes to numpy labels as input"""
    for i in batch_sample(len(x), size):
        if noise_size==0:
            yield x[i], y[i]
        else:
            yield x[i], y[i], np.random.normal(size=(size, noise_size))


def batch(size, noise_size, path_data, seed=25):
    """batch function to use with pipe, takes to numpy labels as input"""
    data = np.load(path_data)
    b = []
    for i in sample(len(data), seed):
        if size == len(b):
            yield b
            b = []
        #normalize = (data[i]-127.5)/127.5
        normalize = data[i]/255
        b.append(normalize)

def placeholder(dtype, shape, x= None, name= None):
    """returns a placeholder with `dtype` and `shape`.

    if tensor `x` is given, converts and uses it as default.

    """
    if x is None: return tf.placeholder(dtype, shape, name)
    try:
        x = tf.convert_to_tensor(x, dtype)
    except ValueError:
        x = tf.cast(x, dtype)
    return tf.placeholder_with_default(x, shape, name)

def profile(sess, wrtr, run, feed_dict= None, prerun= 5, tag= 'flow'):
    for _ in range(prerun): sess.run(run, feed_dict)
    meta = tf.RunMetadata()
    sess.run(run, feed_dict, tf.RunOptions(trace_level= tf.RunOptions.FULL_TRACE), meta)
    wrtr.add_run_metadata(meta, tag)



def variable(name, shape, init= 'rand',
             initializers={'zero': tf.initializers.zeros(),
                           'unit': tf.initializers.ones(),
                           'rand': tf.glorot_uniform_initializer()}):
    """wraps `tf.get_variable` to provide initializer based on usage"""
    return tf.get_variable(name, shape, initializer= initializers.get(init, init))


def normalize(x, name="layer_norm"):
    with tf.variable_scope(name):
        dim = x.shape[-1]
        gain = variable('gain', (1, dim), 'unit')
        bias = variable('bias', (1, dim), 'zero')
        mean, var = tf.nn.moments(x, 1, keep_dims= True)
        return (x - mean) * tf.rsqrt(var + 1e-12) * gain + bias


def spread_image(x, nrow, ncol, height, width):
    return tf.reshape(
        tf.transpose(
            tf.reshape(x, (nrow, ncol, height, width, -1))
            , (0, 2, 1, 3, 4))
        , (1, nrow * height, ncol * width, -1))

########################################################
try:
    from util import Record
except ImportError:
    from src.util import Record
import tensorflow as tf

scope = tf.variable_scope


def profile(sess, wtr, run, feed_dict= None, prerun= 3, tag= 'flow'):
    for _ in range(prerun): sess.run(run, feed_dict)
    meta = tf.RunMetadata()
    sess.run(run, feed_dict, tf.RunOptions(trace_level= tf.RunOptions.FULL_TRACE), meta)
    wtr.add_run_metadata(meta, tag)


def pipe(*args, prefetch= 1, repeat= -1, name= 'pipe', **kwargs):
    """see `tf.data.Dataset.from_generator`"""
    with scope(name):
        return tf.data.Dataset.from_generator(*args, **kwargs) \
                              .repeat(repeat) \
                              .prefetch(prefetch) \
                              .make_one_shot_iterator() \
                              .get_next()


def placeholder(dtype, shape, x= None, name= None):
    """returns a placeholder with `dtype` and `shape`
    if tensor `x` is given, converts and uses it as default
    """
    if x is None: return tf.placeholder(dtype, shape, name)
    try:
        x = tf.convert_to_tensor(x, dtype)
    except ValueError:
        x = tf.cast(x, dtype)
    return tf.placeholder_with_default(x, shape, name)


def trim(x, eos, name= 'trim'):
    """trims a tensor of sequences
    x   : tensor i32 (b, ?)
    eos : tensor i32 ()
       -> tensor i32 (b, t)  the trimmed sequence tensor
        , tensor b8  (b, t)  the sequence mask
        , tensor i32 ()      the maximum non-eos sequence length t
    each row aka sequence in `x` is assumed to be any number of
    non-eos followed by any number of eos
    """
    with scope(name):
        with scope('not_eos'): not_eos = tf.not_equal(x, eos)
        with scope('len_seq'): len_seq = tf.reduce_sum(tf.cast(not_eos, tf.int32), axis= 1)
        with scope('max_len'): max_len = tf.reduce_max(len_seq)
        return x[:,:max_len], not_eos[:,:max_len], max_len


def spread_image(x, nrow, ncol, height, width):
    return tf.reshape(
        tf.transpose(
            tf.reshape(x, (nrow, ncol, height, width, -1))
            , (0, 2, 1, 3, 4))
        , (1, nrow * height, ncol * width, -1))


def get_shape(x, name= 'shape'):
    """returns the shape of `x` as a tuple of integers (static) or int32
    scalar tensors (dynamic)
    """
    with scope(name):
        shape = tf.shape(x)
        shape = tuple(d if d is not None else shape[i] for i, d in enumerate(x.shape.as_list()))
        return shape


def variable(name, shape, init= 'rand', initializers=
             {  'zero': tf.initializers.zeros()
              , 'unit': tf.initializers.ones()
              , 'rand': tf.glorot_uniform_initializer()
             }):
    """wraps `tf.get_variable` to provide initializer based on usage"""
    return tf.get_variable(name, shape, initializer= initializers.get(init, init))


class Linear(Record):
    """a linear transformation from m to n"""

    def __init__(self, n, m, name= 'linear'):
        self.name = name
        with scope(name):
            self.kern = variable('kern', (m, n), 'rand')

    def __call__(self, x, name= None):
        with scope(name or self.name):
            return x @ self.kern


class Affine(Record):
    """an affine transformation from m to n"""

    def __init__(self, n, m, name= 'affine'):
        self.name = name
        with scope(name):
            self.kern = variable('kern', (m, n), 'rand')
            self.bias = variable('bias', (1, n), 'zero')

    def __call__(self, x, name= None):
        with scope(name or self.name):
            return x @ self.kern + self.bias


class Normalize(Record):
    """layer normalization"""

    def __init__(self, dim, name= 'normalize'):
        self.name = name
        with scope(name):
            self.gain = variable('gain', (1, dim), 'unit')
            self.bias = variable('bias', (1, dim), 'zero')

    def __call__(self, x, name= None):
        with scope(name or self.name):
            mean, var = tf.nn.moments(x, 1, keep_dims= True)
            return (x - mean) * tf.rsqrt(var + 1e-12) * self.gain + self.bias
