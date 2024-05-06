import tensorflow as tf
from keras.src import backend
from keras.src import ops
import keras
@keras.saving.register_keras_serializable()


def squeeze_or_expand_to_same_rank(x1, x2, expand_rank_1=True):
    """Squeeze/expand last dim if ranks differ from expected by exactly 1."""
    x1_rank = len(x1.shape)
    x2_rank = len(x2.shape)
    if x1_rank == x2_rank:
        return x1, x2
    if x1_rank == x2_rank + 1:
        if x1.shape[-1] == 1:
            if x2_rank == 1 and expand_rank_1:
                x2 = ops.expand_dims(x2, axis=-1)
            else:
                x1 = ops.squeeze(x1, axis=-1)
    if x2_rank == x1_rank + 1:
        if x2.shape[-1] == 1:
            if x1_rank == 1 and expand_rank_1:
                x1 = ops.expand_dims(x1, axis=-1)
            else:
                x2 = ops.squeeze(x2, axis=-1)
    return x1, x2

def step_msle(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    first_log = tf.math.log(backend.maximum(y_pred, backend.epsilon()) + 1.0)
    second_log = tf.math.log(backend.maximum(y_true, backend.epsilon()) + 1.0)
    
    mu=backend.maximum(0.0,tf.sign(-1.0*(y_pred-1.0)))

    full=tf.math.squared_difference(first_log, second_log)+mu
    return backend.mean(full
        , axis=-1
    )

@keras.saving.register_keras_serializable()


def exp_msle(y_true, y_pred):

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    first_log = tf.math.log(backend.maximum(y_pred, backend.epsilon()) + 1.0)
    second_log = tf.math.log(backend.maximum(y_true, backend.epsilon()) + 1.0)
    return backend.mean(
        tf.math.squared_difference(first_log, second_log)+1*tf.math.exp(-10*y_pred+1), axis=-1
    )

@keras.saving.register_keras_serializable()

def sigmoid_msle(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    first_log = tf.math.log(backend.maximum(y_pred, backend.epsilon()) + 1.0)
    second_log = tf.math.log(backend.maximum(y_true, backend.epsilon()) + 1.0)
    multip=10.0
    mu=backend.maximum(multip*(1.0/(1.0+tf.math.exp((y_pred-1.0))))-multip/2,0.0)
    return backend.mean(
        tf.math.squared_difference(first_log, second_log)+mu, axis=-1
    )

@keras.saving.register_keras_serializable()


def cube_msle(y_true, y_pred):
    epsilon = ops.convert_to_tensor(backend.epsilon())
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)
    first_log = ops.log(ops.maximum(y_pred, epsilon) + 1.0)
    second_log = ops.log(ops.maximum(y_true, epsilon) + 1.0)
    mu=ops.maximum(-1*tf.math.pow((y_pred-1.0), 3.0),0.0)
    return ops.mean(
        tf.math.squared_difference(first_log, second_log)+mu, axis=-1
    )

@keras.saving.register_keras_serializable()
def linear_msle(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    first_log = tf.math.log(backend.maximum(y_pred, backend.epsilon()) + 1.0)
    second_log = tf.math.log(backend.maximum(y_true, backend.epsilon()) + 1.0)
    mu=backend.maximum(-1*tf.math.pow((y_pred-1.0), 1.0),0.0)
    return backend.mean(
        tf.math.squared_difference(first_log, second_log)+mu, axis=-1
    )