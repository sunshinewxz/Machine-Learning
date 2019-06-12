import tensorflow as tf
import numpy as np
from tensorflow.python.estimator import estimator, model_fn
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.canned import optimizers
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.layers import core as core_layers
from tensorflow.python.ops import partitioned_variables, variable_scope
from tensorflow.python.ops.losses import losses

from daisy.models.regression_base import RegressionBase

from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Lambda

_LEARNING_RATE = 0.001


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    sines = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)

    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(tar):
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])

    return look_ahead_mask


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_v, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_v, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_v, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_v, d_model)

        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = Lambda(tf.contrib.layers.layer_norm)
        self.layernorm2 = Lambda(tf.contrib.layers.layer_norm)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = Lambda(tf.contrib.layers.layer_norm)
        self.layernorm2 = Lambda(tf.contrib.layers.layer_norm)
        self.layernorm3 = Lambda(tf.contrib.layers.layer_norm)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Dense(d_model)
        self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding2 = tf.keras.layers.Dense(d_model)
        self.pos_encoding = positional_encoding(target_vocab_size, self.d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding2(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, output_size, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, rate)

        self.final_layer = tf.keras.layers.Dense(output_size)

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


def _transformer_logit_fn_builder(units, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size,
                           output_size, feature_columns, dropout, input_layer_partitioner):
    if not isinstance(units, int):
        raise ValueError('units must be an int.  Given type: {}'.format(type(units)))

    def transformer_logit_fn(features, mode, labels, data_conf):

        with variable_scope.variable_scope(
                'input_from_feature_columns',
                values=tuple(features.items()),
                partitioner=input_layer_partitioner):
            inputs = [tf.expand_dims(feature_column_lib.input_layer(features, [fc]), -1)
                      for fc in feature_columns]
            inputs = tf.concat(inputs, axis=-1)

        # transformer
        with variable_scope.variable_scope('transformer'):
            transformer = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, output_size)
            if mode == model_fn.ModeKeys.TRAIN and dropout > 0:
                inputs = inputs[:, :-12, :]
                tar = inputs[:, 12:, :]
                tar_inp = tar[:, :-1, :]
                tar_real = tar[:, 1:, :]
                combined_mask = create_masks(tar_inp)
                # inp_ = (inputs, tar_inp, True, None, combined_mask, None)
                predictions, _ = transformer(inputs, tar_inp, True, None, combined_mask, None)

        with variable_scope.variable_scope('logits', values=(predictions, )) as logits_scope:
            predictions = core_layers.dense(predictions, units, name=logits_scope)

        return predictions

    return transformer_logit_fn


def _transformer_model_fn(features,
                   labels,
                   mode,
                   head,
                   num_layers,
                   d_model,
                   num_heads,
                   dff,
                   input_vocab_size,
                   target_vocab_size,
                   output_size,
                   feature_columns,
                   optimizer='Adam',
                   dropout=None,
                   input_layer_partitioner=None,
                   config=None,
                   data_conf=None):
    if not isinstance(features, dict):
        raise ValueError('features should be a dictionary of `Tensor`s. '
                         'Given type: {}'.format(type(features)))

    optimizer = optimizers.get_optimizer_instance(optimizer, learning_rate=_LEARNING_RATE)
    num_ps_replicas = config.num_ps_replicas if config else 0

    partitioner = partitioned_variables.min_max_variable_partitioner(max_partitions=num_ps_replicas)

    with variable_scope.variable_scope('transformer', values=tuple(features.items()),
                                       partitioner=partitioner):
        input_layer_partitioner = input_layer_partitioner or (
            partitioned_variables.min_max_variable_partitioner(
                max_partitions=num_ps_replicas,
                min_slice_size=64 << 20))

        logit_fn = _transformer_logit_fn_builder(
            units=head.logits_dimension,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            input_vocab_size=input_vocab_size,
            target_vocab_size=target_vocab_size,
            output_size=output_size,
            feature_columns=feature_columns,
            dropout=dropout,
            input_layer_partitioner=input_layer_partitioner)
        logits = logit_fn(features=features, mode=mode, labels=labels, data_conf=data_conf)

        return head.create_estimator_spec(
            features=features,
            mode=mode,
            labels=labels,
            optimizer=optimizer,
            logits=logits)


class _Transformer(estimator.Estimator):

    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 input_vocab_size,
                 target_vocab_size,
                 output_size,
                 feature_columns,
                 model_dir=None,
                 label_dimension=1,
                 weight_column=None,
                 optimizer='Adam',
                 dropout=None,
                 input_layer_partitioner=None,
                 config=None,
                 warm_start_from=None,
                 loss_reduction=losses.Reduction.SUM,
                 data_conf=None):

        def _model_fn(features, labels, mode, config):
            """Call the defined shared _lstm_model_fn."""
            return _transformer_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                head=head_lib._regression_head_with_mean_squared_error_loss(
                    label_dimension=label_dimension,
                    weight_column=weight_column,
                    loss_reduction=loss_reduction),
                num_layers=num_layers,
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                input_vocab_size=input_vocab_size,
                target_vocab_size=target_vocab_size,
                output_size=output_size,
                feature_columns=tuple(feature_columns or []),
                optimizer=optimizer,
                dropout=dropout,
                input_layer_partitioner=input_layer_partitioner,
                config=config,
                data_conf=data_conf)

        super().__init__(_model_fn, model_dir, config, warm_start_from)


class TransformerRegression(RegressionBase):
    def _build_model(self):
        return _Transformer(num_layers=self.conf.num_layers,
                     d_model=self.conf.d_model,
                     num_heads=self.conf.num_heads,
                     dff=self.conf.dff,
                     input_vocab_size=self.conf.input_vocab_size,
                     target_vocab_size=self.conf.target_vocab_size,
                     output_size=self.conf.output_size,
                     feature_columns=self.feature_columns,
                     label_dimension=self.n_outputs,
                     model_dir=self.model_dir,
                     config=self.run_config,
                     dropout=self.conf.drop_rate,
                     optimizer=tf.train.AdamOptimizer(self.learning_rate),
                     data_conf=self.data_conf)
