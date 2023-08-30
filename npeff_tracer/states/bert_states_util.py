"""Utilities related to states of BERT."""
import dataclasses
import math
import re
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from transformers import BertConfig

###############################################################################


def d_tanh(x):
    return tf.square(1.0 / tf.math.cosh(x))


def d_gelu(x):
    return 0.5 * (tf.math.erf(x / tf.sqrt(2.0)) + 1.0) + tf.exp(-tf.square(x) / 2.0) * x / tf.sqrt(2.0 * math.pi)


def extract_layer_index(v: str) -> Union[None, str]:
    m = re.search(r'/encoder/layer_\._(\d+)/', v)
    return int(m.group(1)) if m else None


###############################################################################


@dataclasses.dataclass
class BertDenseActivations:
    """Information about the inputs and outputs of a tf.keras.layers.Dense within a BERT implmentation."""
    config: BertConfig

    # Result of calling layer.name_scope() on the tf.keras.layers.Dense instance.
    layer_name: str

    layer: tf.keras.layers.Dense

    input_activations: tf.Tensor
    output_preactivations: tf.Tensor

    @property
    def layer_index(self) -> Optional[int]:
        return extract_layer_index(self.layer_name)

    def is_attention_key(self) -> bool:
        return self.layer_name.endswith('/attention/self/key/')

    def is_attention_query(self) -> bool:
        return self.layer_name.endswith('/attention/self/query/')

    def is_attention_value(self) -> bool:
        return self.layer_name.endswith('/attention/self/value/')

    def is_attention_output(self) -> bool:
        return self.layer_name.endswith('/attention/output/dense/')

    def is_ffw_layer1(self) -> bool:
        return self.layer_name.endswith('/intermediate/dense/')

    def is_ffw_layer2(self) -> bool:
        return self.layer_name.endswith('/output/dense/') and not self.is_attention_output()

    def is_pooler(self) -> bool:
        return self.layer_name.endswith('/pooler/dense/')

    def is_classifier(self) -> bool:
        return self.layer_name.endswith('/classifier/')


@dataclasses.dataclass
class BertAttentionDenseActivations:
    """Information activations about within an attention block."""
    config: BertConfig

    query: BertDenseActivations
    key: BertDenseActivations

    value: BertDenseActivations
    output: BertDenseActivations

    def __post_init__(self):
        self.num_attention_heads = self.config.num_attention_heads
        self.attention_head_size = int(self.config.hidden_size / self.num_attention_heads)

    def multihead_value_kernel(self, t=None) -> tf.Tensor:
        # ret.shape = [d_model, num_attention_heads, attention_head_size]
        kernel = self.value.layer.kernel if t is None else t
        return tf.reshape(kernel, [kernel.shape[0], self.num_attention_heads, self.attention_head_size])

    def multihead_value_bias(self, t=None) -> tf.Tensor:
        # ret.shape = [num_attention_heads, attention_head_size]
        bias = self.value.layer.bias if t is None else t
        return tf.reshape(bias, [self.num_attention_heads, self.attention_head_size])

    def multihead_output_kernel(self, t=None) -> tf.Tensor:
        # ret.shape = [num_attention_heads, attention_head_size, d_model]
        kernel = self.output.layer.kernel if t is None else t
        kernel = tf.transpose(kernel)
        ret = tf.reshape(kernel, [kernel.shape[0], self.num_attention_heads, self.attention_head_size])
        return tf.transpose(ret, [1, 2, 0])

    def multihead_output_bias(self, t=None) -> tf.Tensor:
        # ret.shape = [d_model]
        return self.output.layer.bias if t is None else t

    def transpose_for_scores(self, tensor: tf.Tensor) -> tf.Tensor:
        # Essentially copied from the HuggingFace version.
        batch_size = tf.shape(tensor)[0]

        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])


@dataclasses.dataclass
class BertFfwDenseActivations:
    """Information activations about within a FFW block."""
    config: BertConfig

    layer1: BertDenseActivations
    layer2: BertDenseActivations

    def __post_init__(self):
        if self.config.hidden_act != 'gelu':
            raise ValueError(f'Unsupported hidden activation: {self.config.hidden_act}. Only GeLU supported.')
        self.act_fn = tf.nn.gelu
        self.d_act_fn = d_gelu
