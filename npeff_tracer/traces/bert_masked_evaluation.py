"""Evaluation of a BERT model's NPEFF perturbation with some perturbations/connections masked."""
import dataclasses
import itertools
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import tensorflow as tf

from npeff_tracer.states import bert_states

###############################################################################

PoolerInputFisher = bert_states.PoolerInputFisher
AttentionState = bert_states.AttentionState
FfwState = bert_states.FfwState
BertState = bert_states.BertState

###############################################################################


def _default_non_attention_mask(sequence_length: int, n_non_padding: int) -> tf.Tensor:
    ones = tf.ones([n_non_padding], dtype=tf.float32)
    zeros = tf.zeros([sequence_length - n_non_padding], dtype=tf.float32)
    return tf.concat([ones, zeros], axis=0)


def _default_attention_mask(sequence_length: int, n_non_padding: int, n_heads: int) -> tf.Tensor:
    n_padding = sequence_length - n_non_padding
    return tf.concat([
        tf.concat([
            tf.ones([n_heads, n_non_padding, n_non_padding], dtype=tf.float32),
            tf.zeros([n_heads, n_non_padding, n_padding], dtype=tf.float32),
        ], axis=2),
        tf.zeros([n_heads, n_padding, sequence_length], dtype=tf.float32),
    ], axis=1)


class _LayerMaskBase:
    """Base class for the LayerMasks below.

    Just contains some common logic here.
    """

    @property
    def sequence_length(self) -> int:
        return self.residual_mask.shape[0]

    def _get_all_masks(self) -> List[tf.Variable]:
        return [self.residual_mask, self.transformed_mask, self.induced_mask]

    ####################################
    # Masks with dummy batch dimensions.

    @property
    def b_residual_mask(self) -> tf.Tensor:
        # Adds dummy batch dim.
        return tf.expand_dims(self.residual_mask, axis=0)
    
    @property
    def b_transformed_mask(self) -> tf.Tensor:
        # Adds dummy batch dim.
        return tf.expand_dims(self.transformed_mask, axis=0)
    
    @property
    def b_induced_mask(self) -> tf.Tensor:
        # Adds dummy batch dim.
        return tf.expand_dims(self.induced_mask, axis=0)

    @property
    def b_induced_units_mask(self) -> Optional[tf.Tensor]:
        # Adds dummy batch dim.
        induced_units_mask = getattr(self, 'induced_units_mask', None)
        if induced_units_mask is None:
            return induced_units_mask
        return tf.expand_dims(induced_units_mask, axis=0)

    ####################################

    def mask_all(self):
        for mask in self._get_all_masks():
            mask.assign(tf.zeros_like(mask))
 
    def mask_all_of_type(self, mask_name: str):
        mask = getattr(self, mask_name)
        mask.assign(tf.zeros_like(mask))

###########################################################


@dataclasses.dataclass(eq=False)
class AttentionLayerMask(_LayerMaskBase):
    # Assumes that this many tokens at the start of the input as non-padding and
    # that the rest is padding.
    n_non_padding: int

    # residual_mask.shape = [sequence_length]
    residual_mask: tf.Variable

    # transformed_mask.shape = [n_heads, sequence_length_query, sequence_length_key]
    transformed_mask: tf.Variable

    # induced_mask.shape = [n_heads, sequence_length_query, sequence_length_key]
    induced_mask: tf.Variable

    # NOTE: The units here correspond to outputs of the value transformation.
    # induced_units_mask.shape = [n_heads, n_units_per_head]
    induced_units_mask: Optional[tf.Variable] = None

    @property
    def n_heads(self) -> int:
        return self.transformed_mask.shape[0]

    #######################################################

    def reset(self):
        """Sets the masks to their original, non-masked values."""
        initial_residual_mask = _default_non_attention_mask(
            sequence_length=self.sequence_length, n_non_padding=self.n_non_padding)
        initial_transformed_mask = _default_attention_mask(
            sequence_length=self.sequence_length, n_non_padding=self.n_non_padding, n_heads=self.n_heads)

        self.residual_mask.assign(initial_residual_mask)
        self.transformed_mask.assign(initial_transformed_mask)
        self.induced_mask.assign(initial_transformed_mask)

        if self.induced_units_mask is not None:
            self.induced_units_mask.assign(tf.ones_like(self.induced_units_mask))

    #######################################################

    def set_residual_sequence_position(self, sequence_index: int, value: float):
        # Ensure that value is either 0 or 1.
        value = tf.cast(value != 0.0, tf.float32)
        self.residual_mask.scatter_nd_update([[sequence_index]], [value])

    def set_single_attention_position_across_heads(
        self, mask_name: str, query_sequence_index: int, key_sequence_index: int, value: float
    ):
        assert mask_name in ('transformed_mask', 'induced_mask')
        mask = getattr(self, mask_name)
        indices = [
            [head_index, query_sequence_index, key_sequence_index]
            for head_index in range(self.n_heads)
        ]
        # Ensure that value is either 0 or 1.
        value = tf.cast(value != 0.0, tf.float32)
        updates = value + tf.zeros([self.n_heads], dtype=mask.dtype)
        mask.scatter_nd_update(indices, updates)

    def set_single_attention_position(
        self, mask_name: str, head_index: int, query_sequence_index: int, key_sequence_index: int, value: float
    ):
        assert mask_name in ('transformed_mask', 'induced_mask')
        mask = getattr(self, mask_name)
        # Ensure that value is either 0 or 1.
        value = tf.cast(value != 0.0, tf.float32)
        mask.scatter_nd_update(
            [[head_index, query_sequence_index, key_sequence_index]], [value])

    def set_query(self, mask_name: str, head_index: int, query_sequence_index: int, value: float):
        assert mask_name in ('transformed_mask', 'induced_mask')
        mask = getattr(self, mask_name)

        # Ensure that value is either 0 or 1.
        value = tf.cast(value != 0.0, tf.float32)

        mask.scatter_nd_update(
            [[head_index, query_sequence_index]],
            [value + tf.zeros_like(mask[head_index, query_sequence_index])])

    def set_query_across_heads(self, mask_name: str, query_sequence_index: int, value: float):
        assert mask_name in ('transformed_mask', 'induced_mask')
        mask = getattr(self, mask_name)

        # Ensure that value is either 0 or 1.
        value = tf.cast(value != 0.0, tf.float32)

        indices = [
            [head_index, query_sequence_index]
            for head_index in range(self.n_heads)
        ]
        updates = value + tf.zeros([self.n_heads, mask.shape[-1]], dtype=mask.dtype)
        mask.scatter_nd_update(indices, updates)

    def set_induced_unit(self, head_index: int, unit_index: int, value: float):
        # Ensure that value is either 0 or 1.
        value = tf.cast(value != 0.0, tf.float32)
        self.induced_units_mask.scatter_nd_update([[head_index, unit_index]], [value])

    #######################################################

    def mask_residual_sequence_position(self, sequence_index: int):
        self.residual_mask.scatter_nd_update([[sequence_index]], [0.0])

    def mask_single_attention_position(
        self, mask_name: str, head_index: int, query_sequence_index: int, key_sequence_index: int
    ):
        assert mask_name in ('transformed_mask', 'induced_mask')
        mask = getattr(self, mask_name)
        mask.scatter_nd_update(
            [[head_index, query_sequence_index, key_sequence_index]], [0.0])

    def mask_single_attention_position_across_heads(
        self, mask_name: str, query_sequence_index: int, key_sequence_index: int
    ):
        assert mask_name in ('transformed_mask', 'induced_mask')
        mask = getattr(self, mask_name)
        indices = [
            [head_index, query_sequence_index, key_sequence_index]
            for head_index in range(self.n_heads)
        ]
        updates = tf.zeros([self.n_heads], dtype=mask.dtype)
        mask.scatter_nd_update(indices, updates)

    def mask_head(self, mask_name: str, head_index: int):
        assert mask_name in ('transformed_mask', 'induced_mask')
        mask = getattr(self, mask_name)
        mask.scatter_nd_update([[head_index]], [tf.zeros_like(mask[head_index])])

    def mask_query(self, mask_name: str, head_index: int, query_sequence_index: int):
        assert mask_name in ('transformed_mask', 'induced_mask')
        mask = getattr(self, mask_name)
        mask.scatter_nd_update(
            [[head_index, query_sequence_index]],
            [tf.zeros_like(mask[head_index, query_sequence_index])])

    def mask_key(self, mask_name: str, head_index: int, key_sequence_index: int):
        assert mask_name in ('transformed_mask', 'induced_mask')
        mask = getattr(self, mask_name)
        indices = [[head_index, i, key_sequence_index] for i in range(self.n_non_padding)]
        updates = tf.zeros_like(mask[head_index, :self.n_non_padding, key_sequence_index])
        mask.scatter_nd_update(indices, updates)

    def mask_induced_unit(self, head_index: int, unit_index: int):
        self.set_induced_unit(head_index, unit_index, 0.0)

    #######################################################

    @classmethod
    def create_initial(cls, block: AttentionState, n_non_padding: int, support_unit_masks: bool = False) -> 'AttentionLayerMask':
        seq_len = block.sequence_length
        n_heads = block.n_heads
        attention_head_size = block.activations_info.attention_head_size

        initial_residual_mask = _default_non_attention_mask(
            sequence_length=seq_len, n_non_padding=n_non_padding)
        initial_transformed_mask = _default_attention_mask(
            sequence_length=seq_len, n_non_padding=n_non_padding, n_heads=n_heads)

        induced_units_mask = None
        if support_unit_masks:
            induced_units_mask = tf.Variable(tf.ones([n_heads, attention_head_size], dtype=tf.float32), trainable=False)

        ret = cls(
            n_non_padding=n_non_padding,
            residual_mask=tf.Variable(initial_residual_mask, trainable=False),
            transformed_mask=tf.Variable(initial_transformed_mask, trainable=False),
            induced_mask=tf.Variable(initial_transformed_mask, trainable=False),
            induced_units_mask=induced_units_mask,
        )

        return ret


@dataclasses.dataclass(eq=False)
class FfwLayerMask(_LayerMaskBase):
    # Assumes that this many tokens at the start of the input as non-padding and
    # that the rest is padding.
    n_non_padding: int

    # residual_mask.shape = [sequence_length]
    residual_mask: tf.Variable
    
    # transformed_mask.shape = [sequence_length]
    transformed_mask: tf.Variable

    # induced_mask.shape = [sequence_length]
    induced_mask: tf.Variable

    # induced_units_mask.shape = [d_ff]
    induced_units_mask: Optional[tf.Variable] = None

    #######################################################

    def reset(self):
        """Sets the masks to their original, non-masked values."""
        default_mask = _default_non_attention_mask(
            sequence_length=self.sequence_length, n_non_padding=self.n_non_padding)

        self.residual_mask.assign(default_mask)
        self.transformed_mask.assign(default_mask)
        self.induced_mask.assign(default_mask)

        if self.induced_units_mask is not None:
            self.induced_units_mask.assign(tf.ones_like(self.induced_units_mask))

    #######################################################

    def set_sequence_position(self, mask_name: str, sequence_index: int, value: float):
        # Ensure that value is either 0 or 1.
        value = tf.cast(value != 0.0, tf.float32)
        getattr(self, mask_name).scatter_nd_update([[sequence_index]], [value])

    def set_residual_sequence_position(self, sequence_index: int, value: float):
        self.set_sequence_position('residual_mask', sequence_index, value)

    def set_induced_unit(self, unit_index: int, value: float):
        # Ensure that value is either 0 or 1.
        value = tf.cast(value != 0.0, tf.float32)
        self.induced_units_mask.scatter_nd_update([[unit_index]], [value])

    #######################################################

    def mask_sequence_position(self, mask_name: str, sequence_index: int):
        getattr(self, mask_name).scatter_nd_update([[sequence_index]], [0.0])

    def mask_residual_sequence_position(self, sequence_index: int):
        self.residual_mask.scatter_nd_update([[sequence_index]], [0.0])

    def mask_induced_unit(self, unit_index: int):
        self.set_induced_unit(unit_index, 0.0)

    #######################################################

    @classmethod
    def create_initial(cls, block: FfwState, n_non_padding: int, support_unit_masks: bool = False) -> 'FfwLayerMask':
        seq_len = block.sequence_length
        default_mask = _default_non_attention_mask(
            sequence_length=seq_len, n_non_padding=n_non_padding)

        induced_units_mask = None
        if support_unit_masks:
            induced_units_mask = tf.Variable(tf.ones([block.d_ff], dtype=tf.float32), trainable=False)

        return cls(
            n_non_padding=n_non_padding,
            residual_mask=tf.Variable(default_mask, trainable=False),
            transformed_mask=tf.Variable(default_mask, trainable=False),
            induced_mask=tf.Variable(default_mask, trainable=False),
            induced_units_mask=induced_units_mask,
        )


###############################################################################


@dataclasses.dataclass(eq=False)
class MaskedEvaluator:
    """Evaluates perturbations given masks."""

    bert_state: BertState

    # Whether to support per-unit masking where available.
    support_unit_masks: bool = False

    # Created from bert_state if set to None.
    layer_masks: List[Union[AttentionLayerMask, FfwLayerMask]] = None

    # Created from bert_state if set to None.
    pooler_input_fisher: PoolerInputFisher = None

    #######################################################

    def __post_init__(self):
        assert self.bert_state.get_batch_size() == 1, 'Only supporting single example batches.'

        self.n_non_padding = int(self.bert_state.get_n_non_padding()[0].numpy())

        if self.pooler_input_fisher is None:
            self.pooler_input_fisher = self.bert_state.get_pooler_input_fisher()

        if self.layer_masks is None:
            self.layer_masks = self._create_initial_layer_masks()

    def _create_initial_layer_masks(self):
        ret = []
        for block in self.bert_state.residual_blocks:
            if isinstance(block, AttentionState):
                ret.append(AttentionLayerMask.create_initial(block, self.n_non_padding, support_unit_masks=self.support_unit_masks))
            elif isinstance(block, FfwState):
                ret.append(FfwLayerMask.create_initial(block, self.n_non_padding, support_unit_masks=self.support_unit_masks))
            else:
                raise ValueError
        return ret

    #######################################################

    def reset(self):
        """Sets all of the masks to their original, non-masked values."""
        for layer_mask in self.layer_masks:
            layer_mask.reset()

    #######################################################

    def _compute_total_pooler_input_perturbation_masked(self) -> tf.Tensor:
        ret = tf.zeros_like(self.bert_state.residual_blocks[0].input_activations)
        for block, mask in zip(self.bert_state.residual_blocks, self.layer_masks):
            ret = block.compute_layer_output_perturbation_masked(
                ret,
                induced_mask=mask.b_induced_mask,
                transformed_mask=mask.b_transformed_mask,
                residual_mask=mask.b_residual_mask,
                induced_units_mask=mask.b_induced_units_mask,
            )
        return ret[:, self.bert_state.cls_token_index, :]

    @tf.function
    def compute_total_pooler_fisher_dot_products_masked(self) -> tf.Tensor:
        # ret.shape = [n_classes]
        pooler_input_perturbation = self._compute_total_pooler_input_perturbation_masked()
        ret = self.pooler_input_fisher.compute_dot_products(pooler_input_perturbation)
        return tf.squeeze(ret, axis=0)

