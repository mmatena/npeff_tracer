"""Classes to wrap the impact of an LRM-NPEFF component on the internals of BERT."""

import dataclasses
import math
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from transformers import BertConfig

from . import bert_states_util

###############################################################################

d_tanh = bert_states_util.d_tanh
d_gelu = bert_states_util.d_gelu

BertDenseActivations = bert_states_util.BertDenseActivations
BertAttentionDenseActivations = bert_states_util.BertAttentionDenseActivations
BertFfwDenseActivations = bert_states_util.BertFfwDenseActivations

###############################################################################


@dataclasses.dataclass(eq=False)
class PoolerState:
    """Information about the pooler layer."""
    layer: tf.keras.layers.Dense

    # input_activations.shape = [batch_size, d_model]
    input_activations: tf.Tensor

    # output_preactivations.shape = [batch_size, d_model]
    output_preactivations: tf.Tensor

    def __post_init__(self):
        # Assumes (and does not check) that the pooler uses a tanh activation.
        self.d_output_acts_d_output_preacts = d_tanh(self.output_preactivations)

    def get_pooler_kernel_units(self) -> List[tf.Tensor]:
        return tf.unstack(self.pooler.kernel, axis=-1)


@dataclasses.dataclass(eq=False)
class ClassifierState:
    """Information about the classifier layer."""
    layer: tf.keras.layers.Dense

    # input_activations.shape = [batch_size, d_model]
    input_activations: tf.Tensor

    # output_preactivations.shape = [batch_size, d_model]
    output_preactivations: tf.Tensor


@dataclasses.dataclass(eq=False)
class PoolerInputFisher:
    """Information about the Fisher of the network output wrt the pooler input.

    Represented via a set of n_classes vectors for each example. The PEF will be
    sum of outer products of the vectors.
    """

    # fisher_vectors.shape = [batch_size, n_classes, d_model]
    fisher_vectors: tf.Tensor

    @property
    def n_classes(self) -> int:
        return self.fisher_vectors.shape[-2]

    def compute_dot_products(self, pooler_input_perturbations: tf.Tensor) -> tf.Tensor:
        # pooler_input_perturbations.shape = [batch_size, d_model]
        # ret.shape = [batch_size, n_classes]
        return tf.einsum('bcm,bm->bc', self.fisher_vectors, pooler_input_perturbations)

    @classmethod
    def create_from_states(cls, pooler_state: PoolerState, classifier_state: ClassifierState):
        # NOTE: This method maybe could be faster, but this is probably good enough for now.
        inputs = pooler_state.input_activations
        fisher_vectors = []

        batch_size = inputs.shape[0]
        for i in range(batch_size):
            ex_inputs = tf.expand_dims(inputs[i], axis=0)
            ex_fishers = []

            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
                tape.watch(ex_inputs)
                logits = classifier_state.layer(pooler_state.layer(ex_inputs))
                logits = tf.squeeze(logits, axis=0)
                log_probs = tf.math.log_softmax(logits, axis=-1)
                sqrt_probs = tf.sqrt(tf.math.softmax(logits, axis=-1))

                for j in range(log_probs.shape[-1]):
                    log_prob = log_probs[j]
                    with tape.stop_recording():
                        f = tape.gradient(log_prob, ex_inputs)
                        f = tf.squeeze(f, axis=0)
                        ex_fishers.append(sqrt_probs[j] * f)

            fisher_vectors.append(ex_fishers)

        return cls(fisher_vectors=tf.cast(fisher_vectors, tf.float32))


@dataclasses.dataclass(eq=False)
class LayerNormState:
    """Information about a LayerNorm.

    We basically "pass-through" perturbations through the LayerNorm. We assume that
    the expected values and variances do not change with the perturbations.

    Hence this class doesn't contain information about perturbations but only information
    about the LayerNorm parameters and the moments of the inputs of the LayerNorm.
    """

    epsilon: float

    # gamma.shape = [d_model]
    gamma: tf.Tensor

    # beta.shape = [d_model]
    beta: tf.Tensor

    # means.shape = [batch_size, sequence_length]
    means: tf.Tensor

    # variances.shape = [batch_size, sequence_length]
    variances: tf.Tensor

    def get_Gamma_diag(self) -> tf.Tensor:
        # ret.shape = [batch_size, sequence_length, d_model]
        return tf.math.rsqrt(self.variances[..., None] + self.epsilon) * self.gamma

    def get_b(self) -> tf.Tensor:
        # ret.shape = [batch_size, sequence_length, d_model]
        ret = self.means[..., None] * tf.math.rsqrt(self.variances[..., None] + self.epsilon) * self.gamma
        return -ret + self.beta

    def transform_pre_layer_norm_perturbations(self, perturbations: tf.Tensor) -> tf.Tensor:
        # perturbations/ret.shape = [batch_size, sequence_length, d_model]
        return self.get_Gamma_diag() * perturbations


###############################################################################


@dataclasses.dataclass(eq=False)
class AttentionState:
    """Information about a self-attention transformer layer's activations and perturbations.
    
    The information is stored for all sequence positions.
    """

    ########################################
    # Information about induced perturbations.
    #
    # These should capture information that allows us to compute the output perturbations
    # with the various masking strategies that I wish to support.

    # Perturbations that arise from perturbations of the query parameters.
    # query_perturbations.shape = [batch_size, num_attention_heads, seq_len_q, seq_len_k]
    query_logit_perturbations: tf.Tensor

    # Perturbations that arise from perturbations of the key parameters.
    # key_perturbations.shape = [batch_size, num_attention_heads, seq_len_q, seq_len_k]
    key_logit_perturbations: tf.Tensor

    # Perturbations that arise from perturbations of the value parameters.
    # value_perturbations.shape = [batch_size, num_attention_heads, seq_len_k, d_model]
    value_perturbations: tf.Tensor

    # Perturbations that arise from perturbations of the output parameters (except for the output bias).
    # value_perturbations.shape = [batch_size, num_attention_heads, seq_len_k, d_model]
    output_perturbations: tf.Tensor

    # Perturbations that arise from perturbations of the output bias parameters.
    # value_bias_perturbations.shape = [d_model]
    output_bias_perturbations: tf.Tensor

    ########################################
    # Some other cached information about the layer and/or its processing of the examples.

    activations_info: BertAttentionDenseActivations

    # The attention map used to sum values.
    # attention_map.shape = [batch_size, num_heads, seq_len_q, seq_len_k]
    attention_map: tf.Tensor

    # Information about the LayerNorm operating on the output of this residual block.
    layer_norm_state: LayerNormState

    # Parameter perturbations of the query and key matrices.
    # shapes = [d_model, num_attention_heads, attention_head_size]
    P_Q: tf.Tensor
    P_K: tf.Tensor

    # Parameter perturbations of the value matrix.
    # shape = [d_model, num_attention_heads, attention_head_size]
    P_V: tf.Tensor

    # Parameter perturbations of the value bias.
    # shape = [num_attention_heads, attention_head_size]
    p_v: tf.Tensor

    # Parameter perturbations of the output matrix.
    # shape = [num_attention_heads, attention_head_size, d_model]
    P_U: tf.Tensor

    #######################################################

    @property
    def input_activations(self) -> tf.Tensor:
        # input_activations.shape = [batch_size, seq_len, d_model]
        # The QKV will all have the same inputs, so choose one arbitrarily.
        return self.activations_info.query.input_activations

    @property
    def sequence_length(self) -> int:
        return self.attention_map.shape[-1]

    @property
    def n_heads(self) -> int:
        return self.attention_map.shape[-3]

    #######################################################

    def _transform_logits_with_softmax_jacobian(self, logits: tf.Tensor) -> tf.Tensor:
        # logits.shape = [batch, num_attention_heads, seq_len_q, seq_len_k]
        #
        # ret.shape = [batch, num_attention_heads, seq_len_q, seq_len_k]
        term1 = self.attention_map * logits
        term2 = tf.einsum('bnqi,bnqk,bnqk->bnqi', self.attention_map, self.attention_map, logits)
        return term1 - term2

    def _transform_softmax_perturbations_to_pre_layer_norm(self, softmax_perturbations: tf.Tensor) -> tf.Tensor:
        # softmax_perturbations.shape = [batch, num_attention_heads, seq_len_q, seq_len_k]
        #
        # ret.shape = [batch_size, sequence_length, d_model]

        # TODO: Might need to take into account the effect of some of the biases of the values/outputs.

        V = self.activations_info.multihead_value_kernel()
        U = self.activations_info.multihead_output_kernel()
        return tf.einsum('bkh,hnd,ndj,bnqk->bqj', self.input_activations, V, U, softmax_perturbations)

    def _transform_input_perturbations_via_query_key_to_pre_layer_norm(
        self,
        input_perturbations: tf.Tensor,
        transformed_mask: tf.Tensor,
    ) -> tf.Tensor:
        # input_perturbations.shape = [batch, seq_len, d_model]
        # transformed_mask.shape = [batch, num_heads, seq_len_q, seq_len_k]
        # ret.shape = [batch, seq_len, d_model]

        att_acts = self.activations_info

        # shapes = [batch_size, num_heads, seq_length, attention_head_size]
        queries = att_acts.transpose_for_scores(att_acts.query.output_preactivations)
        keys = att_acts.transpose_for_scores(att_acts.key.output_preactivations)

        rsqrt_d_head = tf.math.rsqrt(tf.cast(tf.shape(queries)[-1], tf.float32))

        logits_perturbations_query = rsqrt_d_head * tf.matmul(
            tf.einsum('bsh,hnd->bnsd', input_perturbations, self.P_Q),
            keys,
            transpose_b=True)
        logits_perturbations_key = rsqrt_d_head * tf.matmul(
            queries,
            tf.einsum('bsh,hnd->bnsd', input_perturbations, self.P_K),
            transpose_b=True)

        logits_perturbations = transformed_mask * (logits_perturbations_query + logits_perturbations_key)
        softmax_perturbations = self._transform_logits_with_softmax_jacobian(logits_perturbations)
        return self._transform_softmax_perturbations_to_pre_layer_norm(softmax_perturbations)

    #######################################################

    def transform_input_perturbations_via_query_only_to_pre_layer_norm(
        self,
        input_perturbations: tf.Tensor,
        transformed_mask: tf.Tensor,
    ):
        # input_perturbations.shape = [batch, seq_len, d_model]
        # transformed_mask.shape = [batch, num_heads, seq_len_q, seq_len_k]
        # ret.shape = [batch, seq_len, d_model]

        att_acts = self.activations_info

        # shapes = [batch_size, num_heads, seq_length, attention_head_size]
        keys = att_acts.transpose_for_scores(att_acts.key.output_preactivations)

        rsqrt_d_head = tf.math.rsqrt(tf.cast(tf.shape(keys)[-1], tf.float32))

        logits_perturbations_query = rsqrt_d_head * tf.matmul(
            tf.einsum('bsh,hnd->bnsd', input_perturbations, self.P_Q),
            keys,
            transpose_b=True)

        logits_perturbations = transformed_mask * logits_perturbations_query
        softmax_perturbations = self._transform_logits_with_softmax_jacobian(logits_perturbations)
        return self._transform_softmax_perturbations_to_pre_layer_norm(softmax_perturbations)

    def transform_input_perturbations_via_key_only_to_pre_layer_norm(
        self,
        input_perturbations: tf.Tensor,
        transformed_mask: tf.Tensor,
    ):
        # input_perturbations.shape = [batch, seq_len, d_model]
        # transformed_mask.shape = [batch, num_heads, seq_len_q, seq_len_k]
        # ret.shape = [batch, seq_len, d_model]

        att_acts = self.activations_info

        # shapes = [batch_size, num_heads, seq_length, attention_head_size]
        queries = att_acts.transpose_for_scores(att_acts.query.output_preactivations)

        rsqrt_d_head = tf.math.rsqrt(tf.cast(tf.shape(queries)[-1], tf.float32))

        logits_perturbations_key = rsqrt_d_head * tf.matmul(
            queries,
            tf.einsum('bsh,hnd->bnsd', input_perturbations, self.P_K),
            transpose_b=True)

        logits_perturbations = transformed_mask * logits_perturbations_key
        softmax_perturbations = self._transform_logits_with_softmax_jacobian(logits_perturbations)
        return self._transform_softmax_perturbations_to_pre_layer_norm(softmax_perturbations)

    def transform_input_perturbations_via_value_output_only_to_pre_layer_norm(
        self,
        input_perturbations: tf.Tensor,
        transformed_mask: tf.Tensor,
    ):
        # input_perturbations.shape = [batch, seq_len, d_model]
        # transformed_mask.shape = [batch, num_heads, seq_len_q, seq_len_k]
        # ret.shape = [batch, seq_len, d_model]

        V = self.activations_info.multihead_value_kernel()
        U = self.activations_info.multihead_output_kernel()

        p_transformed_vo = tf.einsum(
            'bnqk,bnqk,bkh,hnd,ndj->bqj',
            self.attention_map, transformed_mask, input_perturbations, V, U)

        return p_transformed_vo

    #######################################################

    def _compute_induced_vo_perturbations_with_units_mask(self, induced_mask: tf.Tensor, induced_units_mask: tf.Tensor):
        # induced_mask_mask.shape = [batch, num_heads, seq_len_q, seq_len_k]
        # induced_units_mask.shape = [batch, num_heads, n_units_per_head]
        assert induced_units_mask is not None

        induced_units_mask = tf.cast(induced_units_mask, tf.float32)

        # The QKV will all have the same inputs, so choose one arbitrarily.
        input_activations = self.activations_info.value.input_activations

        V = self.activations_info.multihead_value_kernel()
        v = self.activations_info.multihead_value_bias()
        U = self.activations_info.multihead_output_kernel()

        P_V, p_v, P_U = self.P_V, self.p_v, self.P_U

        value_perturbations = tf.einsum('bkm,mja,bja,jan->bjkn', input_activations, P_V, induced_units_mask, U)
        value_perturbations = value_perturbations + tf.einsum('ja,bja,jan->bjn', p_v, induced_units_mask, U)[:, :, None, :]

        output_perturbations = tf.einsum('bkm,mja,bja,jan->bjkn', input_activations, V, induced_units_mask, P_U)
        output_perturbations = output_perturbations + tf.einsum('ja,bja,jan->bjn', v, induced_units_mask, P_U)[:, :, None, :]

        p_induced_vo = value_perturbations + output_perturbations
        p_induced_vo = tf.einsum('bhqk,bhqk,bhkm->bqm', self.attention_map, induced_mask, p_induced_vo)

        return p_induced_vo

    #######################################################

    def compute_layer_output_perturbation_masked(
            self,
            input_perturbations: tf.Tensor,
            induced_mask: tf.Tensor, transformed_mask: tf.Tensor, residual_mask: tf.Tensor,
            induced_units_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """The entire perturbation in the hidden states after this layer including the LayerNorm.
        
        NOTE: This ignores the output_bias_perturbations.
        TODO: Might need to take into account the effect of some of the biases of the values/outputs!
        
        input_perturbations.shape = [batch, seq_len, d_model]
        
        induced_mask.shape = [batch, num_heads, seq_len_q, seq_len_k]
        transformed_mask.shape = [batch, num_heads, seq_len_q, seq_len_k]
        residual_mask.shape = [batch, seq_len]

        induced_units_mask.shape = [batch, num_heads, n_units_per_head]
        
        ret.shape = [batch, seq_len, d_model]
        """
        induced_mask = tf.cast(induced_mask, tf.float32)
        transformed_mask = tf.cast(transformed_mask, tf.float32)
        residual_mask = tf.cast(residual_mask, tf.float32)[..., None]

        V = self.activations_info.multihead_value_kernel()
        U = self.activations_info.multihead_output_kernel()

        #############################################
        # Compute the induced perturbations.

        p_induced_qk = induced_mask * (self.query_logit_perturbations + self.key_logit_perturbations)
        p_induced_qk = self._transform_logits_with_softmax_jacobian(p_induced_qk)
        p_induced_qk = self._transform_softmax_perturbations_to_pre_layer_norm(p_induced_qk)

        # p_induced_vo = self.value_perturbations + self.output_perturbations
        # p_induced_vo = tf.einsum('bhqk,bhqk,bhkm->bqm', self.attention_map, induced_mask, p_induced_vo)

        if induced_units_mask is None:
            p_induced_vo = self.value_perturbations + self.output_perturbations
            p_induced_vo = tf.einsum('bhqk,bhqk,bhkm->bqm', self.attention_map, induced_mask, p_induced_vo)
            
        else:
            p_induced_vo = self._compute_induced_vo_perturbations_with_units_mask(induced_mask, induced_units_mask)

        p_induced = p_induced_qk + p_induced_vo

        #############################################
        # Compute the perturbations that pass through the residual connections.

        p_residual = residual_mask * input_perturbations

        #############################################
        # Compute the perturbations that pass through the attention layer.

        p_transformed_qk = self._transform_input_perturbations_via_query_key_to_pre_layer_norm(
            input_perturbations, transformed_mask)

        p_transformed_vo = tf.einsum(
            'bnqk,bnqk,bkh,hnd,ndj->bqj',
            self.attention_map, transformed_mask, input_perturbations, V, U)

        p_transformed = p_transformed_qk + p_transformed_vo

        #############################################
        # Add the perturbations together and apply the LayerNorm.

        return self.layer_norm_state.transform_pre_layer_norm_perturbations(p_transformed + p_residual + p_induced)

    def compute_total_output_perturbation(self, input_perturbations: tf.Tensor) -> tf.Tensor:
        """The entire perturbation in the hidden states after this layer including the LayerNorm.
        
        NOTE: This DOES include the output_bias_perturbations.
        TODO: Might need to take into account the effect of some of the biases of the values/outputs!
        
        input_perturbations.shape = [batch, seq_len, d_model]

        ret.shape = [batch, seq_len, d_model]
        """
        batch_size = tf.shape(input_perturbations)[0]
        seq_len = tf.shape(input_perturbations)[1]
        n_heads = self.activations_info.num_attention_heads

        # Make dummy masks.
        mask = tf.ones([batch_size, n_heads, seq_len, seq_len], dtype=tf.float32)
        residual_mask = tf.ones([batch_size, seq_len], dtype=tf.float32)

        ret = self.compute_layer_output_perturbation_masked(
            input_perturbations,
            induced_mask=mask, transformed_mask=mask, residual_mask=residual_mask)
        
        # Since the compute_layer_output_perturbation_masked function does NOT include the output bias perturbation
        # term, we need to add it.
        #
        # NOTE: This implicitly assumes that the transform_pre_layer_norm_perturbations function
        # is linear. Since it is just matrix multiplication, it should be.
        extra_perturbation = self.layer_norm_state.transform_pre_layer_norm_perturbations(self.output_bias_perturbations)
        return ret + extra_perturbation


@dataclasses.dataclass(eq=False)
class FfwState:
    """Information about a FFW transformer layer's activations and perturbations.
    
    The information is stored for all sequence positions although only the [CLS]
    token's information will be used.
    """
    activations_info: BertFfwDenseActivations

    # Information about the LayerNorm operating on the output of this residual block.
    layer_norm_state: LayerNormState

    # Perturbation term corresponding to perturbation of the first layer.
    # perturbation_1.shape = [batch_size, seq_len, d_model]
    perturbation_1: tf.Tensor

    # Perturbation term corresponding to perturbation of the first layer at
    # the level of the outputs of the first layer. We'd have perturbation_1 equal
    # to the kernel of the second layer transforming perturbation_1_intermediate.
    # 
    # perturbation_1_intermediate.shape = [batch_size, seq_len, d_ff]
    perturbation_1_intermediate: tf.Tensor

    # Perturbation term corresponding to perturbation of the second layer.
    # perturbation_2.shape = [batch_size, seq_len, d_model]
    perturbation_2: tf.Tensor

    # Parameter perturbations of the second layer.
    # P2.shape = [d_ff, d_model]
    # p2.shape = [d_model]
    P2: tf.Tensor
    p2: tf.Tensor

    #######################################################

    @property
    def sequence_length(self) -> int:
        return self.perturbation_1.shape[-2]

    @property
    def d_ff(self) -> int:
        return self.perturbation_1_intermediate.shape[-1]

    #######################################################

    def _transform_input_perturbations_to_pre_layer_norm(self, input_perturbations: tf.Tensor) -> tf.Tensor:
        # The output is pre-LayerNorm.
        # input_perturbations.shape = [batch..., seq_len, d_model]
        # ret.shape = [batch..., seq_len, d_model]
        d_act_fn = self.activations_info.d_act_fn
        W1 = self.activations_info.layer1.layer.kernel
        W2 = self.activations_info.layer2.layer.kernel

        t0 = tf.einsum('...sm,mi->...si', input_perturbations, W1)
        t0 = t0 * d_act_fn(self.activations_info.layer1.output_preactivations)
        return tf.einsum('...si,im->...sm', t0, W2)

    def _compute_induced_perturbations_with_units_mask(self, induced_mask: tf.Tensor, induced_units_mask: tf.Tensor):
        # induced_mask_mask.shape = [batch, seq_len, 1]
        # induced_units_mask.shape = [batch, d_ff]
        assert induced_units_mask is not None

        induced_units_mask = tf.cast(induced_units_mask, tf.float32)

        W2 = self.activations_info.layer2.layer.kernel

        perturbation_1 = tf.einsum('bi,bsi,im->bsm',
                                   induced_units_mask, self.perturbation_1_intermediate, W2)

        perturbation_2 = tf.einsum('bi,bsi,im->bsm',
                                   induced_units_mask, self.activations_info.layer2.input_activations, self.P2)
        perturbation_2 = perturbation_2 + self.p2

        return induced_mask * (perturbation_1 + perturbation_2)

    #######################################################

    def get_pre_layer_norm_induced_perturbation(self):
        return self.perturbation_1 + self.perturbation_2

    def get_post_layer_norm_induced_perturbation(self):
        return self.layer_norm_state.transform_pre_layer_norm_perturbations(self.perturbation_1 + self.perturbation_2)

    #######################################################

    def compute_layer_output_perturbation_masked(
            self,
            input_perturbations: tf.Tensor,
            induced_mask: tf.Tensor, transformed_mask: tf.Tensor, residual_mask: tf.Tensor,
            induced_units_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """The entire perturbation in the hidden states after this layer including the LayerNorm.
        
        input_perturbations.shape = [batch, seq_len, d_model]
        *_mask.shape = [batch, seq_len]
        induced_units_mask.shape = [batch, d_ff]
        
        ret.shape = [batch..., seq_len, d_model]
        """
        induced_mask = tf.cast(induced_mask, tf.float32)[..., None]
        transformed_mask = tf.cast(transformed_mask, tf.float32)[..., None]
        residual_mask = tf.cast(residual_mask, tf.float32)[..., None]

        p_transformed = transformed_mask * self._transform_input_perturbations_to_pre_layer_norm(input_perturbations)
        p_residual = residual_mask * input_perturbations

        if induced_units_mask is None:
            p_induced = induced_mask * (self.perturbation_1 + self.perturbation_2)
        else:
            p_induced = self._compute_induced_perturbations_with_units_mask(induced_mask, induced_units_mask)

        return self.layer_norm_state.transform_pre_layer_norm_perturbations(p_transformed + p_residual + p_induced)

    def compute_total_output_perturbation(self, input_perturbations: tf.Tensor) -> tf.Tensor:
        """The entire perturbation in the hidden states after this layer including the LayerNorm.
        
        input_perturbations.shape = [batch, seq_len, d_model]

        ret.shape = [batch, seq_len, d_model]
        """
        # Use a dummy mask.
        mask = tf.ones(tf.shape(input_perturbations)[:2], dtype=tf.float32)
        return self.compute_layer_output_perturbation_masked(
            input_perturbations,
            induced_mask=mask, transformed_mask=mask, residual_mask=mask)
        

###############################################################################


@dataclasses.dataclass(eq=False)
class BertState:
    """Represents the state of BERT activations across the residual blocks.

    A residual block represents either a self-attention layer or a FFW layer. The
    total number of residual blocks will equal twice the number of "transformer" layers.
    """
    bert_config: BertConfig

    # Information about the pooler.
    pooler_state: PoolerState

    # Information about the classifier.
    classifier_state: ClassifierState

    # The inputs to the BERT model.
    inputs: Dict[str, tf.Tensor]

    # Mask indicating which tokens are padding. Padding tokens have a zero while
    # non-padding tokens have 1.
    # attention_mask.shape = [batch_size, sequence_length]
    # attention_mask.dtype = tf.int32
    attention_mask: tf.Tensor

    # The initial embeddings.
    # embedding.shape = [batch_size, sequence_length, d_model]
    embeddings: tf.Tensor

    # Information about the activations and perturbations for each residual block.
    residual_blocks: List[Union[FfwState, AttentionState]]

    # Index of the cls token in the sequence. This probably won't need to be set to
    # anything other than default for most usecases that I can see.
    cls_token_index: int = 0

    # Perturbation at the last layer in the residual blocks. Used when constructing
    # this.
    _last_layer_perturbation: tf.Tensor = None

    def get_batch_size(self) -> int:
        # Doesn't check for consistency or anything.
        return self.attention_mask.shape[0]

    def get_n_non_padding(self) -> tf.Tensor:
        # ret.shape = [batch_size], dtype=tf.int32
        return tf.reduce_sum(self.attention_mask, axis=-1)

    def get_pooler_input_fisher(self) -> PoolerInputFisher:
        return PoolerInputFisher.create_from_states(
            pooler_state=self.pooler_state,
            classifier_state=self.classifier_state,
        )

    def get_attention_state_for_layer(self, layer_index: int):
        assert layer_index >= 0
        return self.residual_blocks[2 * layer_index]

    def get_ffw_state_for_layer(self, layer_index: int):
        assert layer_index >= 0
        return self.residual_blocks[2 * layer_index + 1]
