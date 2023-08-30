"""Computation of the things in the bert_states file."""
import dataclasses
import math
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from transformers import BertConfig, TFBertForSequenceClassification
from transformers.models.bert import modeling_tf_bert as hf

from npeff.decomp import decomps
from npeff.util import flat_pack

from npeff_tracer.util import monkey_patching

from . import bert_states
from . import bert_states_util

###############################################################################

MonkeyPatcherContext = monkey_patching.MonkeyPatcherContext

d_tanh = bert_states_util.d_tanh
d_gelu = bert_states_util.d_gelu

BertDenseActivations = bert_states_util.BertDenseActivations
BertAttentionDenseActivations = bert_states_util.BertAttentionDenseActivations
BertFfwDenseActivations = bert_states_util.BertFfwDenseActivations

PoolerState = bert_states.PoolerState
ClassifierState = bert_states.ClassifierState
PoolerInputFisher = bert_states.PoolerInputFisher
LayerNormState = bert_states.LayerNormState
AttentionState = bert_states.AttentionState
FfwState = bert_states.FfwState
BertState = bert_states.BertState


###############################################################################
# Private classes

@dataclasses.dataclass
class _BertDenseActivationsCollection:
    inputs: Optional[Dict[str, tf.Tensor]] = None
    attention_mask: Optional[tf.Tensor] = None

    activations: Optional[List[BertDenseActivations]] = None
    layer_norms: Optional[List[LayerNormState]] = None

    def __post_init__(self):
        if self.activations is None:
            self.activations = []
        if self.layer_norms is None:
            self.layer_norms = []

    def get_extended_attention_mask(self):
        input_shape = tf.shape(self.attention_mask)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = tf.reshape(self.attention_mask, (input_shape[0], 1, 1, input_shape[1]))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tf.cast(extended_attention_mask, dtype=tf.float32)
        extended_attention_mask = -10000.0 * (1 - extended_attention_mask)

        return extended_attention_mask

    def get_attention_dense_activations(self, attention_layer_index: int) -> BertAttentionDenseActivations:
        acts = [a for a in self.activations if a.layer_index == attention_layer_index]
        key, = [a for a in acts if a.is_attention_key()]
        query, = [a for a in acts if a.is_attention_query()]
        value, = [a for a in acts if a.is_attention_value()]
        output, = [a for a in acts if a.is_attention_output()]
        return BertAttentionDenseActivations(
            config=key.config,
            key=key,
            query=query,
            value=value,
            output=output,
        )

    def get_ffw_dense_activations(self, ffw_layer_index: int) -> BertFfwDenseActivations:
        acts = [a for a in self.activations if a.layer_index == ffw_layer_index]
        layer1, = [a for a in acts if a.is_ffw_layer1()]
        layer2, = [a for a in acts if a.is_ffw_layer2()]
        return BertFfwDenseActivations(config=layer1.config, layer1=layer1, layer2=layer2)

    def get_pooler_dense_activations(self) -> BertDenseActivations:
        pooler, = [a for a in self.activations if a.is_pooler()]
        return pooler

    def get_classifier_dense_activations(self) -> BertDenseActivations:
        pooler, = [a for a in self.activations if a.is_classifier()]
        return pooler

###############################################################################


@dataclasses.dataclass
class BertStateComputer:
    nmf: Union[decomps.LrmNpeffDecomposition, decomps.LazyLoadedLrmNpeffDecomposition]

    model: TFBertForSequenceClassification
    variables: List[tf.Variable]

    component_index: Optional[int] = None

    def __post_init__(self):
        self.config = self.model.config
        self.n_transformer_layers = self.model.config.num_hidden_layers

        self.packer = flat_pack.FlatPacker([v.shape for v in self.variables])

        self.monkey_patcher = MonkeyPatcherContext()
        self.monkey_patcher.patch_method(
            tf.keras.layers.Dense, "__call__", self._dense_layer_call
        )
        self.monkey_patcher.patch_method(
            tf.keras.layers.LayerNormalization, "__call__", self._layer_norm_call
        )

        # Storage containers used to hold things during the monkey-patched model call.
        self._dense_activations_collection = None

        # Additional things used to store component-specific information.
        self.g = None
        self.var_name_to_g = None

        if self.component_index is not None:
            self.set_component(self.component_index)

    def _dense_layer_call(self, og_call, layer, *args, **kwargs):
        layer_name = layer.name_scope()

        inputs = kwargs['inputs'] if 'inputs' in kwargs else args[0]
        output_preacts = tf.einsum('...i,ij->...j', inputs, layer.kernel) + layer.bias

        # Create and store information about the activations and such.
        bda = BertDenseActivations(
            config=self.config,
            layer_name=layer_name,
            layer=layer,
            input_activations=inputs,
            output_preactivations=output_preacts,
        )
        self._dense_activations_collection.activations.append(bda)

        return layer.activation(output_preacts)

    def _layer_norm_call(self, og_call, layer, *args, **kwargs):
        layer_name = layer.name_scope()

        # This filters out the LayerNorms that we do not currently care about.
        if '/bert/encoder/layer_._' not in layer_name:
            return og_call(layer, *args, **kwargs)

        inputs = kwargs['inputs'] if 'inputs' in kwargs else args[0]

        means, variances = tf.nn.moments(inputs, layer.axis)

        state = LayerNormState(
            epsilon=layer.epsilon,
            gamma=layer.gamma,
            beta=layer.beta,
            means=means,
            variances=variances,
        )
        self._dense_activations_collection.layer_norms.append(state)

        return og_call(layer, *args, **kwargs)

    def _get_g_for_component(self, component_index: int) -> List[tf.Tensor]:
        # g = np.zeros([self.nmf.n_parameters], dtype=np.float32)
        # g[self.nmf.new_to_old_col_indices] = self.nmf.G[component_index]
        g = self.nmf.get_full_normalized_g(component_index)
        return self.packer.decode_tf(tf.cast(g, tf.float32))

    def set_component(self, component_index: int):
        self.component_index = component_index
        self.g = self._get_g_for_component(self.component_index)
        self.var_name_to_g = {v.name: g for v, g in zip(self.variables, self.g)}

    def call(self, *args, **kwargs):
        self._dense_activations_collection = _BertDenseActivationsCollection()
        inputs = kwargs['inputs'] if 'inputs' in kwargs else args[0]
        self._dense_activations_collection.inputs = inputs
        self._dense_activations_collection.attention_mask = inputs['attention_mask']
        with self.monkey_patcher:
            return self.model(*args, **kwargs)

    # def _get_gs_belonging_to_dense_layer(self, layer: Union[str, BertDenseActivations]):
    #     layer_name = layer if isinstance(layer, str) else layer.layer_name
    #     gs = {k: v for k, v in self.var_name_to_g.items() if k.startswith(layer_name)}
    #     kernel, = [v for k, v in gs.items() if 'kernel' in k]
    #     bias, = [v for k, v in gs.items() if 'bias' in k]
    #     return kernel, bias

    def _get_gs_belonging_to_dense_layer(self, layer: Union[str, BertDenseActivations]):
        layer_name = layer if isinstance(layer, str) else layer.layer_name
        gs = {k: v for k, v in self.var_name_to_g.items() if k.startswith(layer_name)}
        
        if gs:
            kernel, = [v for k, v in gs.items() if 'kernel' in k]
            bias, = [v for k, v in gs.items() if 'bias' in k]
        else:
            if isinstance(layer, str):
                raise NotImplementedError('TODO: Support this')
            # This is for when the we are doing NPEFF for only a subset of variables.
            kernel = tf.zeros_like(layer.layer.kernel)
            bias = tf.zeros_like(layer.layer.bias)

        return kernel, bias

    def _get_multihead_query_gs(self, att_acts: BertAttentionDenseActivations):
        P_Q, p_q = self._get_gs_belonging_to_dense_layer(att_acts.query)
        # The shapes are the same, so the *_value_* methods here are NOT a typo.
        P_Q = att_acts.multihead_value_kernel(P_Q)
        p_q = att_acts.multihead_value_bias(p_q)
        return P_Q, p_q

    def _get_multihead_key_gs(self, att_acts: BertAttentionDenseActivations):
        P_K, p_k = self._get_gs_belonging_to_dense_layer(att_acts.key)
        # The shapes are the same, so the *_value_* methods here are NOT a typo.
        P_K = att_acts.multihead_value_kernel(P_K)
        p_k = att_acts.multihead_value_bias(p_k)
        return P_K, p_k

    def _get_multihead_value_gs(self, att_acts: BertAttentionDenseActivations):
        P_V, p_v = self._get_gs_belonging_to_dense_layer(att_acts.value)
        P_V = att_acts.multihead_value_kernel(P_V)
        p_v = att_acts.multihead_value_bias(p_v)
        return P_V, p_v

    def _get_multihead_output_gs(self, att_acts: BertAttentionDenseActivations):
        # Was the use of value here (commented out line) a bug?
        # P_U, p_u = self._get_gs_belonging_to_dense_layer(att_acts.value)
        P_U, p_u = self._get_gs_belonging_to_dense_layer(att_acts.output)
        P_U = att_acts.multihead_output_kernel(P_U)
        p_u = att_acts.multihead_output_bias(p_u)
        return P_U, p_u

    def _add_attention_layer_state(self, state: BertState, transformer_layer_index: int):
        att_acts = self._dense_activations_collection.get_attention_dense_activations(transformer_layer_index)

        # The QKV will all have the same inputs, so choose one arbitrarily.
        input_activations = att_acts.query.input_activations

        # If this is the first layer, we need to collect the embeddings.
        if state.embeddings is None:
            assert transformer_layer_index == 0
            state.embeddings = input_activations
            state._last_layer_perturbation = tf.zeros_like(state.embeddings)

        input_perturbations = state._last_layer_perturbation

        ###################################################
        # Create the attention map.

        # shapes = [batch_size, num_heads, seq_length, attention_head_size]
        queries = att_acts.transpose_for_scores(att_acts.query.output_preactivations)
        keys = att_acts.transpose_for_scores(att_acts.key.output_preactivations)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch_size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(queries, keys, transpose_b=True)
        attention_scores = attention_scores / math.sqrt(att_acts.attention_head_size)

        # Apply the attention mask.
        attention_mask = self._dense_activations_collection.get_extended_attention_mask()
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # attention_map.shape = [batch_size, num_heads, seq_len_q, seq_len_k]
        attention_map = tf.nn.softmax(logits=attention_scores, axis=-1)

        ###################################################
        # Value-output perturbations.

        V = att_acts.multihead_value_kernel()
        v = att_acts.multihead_value_bias()
        U = att_acts.multihead_output_kernel()

        P_V, p_v = self._get_multihead_value_gs(att_acts)
        P_U, p_u = self._get_multihead_output_gs(att_acts)

        value_perturbations = tf.einsum('bkm,mja,jan->bjkn', input_activations, P_V, U)
        value_perturbations = value_perturbations + tf.einsum('ja,jan->jn', p_v, U)[None, :, None, :]

        output_perturbations = tf.einsum('bkm,mja,jan->bjkn', input_activations, V, P_U)
        output_perturbations = output_perturbations + tf.einsum('ja,jan->jn', v, P_U)[None, :, None, :]

        output_bias_perturbations = p_u

        ###################################################
        # Query-key perturbations.

        rsqrt_d_head = tf.math.rsqrt(tf.cast(tf.shape(queries)[-1], tf.float32))

        # kernel shapes = [d_model, num_attention_heads, attention_head_size]
        P_Q, p_q = self._get_multihead_query_gs(att_acts)
        P_K, p_k = self._get_multihead_key_gs(att_acts)

        # shapes = [batch_size, num_heads, seq_len_q, seq_len_k]
        query_logit_perturbations = rsqrt_d_head * tf.matmul(
            tf.einsum('bsh,hnd->bnsd', input_activations, P_Q) + tf.expand_dims(p_q, axis=-2),
            keys,
            transpose_b=True)
        key_logit_perturbations = rsqrt_d_head * tf.matmul(
            queries,
            tf.einsum('bsh,hnd->bnsd', input_activations, P_K) + tf.expand_dims(p_k, axis=-2),
            transpose_b=True)

        # TODO: Might need to take into account the effect of some of the biases of the values/outputs!

        ###################################################
        # Create the attention state object.

        layer_norm_state = self._dense_activations_collection.layer_norms[len(state.residual_blocks)]

        att_state = AttentionState(
            query_logit_perturbations=query_logit_perturbations,
            key_logit_perturbations=key_logit_perturbations,
            value_perturbations=value_perturbations,
            output_perturbations=output_perturbations,
            output_bias_perturbations=output_bias_perturbations,
            activations_info=att_acts,
            attention_map=attention_map,
            layer_norm_state=layer_norm_state,
            P_Q=P_Q,
            P_K=P_K,
            P_V=P_V,
            p_v=p_v,
            P_U=P_U,
        )

        # Update the perturbation that gets passed through the layers.
        state._last_layer_perturbation = att_state.compute_total_output_perturbation(input_perturbations)

        state.residual_blocks.append(att_state)

    def _add_ffw_layer_state(self, state: BertState, transformer_layer_index: int):
        ffw_acts = self._dense_activations_collection.get_ffw_dense_activations(transformer_layer_index)

        input_activations = ffw_acts.layer1.input_activations
        input_perturbations = state._last_layer_perturbation

        d_act_fn = ffw_acts.d_act_fn

        # W1 = ffw_acts.layer1.layer.kernel
        W2 = ffw_acts.layer2.layer.kernel

        P1, p1 = self._get_gs_belonging_to_dense_layer(ffw_acts.layer1)
        P2, p2 = self._get_gs_belonging_to_dense_layer(ffw_acts.layer2)

        # ###################################################
        # # Compute transformed_input_perturbations
        # t0 = tf.einsum('bsm,mi->bsi', input_perturbations, W1)
        # t0 = t0 * d_act_fn(ffw_acts.layer1.output_preactivations)

        ###################################################
        # Compute perturbation_1
        t1 = tf.einsum('bsm,mi->bsi', input_activations, P1)
        # t1 = t1 + tf.einsum('bsm,mi->bsi', input_perturbations, W1)
        t1 = t1 + p1
        t1 = t1 * d_act_fn(ffw_acts.layer1.output_preactivations)
        perturbation_1_intermediate = t1
        perturbation_1 = tf.einsum('bsi,im->bsm', t1, W2)

        ###################################################
        # Compute perturbation_2

        perturbation_2 = tf.einsum('bsi,im->bsm', ffw_acts.layer2.input_activations, P2)
        perturbation_2 = perturbation_2 + p2

        ###################################################
        # Create the FFW state object.

        layer_norm_state = self._dense_activations_collection.layer_norms[len(state.residual_blocks)]

        ffw_state = FfwState(
            activations_info=ffw_acts,
            layer_norm_state=layer_norm_state,
            perturbation_1=perturbation_1,
            perturbation_1_intermediate=perturbation_1_intermediate,
            perturbation_2=perturbation_2,
            P2=P2,
            p2=p2,
        )

        # Update the perturbation that gets passed through the layers.
        state._last_layer_perturbation = ffw_state.compute_total_output_perturbation(input_perturbations)

        state.residual_blocks.append(ffw_state)

    def create_bert_state(self, component_index: Optional[int] = None) -> BertState:
        """Creates bert state corresponding to the activations stored during the call fn."""
        if component_index is not None:
            self.set_component(component_index)

        # Make sure that the component has been set somewhere.
        assert self.component_index is not None

        pooler = self._dense_activations_collection.get_pooler_dense_activations()
        pooler_state = PoolerState(
            layer=pooler.layer,
            input_activations=pooler.input_activations,
            output_preactivations=pooler.output_preactivations,
        )

        classifier = self._dense_activations_collection.get_classifier_dense_activations()
        classifier_state = ClassifierState(
            layer=classifier.layer,
            input_activations=classifier.input_activations,
            output_preactivations=classifier.output_preactivations,
        )

        state = BertState(
            bert_config=self.config,
            pooler_state=pooler_state,
            classifier_state=classifier_state,
            inputs=self._dense_activations_collection.inputs,
            attention_mask=self._dense_activations_collection.attention_mask,
            embeddings=None,
            residual_blocks=[],
        )

        for transformer_layer_index in range(self.n_transformer_layers):
            self._add_attention_layer_state(state, transformer_layer_index)
            self._add_ffw_layer_state(state, transformer_layer_index)

        return state
