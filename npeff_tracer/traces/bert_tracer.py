"""Tracing of perturbations for a BERT model."""
import collections
import dataclasses
import json
import os
from typing import Dict, List

import h5py
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from transformers import BertTokenizer

from npeff.util import hdf5_util

from npeff_tracer.states import bert_states
from npeff_tracer.states import bert_states_util

from . import bert_graph
from . import bert_masked_evaluation

###############################################################################

PoolerInputFisher = bert_states.PoolerInputFisher
AttentionState = bert_states.AttentionState
FfwState = bert_states.FfwState
BertState = bert_states.BertState

AttentionLayerMask = bert_masked_evaluation.AttentionLayerMask
FfwLayerMask = bert_masked_evaluation.FfwLayerMask
MaskedEvaluator = bert_masked_evaluation.MaskedEvaluator

EvaluableGraph = bert_graph.EvaluableGraph

###############################################################################

# Typedefs
NodeIndex = bert_graph.NodeIndex
EdgeIndex = bert_graph.EdgeIndex

###############################################################################


@dataclasses.dataclass
class BertTrace:
    # The list of non-padding tokens of the example.
    tokens: List[str]
    n_transformer_layers: int
    n_attention_heads: int

    # shape of np.ndarrays = [n_classes]
    induced_edge_contributions: Dict[EdgeIndex, np.ndarray]

    # shape of np.ndarrays = [n_classes]
    subtree_contributions: Dict[EdgeIndex, Dict[EdgeIndex, np.ndarray]]

    # Helper for serialization of edge types.
    _EDGE_TYPE_SERIALIZATION = ('residual', 'transformed', 'induced')

    def save(self, filepath: str):
        # Create metadata.

        metadata = json.dumps({
            'tokens': self.tokens,
            'n_transformer_layers': int(self.n_transformer_layers),
            'n_attention_heads': int(self.n_attention_heads),
        })

        # Save the induced_edge_contributions.

        iec_keys = list(sorted(self.induced_edge_contributions.keys()))

        iec_keys_array = np.array([[*src, *dst] for src, dst, _ in iec_keys], dtype=np.int32)
        iec_values_array = np.stack([self.induced_edge_contributions[k] for k in iec_keys], axis=0)

        # Save the subtree_contributions.

        stc_outer_keys = list(sorted(self.subtree_contributions.keys()))

        stc_inner_sizes_array = []
        stc_inner_keys_array = []
        stc_values_array = []

        for outer_key in stc_outer_keys:
            inner_dict = self.subtree_contributions[outer_key]
            stc_inner_sizes_array.append(len(inner_dict))

            inner_keys = list(sorted(inner_dict.keys()))

            stc_inner_keys_array.extend(
                [*src, *dst, self._edge_type_to_int(edge_type)]
                for src, dst, edge_type in inner_keys
            )
            stc_values_array.extend(inner_dict[k] for k in inner_keys)

        stc_outer_keys_array = np.array([[*src, *dst] for src, dst, _ in stc_outer_keys], dtype=np.int32)
        stc_inner_sizes_array = np.array(stc_inner_sizes_array, dtype=np.int32)
        stc_inner_keys_array = np.array(stc_inner_keys_array, dtype=np.int32)
        stc_values_array = np.stack(stc_values_array, axis=0)

        # Save to h5.

        with h5py.File(os.path.expanduser(filepath), "w") as f:
            data_group = f.create_group('data')
            data_group.attrs['metadata'] = metadata

            iec_group = data_group.create_group('induced_edge_contributions')
            hdf5_util.save_h5_ds(iec_group, 'keys', iec_keys_array)
            hdf5_util.save_h5_ds(iec_group, 'values', iec_values_array)

            stc_group = data_group.create_group('subtree_contributions')
            hdf5_util.save_h5_ds(stc_group, 'outer_keys', stc_outer_keys_array)
            hdf5_util.save_h5_ds(stc_group, 'inner_sizes', stc_inner_sizes_array)
            hdf5_util.save_h5_ds(stc_group, 'inner_keys', stc_inner_keys_array)
            hdf5_util.save_h5_ds(stc_group, 'values', stc_values_array)

    @classmethod
    def _edge_type_to_int(cls, edge_type: str) -> int:
        return cls._EDGE_TYPE_SERIALIZATION.index(edge_type)

    @classmethod
    def _edge_type_from_int(cls, edge_type: int) -> str:
        return cls._EDGE_TYPE_SERIALIZATION[int(edge_type)]

    @classmethod
    def load(cls, filepath: str) -> 'BertTrace':
        # Load arrays and metadata directly from the h5 file.
        with h5py.File(os.path.expanduser(filepath), "r") as f:
            metadata = json.loads(f['data'].attrs['metadata'])

            iec_keys_array = hdf5_util.load_h5_ds(f['data/induced_edge_contributions/keys'])
            iec_values_array = hdf5_util.load_h5_ds(f['data/induced_edge_contributions/values'])

            stc_outer_keys_array = hdf5_util.load_h5_ds(f['data/subtree_contributions/outer_keys'])
            stc_inner_sizes_array = hdf5_util.load_h5_ds(f['data/subtree_contributions/inner_sizes'])
            stc_inner_keys_array = hdf5_util.load_h5_ds(f['data/subtree_contributions/inner_keys'])
            stc_values_array = hdf5_util.load_h5_ds(f['data/subtree_contributions/values'])

        # Create the induced_edge_contributions.
        induced_edge_contributions = {}
        for k, v in zip(iec_keys_array, iec_values_array):
            src = tuple(int(x) for x in k[0:2])
            dst = tuple(int(x) for x in k[2:4])
            edge = (src, dst, 'induced')
            induced_edge_contributions[edge] = v

        # Create the subtree_contributions.
        subtree_contributions = {}
        inner_offset = 0
        for ok, size in zip(stc_outer_keys_array, stc_inner_sizes_array):
            o_src = tuple(int(x) for x in ok[0:2])
            o_dst = tuple(int(x) for x in ok[2:4])
            o_edge = (o_src, o_dst, 'induced')

            inner_keys = stc_inner_keys_array[inner_offset : inner_offset + size]
            inner_values = stc_values_array[inner_offset : inner_offset + size]

            inner_dict = {}
            for ik, iv in zip(inner_keys, inner_values):
                i_src = tuple(int(x) for x in ik[0:2])
                i_dst = tuple(int(x) for x in ik[2:4])
                i_edge_type = cls._edge_type_from_int(ik[4])
                i_edge = (i_src, i_dst, i_edge_type)
                inner_dict[i_edge] = iv

            subtree_contributions[o_edge] = inner_dict

            inner_offset += size

        # Create the instance.
        return cls(
            tokens=metadata['tokens'],
            n_transformer_layers=metadata['n_transformer_layers'],
            n_attention_heads=metadata['n_attention_heads'],
            induced_edge_contributions=induced_edge_contributions,
            subtree_contributions=subtree_contributions,
        )


###############################################################################


@dataclasses.dataclass
class BertTracer:
    masked_evaluator: MaskedEvaluator

    tokenizer: BertTokenizer

    use_tqdm: bool = True

    def __post_init__(self):
        # self.cls_state = self.masked_evaluator.cls_state
        self.graph = EvaluableGraph(masked_evaluator=self.masked_evaluator)

    #######################################################

    def reset(self):
        self.graph.reset()

    #######################################################
    
    def _compute_per_edge_induced(self) -> Dict[EdgeIndex, tf.Tensor]:
        self.reset()
        self.graph.mask_all_of_type('induced')

        # NOTE: The reverse topological sort not really needed here.
        nodes = self.graph.get_reverse_topologically_sorted_nodes()

        edge_to_dps = {}

        for node in tqdm(nodes) if self.use_tqdm else nodes:
            for parent in list(self.graph.parents['induced'][node]):
                self.graph.add_edge(parent, node, 'induced')
                dps = self.masked_evaluator.compute_total_pooler_fisher_dot_products_masked()
                self.graph.remove_edge(parent, node, 'induced')
                edge_to_dps[(parent, node, 'induced')] = dps

        return edge_to_dps

    def _sort_map_keys_by_magnitude(self, edge_to_dps):
        # Sorted in descending order of L2 magnitude.
        mags = {k: tf.reduce_sum(tf.square(v)).numpy() for k, v in edge_to_dps.items()}
        return list(sorted(list(edge_to_dps.keys()), key=lambda k: mags[k], reverse=True))

    def _compute_downstream_trace_for_edge(self, edge: EdgeIndex) -> Dict[EdgeIndex, tf.Tensor]:
        # The deltas are from the induced edge only with all propagative edges.
        src, dst, edge_type = edge
        assert edge_type == 'induced', 'Must be induced edge.'
        assert src in self.graph.full_parents['induced'][dst], 'Induced edge does not exist.'

        self.reset()
        self.graph.mask_all()
        self.graph.add_edge(src, dst, 'induced')
        self.graph.unmask_propagative_subtree_of_node(dst)

        base_fisher_dps = self.masked_evaluator.compute_total_pooler_fisher_dot_products_masked()

        edge_to_delta = {}
        subtree_edges = self.graph.get_edges_of_propagative_subtree_of_node(dst)
        for edge in tqdm(subtree_edges) if self.use_tqdm else subtree_edges:
            self.graph.remove_edge(*edge)
            trial_fisher_dps = self.masked_evaluator.compute_total_pooler_fisher_dot_products_masked()
            edge_to_delta[edge] = trial_fisher_dps - base_fisher_dps
            self.graph.add_edge(*edge)

        return edge_to_delta

    def compute_trace(self, n_downstream_traces: int) -> BertTrace:
        # Does NOT reset the network afterwards.
        induced_dps = self._compute_per_edge_induced()
        
        edges_by_mag_desc = self._sort_map_keys_by_magnitude(induced_dps)

        subtree_deltas = {}
        for edge in edges_by_mag_desc[:n_downstream_traces]:
            subtree_deltas[edge] = self._compute_downstream_trace_for_edge(edge)

        bert_state = self.graph.bert_state
        tokens = [
            self.tokenizer.convert_ids_to_tokens(int(token_id.numpy()))
            for token_id in tf.squeeze(bert_state.inputs['input_ids'], axis=0)
        ]

        return BertTrace(
            tokens=tokens,
            n_transformer_layers=bert_state.bert_config.num_hidden_layers,
            n_attention_heads=bert_state.bert_config.num_attention_heads,
            induced_edge_contributions=induced_dps,
            subtree_contributions=subtree_deltas,
        )
