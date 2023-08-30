"""Representation of BERT operating on an example as a maskable graph."""
import collections
import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from npeff_tracer.states import bert_states
from npeff_tracer.states import bert_states_util


from . import bert_masked_evaluation

###############################################################################

PoolerInputFisher = bert_states.PoolerInputFisher
AttentionState = bert_states.AttentionState
FfwState = bert_states.FfwState
BertState = bert_states.BertState

AttentionLayerMask = bert_masked_evaluation.AttentionLayerMask
FfwLayerMask = bert_masked_evaluation.FfwLayerMask
MaskedEvaluator = bert_masked_evaluation.MaskedEvaluator

###############################################################################

# Typedefs

# First is residual block index. Second is sequence position.
NodeIndex = Tuple[int, int]

# (src_node, dst_node, edge_type), where edge_type in EDGE_TYPES.
EdgeIndex = Tuple[NodeIndex, NodeIndex, str]

###############################################################################


def _parent_map_to_child_map(parent_map: Dict[Any, Set[Any]]) -> Dict[Any, Set[Any]]:
    """Reverses map from node to list of parents to map from node to list of children."""
    child_map = collections.defaultdict(set)
    for child, parents in parent_map.items():
        assert not isinstance(parents, str), 'Probably called this on the wrong object.'
        for parent in parents:
            child_map[parent].add(child)
    return child_map


###############################################################################


EDGE_TYPES = ('residual', 'transformed', 'induced')
PROPAGATIVE_EDGE_TYPES = ('residual', 'transformed')


def _assert_valid_edge_type(edge_type: str):
    if edge_type not in EDGE_TYPES:
        raise ValueError(f'Invalid edge type: {edge_type}')


@dataclasses.dataclass
class EvaluableGraph:
    """
    The graph will be something like three DAGs sharing a node set but with different edge sets. Alternately,
    we can think about it as a directed acyclic multigraph with three classes of edges: residual, transformed,
    and induced. 

    Nnodes correspond to each non-masked sequence position after each residual block. Hence they can
    effectively be indexed via the set [n_residual_blocks] x [n_non_padding], with 0-based indices, for an example. 
    However, the last two residual blocks will only have a single node each corresponding to the [CLS] token. There
    will also be a single dummy "output" node with index (n_residual_blocks, 0).

    Edges only include direct connections. Hence edges can only exist between nodes with indices (i, j) -> (i + 1, k).

    The masks in the MaskedEvaluator are associated with the source node of an edge from this graph representation.

    When I use "propagative", I mean the "residual" and "transformed" edge classes.

    """
    masked_evaluator: MaskedEvaluator

    def __post_init__(self):
        self.bert_state = self.masked_evaluator.bert_state
        self.layer_masks = self.masked_evaluator.layer_masks
        self.pooler_input_fisher = self.masked_evaluator.pooler_input_fisher

        self.cls_token_index = self.bert_state.cls_token_index

        self.n_residual_blocks = len(self.bert_state.residual_blocks)
        self.n_non_padding = int(self.bert_state.get_n_non_padding()[0].numpy())

        # The full_* versions of this won't get changed when stuff gets masked.
        self.full_nodes, self.full_parents, self.full_children = self._construct_full_graph()

        # NOTE: This will reset the masked_evaluator as well.
        self.reset()

    #######################################################

    def _construct_full_graph(self):
        # last_two_block_indices = (self.n_residual_blocks - 1, self.n_residual_blocks - 2)

        nodes = []
        # Map from edge_type to child node index to set of parents.
        parents = {t: collections.defaultdict(set) for t in EDGE_TYPES}

        # Set up stuff for the dummy output node.
        output_node = (self.n_residual_blocks, 0)
        nodes.append(output_node)
        #
        assert isinstance(self.bert_state.residual_blocks[-1], FfwState)
        for t in EDGE_TYPES:
            parents[t][output_node].add((self.n_residual_blocks - 1, self.cls_token_index))

        for block_index, block in enumerate(self.bert_state.residual_blocks):
            # block_seq_inds = range(self.n_non_padding) if block_index not in last_two_block_indices else [self.cls_token_index]
            block_seq_inds = [self.cls_token_index] if block_index == self.n_residual_blocks - 1 else range(self.n_non_padding)
            for seq_index in block_seq_inds:

                # Create the node.
                node_index = (block_index, seq_index)
                nodes.append(node_index)

                # The nodes in the first residual block do not have any parents.
                if block_index == 0:
                    continue

                parent_block = self.bert_state.residual_blocks[block_index - 1]

                if isinstance(parent_block, AttentionState):
                    parents['residual'][node_index].add((block_index - 1, seq_index))

                    for parent_seq_index in range(self.n_non_padding):
                        parents['transformed'][node_index].add((block_index - 1, parent_seq_index))
                        parents['induced'][node_index].add((block_index - 1, parent_seq_index))

                elif isinstance(parent_block, FfwState):
                    for t in EDGE_TYPES:
                        parents[t][node_index].add((block_index - 1, seq_index))

                else:
                    raise ValueError

        children = {t: _parent_map_to_child_map(v) for t, v in parents.items()}

        return nodes, parents, children

    def _construct_initial_graph(self):
        self.nodes, self.parents, self.children = self._construct_full_graph()

    #######################################################

    def get_reverse_topologically_sorted_nodes(self) -> List[NodeIndex]:
        nodes = list(self.nodes)
        nodes.sort(reverse=True)
        return nodes

    #######################################################

    def _set_masked_edge(self, src: NodeIndex, dst: NodeIndex, edge_type: str, value: float):
        mask_name = f'{edge_type}_mask'
        layer_mask = self.layer_masks[src[0]]

        if isinstance(layer_mask, AttentionLayerMask):
            if edge_type == 'residual':
                assert src[1] == dst[1]
                sequence_index = src[1]
                layer_mask.set_residual_sequence_position(sequence_index, value)
            else:
                # TODO: Double check that this correspondence between query/key and src/dst makes sense.
                _, query_sequence_index = dst
                _, key_sequence_index = src
                layer_mask.set_single_attention_position_across_heads(
                    mask_name, query_sequence_index, key_sequence_index, value)

        elif isinstance(layer_mask, FfwLayerMask):
            assert src[1] == dst[1]
            sequence_index = src[1]
            layer_mask.set_sequence_position(mask_name, sequence_index, value)

        else:
            raise ValueError

    #######################################################

    def reset(self):
        self.masked_evaluator.reset()
        self._construct_initial_graph()
        # TODO: Maybe more logic here as I add mode functionality here.

    def add_edge(self, src: NodeIndex, dst: NodeIndex, edge_type: str):
        # src <=> parent, dst <=> child
        # Does NOT do anything if the edge already exists.
        _assert_valid_edge_type(edge_type)
        self.parents[edge_type][dst].add(src)
        self.children[edge_type][src].add(dst)
        self._set_masked_edge(src, dst, edge_type, 1)

    def remove_edge(self, src: NodeIndex, dst: NodeIndex, edge_type: str):
        # src <=> parent, dst <=> child
        _assert_valid_edge_type(edge_type)
        self.parents[edge_type][dst].remove(src)
        self.children[edge_type][src].remove(dst)
        self._set_masked_edge(src, dst, edge_type, 0)

    def mask_all(self):
        # Remove all edges from the graph.
        self.parents = {t: collections.defaultdict(set) for t in EDGE_TYPES}
        self.children = {t: collections.defaultdict(set) for t in EDGE_TYPES}

        # Tell the masked evaluator to mask everthing.
        for layer_mask in self.layer_masks:
            layer_mask.mask_all()

    def mask_all_of_type(self, edge_type: str):
        # Remove edges from the graph.
        self.parents[edge_type] = collections.defaultdict(set)
        self.children[edge_type] = collections.defaultdict(set)

        # Tell the masked evaluator to mask everthing.
        mask_name = f'{edge_type}_mask'
        for layer_mask in self.layer_masks:
            layer_mask.mask_all_of_type(mask_name)

    #######################################################

    def is_output_node(self, node: NodeIndex) -> bool:
        return node[0] == self.n_residual_blocks

    def has_any_propagative_children(self, node: NodeIndex) -> bool:
        # Induced edges do NOT count as propagative.
        for edge_type in PROPAGATIVE_EDGE_TYPES:
            if len(self.children[edge_type][node]):
                return True
        return False

    #######################################################

    def get_edges_of_propagative_subtree_of_node(self, node: NodeIndex):
        edges = []

        # Do a BFS.
        visited = set()
        queue = collections.deque([node])

        while queue:
            current = queue.popleft()

            if current in visited:
                continue

            visited.add(current)

            for edge_type in PROPAGATIVE_EDGE_TYPES:
                for child in self.full_children[edge_type][current]:
                    queue.append(child)
                    edges.append((current, child, edge_type))

        return edges

    #######################################################

    def unmask_propagative_subtree_of_node(self, node: NodeIndex):
        subtree_edges = self.get_edges_of_propagative_subtree_of_node(node)
        for src, dst, edge_type in subtree_edges:
            self.add_edge(src, dst, edge_type)

