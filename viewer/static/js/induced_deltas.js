// Data structures and processing related to induced deltas.

///////////////////////////////////////////////////////////////////////////////
// Some utility functions.

function sq_l2_magnitude(a) {
    let ret = 0.0;
    for(const x of a) {
        ret += x**2;
    }
    return ret;
}

function l1_magnitude(a) {
    let ret = 0.0;
    for(const x of a) {
        ret += Math.abs(x);
    }
    return ret;
}


///////////////////////////////////////////////////////////////////////////////

class EdgeDeltaMagnitudes {

    constructor(connection_magnitudes) {
        this.connection_magnitudes = connection_magnitudes;
    }

    ///////////////////////////////////////////////////////

    n_classes() {
        return this.connection_magnitudes[0].delta.length;
    }

    ///////////////////////////////////////////////////////

    clone() {
        const connection_magnitudes = [];
        for(const item of this.connection_magnitudes) {
            connection_magnitudes.push(item);
        }
        return new EdgeDeltaMagnitudes(connection_magnitudes);
    }

    ///////////////////////////////////////////////////////

    filter_by_minimum_magnitude(min_magnitude) {
        this.connection_magnitudes = this.connection_magnitudes.filter(x => x.magnitude >= min_magnitude);
    }

    ///////////////////////////////////////////////////////

    get_all_connections_of_type_for_transformer_layer(transformer_layer_index, res_block_type, edge_type) {
        return this.connection_magnitudes.filter(
            x => x.transformer_layer_index === transformer_layer_index
                    && x.res_block_type === res_block_type
                    && x.edge_type === edge_type
        );
    }

    find_connection_for_edge(edge) {
        const edge_str_key = JSON.stringify(edge);
        return this.connection_magnitudes.find(x => x.edge_str_key == edge_str_key);
    }

    ///////////////////////////////////////////////////////

    static from_edge_deltas_list(edge_deltas_list, magnitude_fn = sq_l2_magnitude) {
        const connection_magnitudes = [];

        for(const {key, value} of edge_deltas_list) {
            const magnitude = magnitude_fn(value);
            const [src, dst, edge_type] = key;

            const res_block_index = src[0];
            const res_block_type = (res_block_index % 2) ? 'ffw' : 'attention';
            const transformer_layer_index = (res_block_index - (res_block_index % 2)) / 2;

            connection_magnitudes.push({
                'edge_type': edge_type,
                'res_block_type': res_block_type,
                'transformer_layer_index': transformer_layer_index,
                // Sequence indices.
                'src': src[1],
                'dst': dst[1],
                // The delta in pooler fisher dot products of the edge.
                'delta': value,
                // Magnitude of the delta according to magnitude_fn.
                'magnitude': magnitude,
                // Helpful when filtering.
                'edge_str_key': JSON.stringify(key),
            });
        }

        return new EdgeDeltaMagnitudes(connection_magnitudes);
    }
}


///////////////////////////////////////////////////////////////////////////////

// For the induced deltas, they are the NEGATIVE of the contribution of that induced edge.


class SubtreeDeltas {
    constructor(induced_delta_mags, root_to_subtree_delta_mags) {
        this.induced_delta_mags = induced_delta_mags;
        this.root_to_subtree_delta_mags = root_to_subtree_delta_mags;

        this.root_to_connection_item = {}
        for(const edge of Object.keys(this.root_to_subtree_delta_mags)) {
            const connection = this.induced_delta_mags.find_connection_for_edge(JSON.parse(edge));
            this.root_to_connection_item[edge] = connection;
        }

        // These should all be induced edges. Will be sorted in descending order of
        // associated magnitude.
        this.subtree_root_edges = Object.keys(this.root_to_subtree_delta_mags);
        this.subtree_root_edges.sort((a, b) => this.root_to_connection_item[b].magnitude - this.root_to_connection_item[a].magnitude);
        this.subtree_root_edges = this.subtree_root_edges.map(x => JSON.parse(x));
    }


    get_delta_mags_for_root_edge(edge) {
        const [src, dst, edge_type] = edge;
        if(edge_type !== 'induced') { throw Exception(edge_type); }

        const root_item = this.induced_delta_mags.find_connection_for_edge(edge);

        const ret = this.root_to_subtree_delta_mags[JSON.stringify(edge)].clone();
        ret.connection_magnitudes.push(root_item);

        return ret;
    }


    get_total_perturbation_delta() {
        // Returns the total perturbations measured by pooler Fisher dot products
        // of the FULL perturbation of this component. This wasn't saved, so we
        // compute it from the induced deltas.
        const n_classes = this.induced_delta_mags.n_classes();

        const ret = [];
        for(let i=0;i<n_classes;i++) { ret.push(0); }

        for(const {delta} of this.induced_delta_mags.connection_magnitudes) {
            // Each delta for an induced edge is the NEGATIVE of the contribution of
            // that induced edge. Hence we add its negative.
            for(let i=0;i<n_classes;i++) {
                ret[i] = ret[i] - delta[i];
            }

        }

        return ret;
    }

    
    static from_response(response, magnitude_fn = sq_l2_magnitude) {
        const {induced_deltas, subtree_deltas} = response;

        const induced_delta_mags = EdgeDeltaMagnitudes.from_edge_deltas_list(induced_deltas, magnitude_fn);

        // // These should all be induced edges.
        // const subtree_root_edges = subtree_deltas.map(x => x.key);

        // Map from stringfied root edge to EdgeDeltaMagnitudes.
        const root_to_subtree_delta_mags = {};
        for (const {key, value} of subtree_deltas) {
            const str_key = JSON.stringify(key);
            root_to_subtree_delta_mags[str_key] = EdgeDeltaMagnitudes.from_edge_deltas_list(value, magnitude_fn);
        }

        return new SubtreeDeltas(induced_delta_mags, root_to_subtree_delta_mags);
    }
}