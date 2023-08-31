// Contains the graph.


class GraphViewer {

    ///////////////////////////////////////////////////////
    // Some constants.

    WIDTH_PER_TOKEN = 50;
    HEIGHT_PER_TRANSFORMER_LAYER = 150;
    ATTENTION_LAYER_HEIGHT = 95;
    SEQUENCE_REP_HEIGHT = 20;

    ///////////////////////////////////////////////////////

    constructor(metadata, options) {
        this.FFW_LAYER_HEIGHT = this.HEIGHT_PER_TRANSFORMER_LAYER - this.ATTENTION_LAYER_HEIGHT;
        this.TOKENS_FOOTER_HEIGHT = this.SEQUENCE_REP_HEIGHT;

        this.options = this._fill_out_options(options || {});

        this.n_transformer_layers = metadata.n_transformer_layers
        this.n_attention_heads = metadata.n_attention_heads
        this.tokens = metadata.tokens;

        // Only the non-padding tokens get sent over.
        this.sequence_length = metadata.tokens.length;

        this._create_initial_transformer_layers_data()

        // Edge thicknesses will be divided by this if 'edge_thickness_style' is not 'constant'.
        // Defaults to 1.0, which is no normalization.
        this._edge_thickness_normalizing_constant = 1.0
    }

    _fill_out_options(overrides) {
        const EDGE_THICKNESS_STYLES = {'constant': true, 'sqrt': true, 'log': true};
        const ret = {
            'edge_thickness_style': 'constant',
            // Set to integer to be thickness normalized to. Otherwise, leave to a false-y value
            // to not normalize.
            'normalize_edge_thickness': false,
            ...overrides,
        };
        if(!EDGE_THICKNESS_STYLES[ret.edge_thickness_style]) { throw new Exception(ret.edge_thickness_style); }

        return ret;
    }

    _create_initial_transformer_layers_data() {
        this.transformer_layers_data = [];
        for(let i=0; i<this.n_transformer_layers; i++) {
            this.transformer_layers_data.push({
                'attention': {'induced': [], 'transformed': [], 'residual': []},
                'ffw': {'induced': [], 'transformed': [], 'residual': []},
            });
        }
    }
    ///////////////////////////////////////////////////////

    _get_layers_height() {
        return this.n_transformer_layers * this.HEIGHT_PER_TRANSFORMER_LAYER;
    }

    _get_total_height() {
        return this._get_layers_height() + this.TOKENS_FOOTER_HEIGHT;
    }

    ///////////////////////////////////////////////////////


    _render_tokens_footer(tokens_footer_g) {
        this.token_containers = tokens_footer_g.selectAll()
            .data(this.tokens)
            .enter()
            .append('g')
            .attr('class', 'token-container')
            .attr("transform", (d, i) => `translate(${i * this.WIDTH_PER_TOKEN}, 0)`);

        this.token_containers.append('rect')
            .attr('class', 'token-container-rect')
            .attr('width', this.WIDTH_PER_TOKEN)
            .attr('height', this.TOKENS_FOOTER_HEIGHT)
            .attr('rx', 5);

        this.token_containers.append('text')
            .attr('class', 'token-container-text')
            .attr('x', this.WIDTH_PER_TOKEN / 2)
            // .attr('y', this.SEQUENCE_REP_HEIGHT / 2)
            .attr('y', this.SEQUENCE_REP_HEIGHT * 0.7)
            .attr('text-anchor', "middle")
            .style('line-height', this.TOKENS_FOOTER_HEIGHT)
            .text(d => d);
    }

    _render_layers_token_sequence(selection) {
        const token_containers = selection.selectAll()
            .data(this.tokens)
            .enter()
            .append('g')
            .attr('class', 'layers-token-container')
            // The on click behavior is temporary for when I want to figure out positions of tokens.
            // Comment or uncomment it as needed.
            .on('click', (d, i) => console.log(i, d))
            .attr("transform", (d, i) => `translate(${i * this.WIDTH_PER_TOKEN}, 0)`);

        token_containers.append('rect')
            .attr('class', 'layers-token-container-rect')
            .attr('width', this.WIDTH_PER_TOKEN)
            .attr('height', this.SEQUENCE_REP_HEIGHT)
            .attr('rx', 5);

        token_containers.append('text')
            .attr('class', 'layers-token-container-text')
            .attr('x', this.WIDTH_PER_TOKEN / 2)
            // .attr('y', this.SEQUENCE_REP_HEIGHT / 2)
            .attr('y', this.SEQUENCE_REP_HEIGHT * 0.7)
            .attr('text-anchor', "middle")
            .style('line-height', this.SEQUENCE_REP_HEIGHT)
            .text(d => d);

        return token_containers
    }

    _initial_render_transformer_layers(layer_gs) {
        ///////////////////////////////
        // Attention layer.

        // Add container for everything related to the self-attention sublayer.
        this.attention_layer_gs = layer_gs.append('g')
            .attr('class', 'attention-layer-container')
            .attr("transform", `translate(0, ${this.HEIGHT_PER_TRANSFORMER_LAYER - this.ATTENTION_LAYER_HEIGHT})`)
            .datum(d => d['attention']);

        // Render the token sequence representation at the top of the residual block's space.
        const attention_token_sequence = this.attention_layer_gs.append('g')
            .attr('class', 'layers-token-sequence');
        this._render_layers_token_sequence(attention_token_sequence);

        ///////////////////////////////
        // FFW layer.

        this.ffw_layer_gs = layer_gs.append('g')
            .attr('class', 'ffw-layer-container')
            .datum(d => d['ffw']);

        // Render the token sequence representation at the top of the residual block's space.
        const ffw_token_sequence = this.ffw_layer_gs.append('g')
            .attr('class', 'layers-token-sequence');
        this._render_layers_token_sequence(ffw_token_sequence);
    }

    // Render the things that should not change when the edges
    // in the graph change.
    initial_render() {
        this.main_container = d3.select('#main-container');
        this.svg = this.main_container.append('svg')
            .attr('class', 'induced-deltas-viewer')
            .attr('width', this.sequence_length * this.WIDTH_PER_TOKEN)
            .attr('height', this._get_total_height());

        this.main_g = this.svg.append('g');

        this.layer_gs = this.main_g.selectAll()
            .data(this.transformer_layers_data)
            .enter()
            .append('g')
            .attr("transform",
                  (d, i) => 'translate(0,' + (this.HEIGHT_PER_TRANSFORMER_LAYER * (this.n_transformer_layers - i - 1)) + ')');
        this._initial_render_transformer_layers(this.layer_gs);

        this.tokens_footer_g = this.main_g.append('g')
            .attr("transform", `translate(0, ${this._get_layers_height()})`);
        this._render_tokens_footer(this.tokens_footer_g);
    }

    ///////////////////////////////////////////////////////

    _make_attention_d(d, x_offset) {
        const query_index = d.dst;
        const key_index = d.src;

        const qx = (query_index + x_offset) * this.WIDTH_PER_TOKEN;
        const qy = this.SEQUENCE_REP_HEIGHT;

        const kx = (key_index + x_offset) * this.WIDTH_PER_TOKEN;
        const ky = this.ATTENTION_LAYER_HEIGHT;

        // Simple straight line.
        // return `M ${kx} ${ky} L ${qx} ${qy}`;
        // C x1 y1, x2 y2, x y

        // Cubic Bezier.
        // return `M ${kx} ${ky} C ${kx} ${qy}, ${qx} ${ky}, ${qx} ${qy}`;
        // return `M ${kx} ${ky} C ${(kx + qx)/2} ${qy}, ${(kx + qx)/2} ${ky}, ${qx} ${qy}`;
        return `M ${kx} ${ky} C ${(kx + qx)/2} ${(ky + 3 * qy) / 4}, ${(kx + qx)/2} ${(3 * ky + qy) / 4}, ${qx} ${qy}`;
    }

    _make_attention_residual_d(d, x_offset) {
        // d.src and d.dst should be the same, so pick one arbitrarily.
        const sequence_position = d.src;
        const x = (sequence_position + x_offset) * this.WIDTH_PER_TOKEN;
        // Simple straight line.
        return `M ${x} ${this.ATTENTION_LAYER_HEIGHT} L ${x} ${this.SEQUENCE_REP_HEIGHT}`;
    }


    _mask_ffw_d(d, x_offset) {
        // d.src and d.dst should be the same, so pick one arbitrarily.
        const sequence_position = d.src;
        const x = (sequence_position + x_offset) * this.WIDTH_PER_TOKEN;
        // Simple straight line.
        return `M ${x} ${this.FFW_LAYER_HEIGHT} L ${x} ${this.SEQUENCE_REP_HEIGHT}`;
    }

    _magnitude_to_raw_edge_thickness(magnitude) {
        const style = this.options.edge_thickness_style;
        if(style === 'constant') {
            // The css file should set the width.
            return undefined;
        } else if(style === 'sqrt') {
            return 150 * Math.sqrt(magnitude);
        } else if(style === 'log') {
            throw new Exception('TODO');
        } else {
            throw new Exception(style);
        }
    }

    _get_edge_stroke_width(d) {
        const style = this.options.edge_thickness_style;
        if(style === 'constant') {
            // The css file should set the width.
            return undefined;
        }

        const {magnitude} = d;

        let width = this._magnitude_to_raw_edge_thickness(magnitude)
        width /= this._edge_thickness_normalizing_constant

        return `${width}px`;
    }


    render_edges() {
        /////////////////////////////////////////////////////////////
        // Attention layer.

        // Remove existing connecting arrows.
        this.attention_layer_gs.selectAll('.attention-layer-path').remove();

        // Add some connecting arrows.
        this.attention_layer_gs.selectAll()
            .data(d => d['residual'])
            .enter()
            .append('path')
            .attr('class', 'attention-layer-path residual-mask-path')
            .style('stroke-width', d => this._get_edge_stroke_width(d))
            .on('click', this.on_edge_click)
            .attr('d', d => this._make_attention_residual_d(d, 0.25));
        this.attention_layer_gs.selectAll()
            .data(d => d['transformed'])
            .enter()
            .append('path')
            .attr('class', 'attention-layer-path transformed-mask-path')
            .style('stroke-width', d => this._get_edge_stroke_width(d))
            .on('click', this.on_edge_click)
            .attr('d', d => this._make_attention_d(d, 0.5));
        this.attention_layer_gs.selectAll()
            .data(d => d['induced'])
            .enter()
            .append('path')
            .attr('class', 'attention-layer-path induced-mask-path')
            .style('stroke-width', d => this._get_edge_stroke_width(d))
            .on('click', this.on_edge_click)
            .attr('d', d => this._make_attention_d(d, 0.75));

        /////////////////////////////////////////////////////////////
        // FFW layer.

        // Remove existing connecting arrows.
        this.ffw_layer_gs.selectAll('.ffw-layer-path').remove();

        // Add some connecting arrows.

        this.ffw_layer_gs.selectAll()
            .data(d => d['residual'])
            .enter()
            .append('path')
            .attr('class', 'ffw-layer-path residual-mask-path')
            .on('click', this.on_edge_click)
            .style('stroke-width', d => this._get_edge_stroke_width(d))
            .attr('d', d => this._mask_ffw_d(d, 0.25));
        this.ffw_layer_gs.selectAll()
            .data(d => d['transformed'])
            .enter()
            .append('path')
            .attr('class', 'ffw-layer-path transformed-mask-path')
            .style('stroke-width', d => this._get_edge_stroke_width(d))
            .on('click', this.on_edge_click)
            .attr('d', d => this._mask_ffw_d(d, 0.5));
        this.ffw_layer_gs.selectAll()
            .data(d => d['induced'])
            .enter()
            .append('path')
            .attr('class', 'ffw-layer-path induced-mask-path')
            .style('stroke-width', d => this._get_edge_stroke_width(d))
            .on('click', this.on_edge_click)
            .attr('d', d => this._mask_ffw_d(d, 0.75));
    }

    ///////////////////////////////////////////////////////
    // Possible listeners, override on object to set.
    // These all default to no-ops.
    // TODO: Probably better way of doing this.

    on_edge_click(d, i) {}

    ///////////////////////////////////////////////////////

    // edges_data is instance of EdgeDeltaMagnitudes.
    set_edges_data(edges_data) {
        this._compute_edge_thickness_normalizing_constant(edges_data);

        for(let i=0; i<this.n_transformer_layers; i++) {
            for(const edge_type of ['induced', 'transformed', 'residual']) {
                for(const res_block_type of ['attention', 'ffw']) {

                    const data = edges_data.get_all_connections_of_type_for_transformer_layer(
                        i, res_block_type, edge_type);

                    // Sort in ascending order of magnitude to help with clicking on them.
                    data.sort((a, b) => a.magnitude - b.magnitude);

                    this.transformer_layers_data[i][res_block_type][edge_type] = data;
                }
            }
        }
    }

    // edges_data is instance of EdgeDeltaMagnitudes.
    _compute_edge_thickness_normalizing_constant(edges_data) {
        const style = this.options.edge_thickness_style;
        const max_edge_thickness = this.options.normalize_edge_thickness;

        if (!max_edge_thickness || style == 'constant') {
            this._edge_thickness_normalizing_constant = 1.0;
            return;
        }

        let max_raw_width = -1;

        for(const {magnitude} of edges_data.connection_magnitudes){
            const raw_width = this._magnitude_to_raw_edge_thickness(magnitude);
            max_raw_width = raw_width > max_raw_width ? raw_width : max_raw_width;
        }

        this._edge_thickness_normalizing_constant = max_raw_width / max_edge_thickness;
    }

}
