const DATA_URL = '/data';

d3.json(DATA_URL).then(run);


let viewer;
let induced_deltas_data;
let subtree_deltas;

let total_perturbation_delta;

let active_induced_deltas_data;


function on_edge_click(d, i) {
    if(!subtree_deltas.root_to_subtree_delta_mags[d.edge_str_key]) { return; }

    induced_deltas_data = subtree_deltas.get_delta_mags_for_root_edge(JSON.parse(d.edge_str_key));

    active_induced_deltas_data = induced_deltas_data.clone();
    viewer.set_edges_data(active_induced_deltas_data);
    viewer.render_edges();
}


function run(response) {
    subtree_deltas = SubtreeDeltas.from_response(response);

    total_perturbation_delta = subtree_deltas.get_total_perturbation_delta();

    induced_deltas_data = subtree_deltas.induced_delta_mags;

    const viewer_options = {
        'edge_thickness_style': 'sqrt',
        'normalize_edge_thickness': 10,
    };

    viewer = new GraphViewer(response.metadata, viewer_options);

    // Add listeners.
    viewer.on_edge_click = on_edge_click;

    viewer.initial_render();
    viewer.render_edges();

    active_induced_deltas_data = induced_deltas_data.clone();
    viewer.set_edges_data(active_induced_deltas_data);
    viewer.render_edges();
    
    viewer.svg.on('dblclick', render_all_induced_onlys);
}


function render_all_induced_onlys() {
    induced_deltas_data = subtree_deltas.induced_delta_mags;

    active_induced_deltas_data = induced_deltas_data.clone();
    viewer.set_edges_data(active_induced_deltas_data);
    viewer.render_edges();
}
