"""Server for the NPEFF trace viewer."""
from npeff_tracer.util import vat_da_faak_huggingface

import collections
import os

from absl import app as absl_app
from absl import flags

from flask import Flask, jsonify, render_template

from npeff_tracer.traces import bert_tracer

FLAGS = flags.FLAGS

flags.DEFINE_string("trace_filepath", None, "Filepath of h5 file containing the trace.")

###############################################################################

app = Flask(__name__)

###############################################################################


def _dict_of_arrays_to_kv_list(d):
    return [
        {'key': k, 'value': v.tolist()}
        for k, v in d.items()
    ]


###############################################################################


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/data')
def data():
    trace = bert_tracer.BertTrace.load(FLAGS.trace_filepath)

    metadata = {
        'tokens': [t for t in trace.tokens if t != '[PAD]'],
        'n_transformer_layers': trace.n_transformer_layers,
        'n_attention_heads': trace.n_attention_heads,
    }

    induced_deltas = _dict_of_arrays_to_kv_list(trace.induced_edge_contributions)
    
    subtree_deltas = [
        {'key': k, 'value': _dict_of_arrays_to_kv_list(v)}
        for k, v in trace.subtree_contributions.items()
    ]

    return jsonify({
        'metadata': metadata,
        'induced_deltas': induced_deltas,
        'subtree_deltas': subtree_deltas,
    })


###############################################################################

def main(_):
    app.run(debug=True)

###############################################################################


if __name__ == '__main__':
    absl_app.run(main)
