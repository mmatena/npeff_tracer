R"""Computes the traces for a BERT LRM-NPEFF on some examples.

The examples are provided as a CSV file. The file must have no header rows (i.e. be
all data). Each row corresponds to an example. The file must have either one or two columns.
Each column corresponds to a "segment" of the example (e.g. premise and hypothesis for an NLI
task). The second column can be left empty to signify a single segment if using two columns.

A separate output file will be created for each example. If there is more than one example, each
file will have a "-example{index}" suffix appended to it.
"""
from npeff_tracer.util import vat_da_faak_huggingface

import csv
import os

from absl import app
from absl import flags

import tensorflow as tf

from npeff.decomp import decomps
from npeff.models import npeff_models

from npeff_tracer.states import bert_state_computation
from npeff_tracer.traces import bert_masked_evaluation
from npeff_tracer.traces import bert_tracer


FLAGS = flags.FLAGS

flags.DEFINE_string("output_filepath", None, "Filepath of h5 file to write traces to.")

flags.DEFINE_string("examples_filepath", None, "Filepath to CSV file of examples. See the file-level commment "
                                               "of this script for info on its format.")
flags.DEFINE_string("decomposition_filepath", None, "Filepath of LRM-NPEFF decomposition.")
flags.DEFINE_integer("component_index", None, "NPEFF component to perform trace for.")

flags.DEFINE_integer("n_subtrees_to_trace", 8, "Number edges with induced perturbations to trace. The "
                                               "edges with the largest contributions to the perturbations "
                                               "of the logits will be traced first.")

flags.DEFINE_string("model", None, "String indicating model to use.")
flags.DEFINE_bool("from_pt", True, "Whether the model is from PyTorch.")

flags.DEFINE_string("tokenizer", None, "Tokenizer to use. Defaults to --model if not set for a text task.")
flags.DEFINE_integer("max_sequence_length", 128, "Maximum sequence length to use.")

###############################################################################


def _row_to_example(tokenizer, row):
    row = (r.strip() for r in row)
    row = [r for r in row if r]

    text_a = row[0]
    text_b = None

    if len(row) == 2:
        text_b = row[1]
    elif len(row) != 1:
        raise ValueError

    example = tokenizer.encode_plus(
        text_a,
        text_b,
        add_special_tokens=True,
        max_length=FLAGS.max_sequence_length,
        return_token_type_ids=True,
        truncation=True,
    )

    # The tf.expand_dims adds a dummy batch dim.
    return {k: tf.expand_dims(tf.cast(v, tf.int32), axis=0) for k, v in example.items()}


def read_examples(tokenizer):
    examples_filepath = os.path.expanduser(FLAGS.examples_filepath)
    with open(examples_filepath, 'r') as f:
        rows = list(csv.reader(f))
    return [_row_to_example(tokenizer, r) for r in rows]


def process_example(nmf, model, variables, tokenizer, example, example_index, n_examples):
    bsc = bert_state_computation.BertStateComputer(
        nmf=nmf,
        model=model,
        variables=variables,
        component_index=FLAGS.component_index,
    )

    bsc.call(example, training=False)
    bert_state = bsc.create_bert_state()

    masked_evaluator = bert_masked_evaluation.MaskedEvaluator(
        bert_state=bert_state,
    )
    tracer = bert_tracer.BertTracer(
        masked_evaluator=masked_evaluator,
        tokenizer=tokenizer,
        use_tqdm=True,
    )

    trace = tracer.compute_trace(FLAGS.n_subtrees_to_trace)

    trace_filepath = os.path.expanduser(FLAGS.output_filepath)
    if n_examples > 1:
        trace_filepath = f'{trace_filepath}-example{example_index}'

    trace.save(trace_filepath)


def main(_):
    model_str = os.path.expanduser(FLAGS.model)

    model = npeff_models.from_pretrained(model_str, from_pt=FLAGS.from_pt)
    variables = model.trainable_variables

    tokenizer = npeff_models.load_tokenizer(FLAGS.tokenizer or model_str)
    examples = read_examples(tokenizer)

    nmf = decomps.LazyLoadedLrmNpeffDecomposition.load(FLAGS.decomposition_filepath)

    for example_index, example in enumerate(examples):
        process_example(nmf, model, variables, tokenizer, example, example_index, len(examples))


if __name__ == "__main__":
    app.run(main)
