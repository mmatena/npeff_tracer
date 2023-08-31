# Perturbation Tracing for BERT LRM-NPEFFs

<!-- Code to compute and visualize NPEFF perturbations in transfomers. -->

Recall that LRM-NPEFF produces a "pseudo-Fisher" vector for each component.
Perturbing the parameters of the model in the direction of this vector perturbs
the model's predictions preferentially on examples with a large coefficient for
that component.
<!--  -->
These parameter perturbations thus get translated to perturbations in the internal
activations of the model, which in turn get propagated to perturbations in the model's
predictions.

Here, we explore computation and visualization of where the relevant activation
perturbations get introduced and how they get propagated to the model's output.
While similar analysis can be conducted for other architectures, we only explore
the BERT transformer models here.

## Overview

### Trace Computation

The `scripts/compute_bert_npeff_trace.py` script can be used to save traces for an
LRM-NPEFF component on a set of examples. The examples are provided via a csv file.
More information about the script and its flags is contained within its file.

### Trace Visualization

Once a trace has been computed and saved, it can be visualized using code in `viewer`
subdirectory. Run the `viewer/app.py` executable with its `--trace_filepath` flag
set to the trace you wish to view. This will start up a server that serves a webpage
at http://localhost:5000 containing an interactive visualization of the trace.
