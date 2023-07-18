# deep-gmf
How do deep neural networks (DNN) learn to recognize faces? Here, we use generative model of faces (GMF), a 3D morphable model, in combination with a computer graphics pipeline to sythesize images of faces based on variations in 3D shape, texture, gender, age, ethnicity, and other (3D) transformations like random rotations, translations, light source/orientation, and background. We then use DNNs to predict particular features (like face ID) and investigate to what extend the intermediate representations of the DNN (i.e., it's layer activations) represent the inverse of the generative model.

## Pipeline
Below, I describe the entire analysis pipeline from start to end. Note that all scripts
are meant to be run from the root of this code repository, e.g.:

`python scripts/train_model ResNet6 gmf_112x112_binoc_emo`

### Step  1: generate stimuli
The first step is to use the GFG computer graphics toolbox to generate a stimulus set.
There is a Python command-line utility for this (in `scripts/generate_stimuli`): 
`generate_faces.py`, which uses the GFG with some sensible defaults to create a set of
stimuli of N faces with K variations (possibly with binocular versions, i.e., one image
per virtual eye).

After generating the stimuli, run the `scripts/generate_stimuli/aggregate_id_params.py`
command-line script to aggregate all image/face ID metadata into a dataframe and split
the data into a train, validation, and test set (with unique face ID per subset).

The resulting CSV file will be saved in the parent directory of the directory with
your images (with the name: `{name_of_data_directory}.csv`).

### Step 2: train model

To train a model on a GMF dataset, you can use the `train_model.py` CLI tool (in the
`scripts` directory). An example call would be:

`python scripts/train_model.py --model ResNet6 --dataset path_to_dataset --target id`

which would train a ResNet6 model to classify face identity.

### Step 3: compress model

After training a model, which will be saved in the `trained_models` directory, an 
incremental PCA model can be trained on each layer's activations (for a number of batches)
so that it can be compressed to the same number of variables (which will speed up 
feature decoding considerably). A CLI tool for this can be found in `scripts/compress_model.py`
and an example call would be:

`python scripts/compress_model.py trained_models/ResNet6_dataset-datasetname_target-id/epoch-050 --n-batches 32`

This call will save an HDF5 file in the same directory with the per-layer PCA paramters, e.g.,
`trained_models/ResNet6_dataset-datasetname_target-id/epoch-050_compressed.h5`

### Step 4: decompose model

To extract the (compressed) model activations for all stimuli in the train, validation, 
and test set, you can use the `scripts/decompose_model.py` CLI tool; and example call would be:

`python scripts/decompose_model.py trained_models/ResNet6_dataset-datasetname_target-id/epoch-050`

which will another HDF5 file to the same directory as the saved model, e.g.:

`trained_models/ResNet6_dataset-datasetname_target-id/epoch-050_decomposed.h5`

Note that this file contains three groups ('training', 'validation', 'testing') that contain
the activations for the separate splits.

### Step 5: analyze model

Finally, the model activations can be analyzed using the `scripts/analyze_model.py` CLI tool,
which can be called as follows:

`python scripts/analyze_model.py trained_models/ResNet6_dataset-datasetname_target-id/epoch-050`

In this analysis, each of the generative features will be decoded from the (compressed)
layer activations by fitting a model on the train-set (and, for regression models, the 
regularization parameter[s] will be optimized on the validation set) and cross-validated
to the test set. Both the layerwise performance will be saved as well as the layerwise
predictions, both in the `results` directory.

## ToDo
Some things that need work/need to be fixed:

* Right now, if you use a query (e.g., to train on only a subset of images), compression
is still done on all data, which might introduce leakage into the test set; one idea is to
save the queried dataframe to the `trained_models` directory which can then be reused for
compression and decomposition