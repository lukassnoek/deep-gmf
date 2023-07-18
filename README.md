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




### Step 3: compress model

### Step 4: decompose model

### Step 5: analyze model