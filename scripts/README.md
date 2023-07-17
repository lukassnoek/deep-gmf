# Pipeline

To train, preprocess, and analyze a DNN using GMF stimuli, do the following:

- Generate stimuli (see `./generate_stimuli/generate_faces.py` for a CLI tool)
- Aggregate stimulus parameters into a single CSV (see `./aggregate_id_params.py`, which will also create a train/val/test split)
    - See `./generate_stimuli/generate_gmf112x112_binocular.sh` for an example
- Train the DNN model (using the CLI from `./train_model.py`)
- Learn a PCA model per DNN layer (using the CLI from `./compress_model.py`)
- Extract PCA-compressed DNN activations and associated stimulus parameters (using the CLI from `./decompose_model.py`)
- Estimate decodability of each generative stimulus feature (using the CLI from `./analyze_model.py`)
