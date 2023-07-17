# Data

- `emo_config.tsv`: AU parameters for canonical AUs for six basic emotions (taken from Christoph's dataset, used for the Patterns paper, which were in turn estimated from Yu, Garrod, & Schyns, 2012)
- `lights.yaml`: simple directional light config file, to be used within the GFG renderer
- `idm_St.npy`: standard deviation (S) of original PCA-transformed texture (T) variables (note: there are 5 frequency bands per component)
- `idm_Sv.npy`: standard deviation (S) of original PCA-transformed vertex (V; "shape") variables
- `backgrounds`: folder with 256 different backgrounds (such that each of the 256 instances per face identity can have a unique background, making sure background can be treated as a categorical variable, with 256 classes, which is counterbalanced across identities)
- `example_images_for_presentations`: images that can be used to explain/visualize the generative stimulus model

