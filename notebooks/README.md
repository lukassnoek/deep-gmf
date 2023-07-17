# Notebooks

Bunch of Jupyter Notebooks to create figures for the Wellcome grant.

- `analysis_mediation.ipynb`: some proof-of-concept analyses to show that you can project layer activations (X) into generative feature space (Z, e.g., shape space) from which you can decode identity (Y, or some other generative variable), i.e., X --> Z --> Y (a "mediation analysis", if you will); hasn't been used for the grant
- `animation_scratchpad.ipynb`: includes some code to mess around with GFG animation features
- `extrapolation.ipynb`: analyses/visualizations of intra/extrapolation capabilities of ResNets trained on (selections of) GMF stimuli
- `parametric.ipynb`: code for figures showing parametric psychophysics curves (for lack of a better term) of DNN layers; note that these analyses are non-typical, as they model and visualize the relationship of a parametric stimulus feature (e.g., rotation), X, with the *decodability of another feature from a DNN layer* (e.g., negative log likelihood, or just P(Y_true), or R-squared for continuous targets)
- `layerwise_performance.ipynb`: simple figure to show the feature decodability performance for each feature and each layer
- `scratchpad.ipynb`: includes some code to generate simple images using the GFG (often use this to create images for presentations)
- `transfer_learning.ipynb`: code to create the "tranfer learning" figure for the Wellcome grant

Note that most analysis/visualization notebooks presume that the appropriate data (i.e., results from the `analyze_model.py` script) are available in the `results/` directory. Also, the code in the notebooks is super messy; sorry.


