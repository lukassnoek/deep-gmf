# Stimulus generation

This directory contains code to generate large batches of face stimuli using the GFG
toolbox and its identity model (GMF; generative model of faces). I created a simple
CLI (`generate_faces.py`) to do so (relatively) easily, including generation of
"binocular stimuli", i.e., two images of the same face with two slightly offset
horizontal cameras simulating two eyes. The file (`generate_faces.py`) is extensively
documented so hopefully its self-explanatory.

Check out `generate_gmf112x112_binocular.sh` for an example call to the CLI to generate
a dataset with images of 1024 different identities with 256 variations per identity
(resulting in a dataset with 1024 * 256 images).

