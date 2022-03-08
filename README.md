# deep-gmf
How do deep neural networks (DNN) learn to recognize faces? Here, we use generative model of faces (GMF) to sythesize images of faces based on variations in 3D shape, texture, gender, age, ethnicity, and other (3D) transformations like random rotations, translations, light source/orientation, and background. We then use DNNs to predict particular features (like face ID) and investigate to what extend the intermediate representations of the DNN (i.e., it's layer activations) represent the inverse of the generative model.
