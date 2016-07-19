# ActiveSpBdryLearn

Underlying algorithm to select a small subset of superpixel boundaries for training a superpixel edge classifier. The methods for this implementation is published in the following papers:

1. Parag, T. et.al. (2014). Small Sample Learning of Superpixel Classifiers for EM Segmentation. In Proc. of MICCAI 2014.(https://www.researchgate.net/publication/265683774_Small_Sample_Learning_of_Superpixel_Classifiers_for_EM_Segmentation)
2. Parag, T. et.al. (2015).  Efficient Classifier Training to Minimize False Merges in Electron Microscopy Segmentation. In Proc. of ICCV 2015. (https://www.researchgate.net/publication/287215311_Efficient_Classifier_Training_to_Minimize_False_Merges_in_Electron_Microscopy_Segmentation) 

Although the method was developed with an interactive interface in mind, i.e., where the user will be presented with an edge and asked for its label, the codes provided in this repo needs the actual labels to be provided with the superpixel features. It should not be difficult to attach this code to a GUI designed for interactive use, as it has been done in Janelia FlyEM NeuroProof package (https://github.com/janelia-flyem/NeuroProof).

Once a superpixel boundary has been learned using these codes, it can be used for agglomeration using the codes at: 
 https://github.com/paragt/Neuroproof_minimal


# Build
Linux: Install miniconda on your workstation. Create and activate the conda environment using the following commands:

  conda create -n my_conda_env -c flyem vigra opencv 

  source activate my_conda_env

Then follow the usual procedure of building:

  mkdir build 
  cd build

  cmake -DCMAKE_PREFIX_PATH=[CONDA_ENV_PATH]/my_conda_env ..

# Example

Given a text file with feature values and labels for all superpixel boundaries, the following command will train a boundary classifier with 10000 training examples. The method utilizes a similarity matrix that discards all connections with weights larger than 5.0.

build/InteractiveLearnMain all_feature_labels_FIB2_123.txt int_classifier_FIB_123_e10000.xml 6 10000 5.0 > log_train_FIB2_123_e10000.txt

The sample feature file is uploaded to dropbox for size constraint: https://www.dropbox.com/sh/6xq3t0wfhdxbb1l/AACtoyLQyDbmjra-NvHDAiBga?dl=0