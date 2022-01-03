# Project features

### Objectives

This project aims to observe which features are most helpful in predicting different classes of tumors: BRCA, KIRC, COAD, LUAD and PRAD and to see general trends that may aid us in model selection and hyperparameter selection. The goal is to classify the tumors. To achieve this, machine learning classification methods to fit a function that can predict the discrete class of new input were employed.

### Concepts covered

* Importing data.
* Understanding the data and data exploration.
* Dimensionality reduction using PCA (Principla Component Analysis).
* Training the model and model selection based on accuracy.
* Comprehensive classifiers comparison.

### Datatsets

The dataset we will be working with is part of the RNA-Seq (HiSeq) PANCAN data set, it is a random extraction of gene expressions of patients having different types of tumor: BRCA, KIRC, COAD, LUAD and PRAD.

Link to dataset's official page: <https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq>

# Repository contents

This repository contains:

* **Data files (x2):** data.csv stores the features and relative expression data for each sample and labels.csv stores the cancer class relative to each sample.

* **main.py:** python source code.

* **main.ipynb:** Jupyter notebook with source code.

* **cancerml.yml:** conda environment with all standard Python modules installed, as well as SK-learn, Keras, Numpy and Pandas is available.

* **README.md:** entry point of the project.

* **report.pdf:** brief report with an analysis of preliminary results.

# Prerequisites

To run simulations, you will need:

* Python and a local programming environment set up on your computer. A conda environment with all standard Python modules installed, as well as SK-learn, Keras, Numpy and Pandas is available. To setup it up, download the cancerml.yml file and run the following:
~~~
conda env create -f cancerml.yml
~~~

Then run the following to execute the **main.py** file:
~~~
python3 main.py
~~~

* Jupyter Notebook to open the **main.ipynb** file. Jupyter Notebooks are extremely useful when running machine learning experiments. You can run short blocks of code and see the results quickly, making it easy to test and debug your code.

# References

* Dua, D. and Graff, C. (2019). **UCI Machine Learning Repository** <http://archive.ics.uci.edu/ml>. Irvine, CA: University of California, School of Information and Computer Science.

* Cancer Genome Atlas Research Network et al. **The Cancer Genome Atlas Pan-Cancer analysis project**. Nat Genet 45, 1113–1120 (2013).

* Pedregosa, F. et al. **Scikit-learn: Machine Learning in Python**.
