# Implementation of the Self-Supervised Image Enhancement System
### Neural Image Correction and Enhancement Routine

This repository contains a PyTorch implementation of: 

**"NICER: Aesthetic Image Enhancement with Humans in the Loop" [ACHI2020]**

by [M. Fischer](https://github.com/mr-Mojo), [K. Kobs](http://www.dmir.uni-wuerzburg.de/staff/kobs/) and [A. Hotho](http://www.dmir.uni-wuerzburg.de/staff/hotho/). 
The publication can be found at the [ThinkMind(TM) Digital Library](https://www.thinkmind.org/index.php?view=article&articleid=achi_2020_5_390_20186). 

with an Image Assessor based on **"Self-Supervised Multi-Task Pretraining Improves Image Aesthetic Assessment" [CVPR2021]**

by [J. Pfister](https://github.com/janpf), [K. Kobs](http://www.dmir.uni-wuerzburg.de/staff/kobs/) and [A. Hotho](http://www.dmir.uni-wuerzburg.de/staff/hotho/). 
The publication can be found at [CVPR 2021 open access](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/html/Pfister_Self-Supervised_Multi-Task_Pretraining_Improves_Image_Aesthetic_Assessment_CVPRW_2021_paper.html). 



## Installation

To install and run this framework, it is recommended that you create a `conda` environment. For further information on managing conda environments, confer 
[the docs](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). 
Afterwards, head over to [PyTorch](https://pytorch.org/get-started/locally) and install the appropriate PyTorch and Cuda versions. 

Once PyTorch is installed, go ahead and clone this repository. Then install the required libraries:

`pip install -r requirements.txt`

This should work™...at least for me it did. Dx

## Usage

This is where the fun begins! (or not because the code is a mess...)

For generally trying out the Image Enhancement System, the main.py can be run. A few basic options can be changed during
runtime in the GUI, but for deeper changes of hyperparameters the config.py file has to be used.

As is the program is configured with the optimal parameters for SGD. Two config files are provided. The first is
config_Hopt_SGD.py which is configured for optimal parameters with SGD, and the second is config_Hopt_CMA-ES.py which
is configured for optimal parameters with CMA-ES.

For the analysis of the hyperparameter search, hyperparameter plotting.py in the analysis folder was used, for the survey
the survey_evaluation.ipynb Jupyter Notebook was used, also found in the analysis folder.

Results of our hyperparameter search and survey are found in the analysis/results folder, hyperparametersearch_new contains
the data related to SGD and hyperparametersearch_cma the data related to CMA-ES.

The recovered photos this data is based on can be found on the cluster, the same is for any datasets used.

Survey data can be found in analysis/sets/survey_results, both the raw data from MTurk and the preprocessed data is available.
