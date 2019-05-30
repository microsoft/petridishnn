# Project Petridish: Efficient Forward Architecture Search
Code for Efficient Forward Neural Architecture Search. 

# Installation on development machine
We have developed and tested Petridish on Ubuntu 16.04 LTS (64-bit), Anaconda python distribution and Tensorflow

## Installing the software
1. Install [Anaconda python distribution for Ubuntu](https://www.anaconda.com/distribution/)
2. Create a python 3.6 environment `conda create python=3.6 -n py36`
3. Follow instructions to install a recent [Tensorflow (TF) version](https://www.tensorflow.org/install). 1.12 is tested.
4. Clone the repo: `git clone petridishnn`
5. Install dependency packages `python -m pip install -r <path_to_petridishnn>/requirements.txt`
6. Petridish needs some environment variables:
`GLOBAL_LOG_DIR`: directory where logs will be written to by jobs running locally.
`GLOBAL_MODEL_DIR`: directory where models will be written to by jobs running locally.
`GLOBAL_DATA_DIR`: directory from where local jobs will read data.
Set them to appropriate values in your bashrc. E.g. `export GLOBAL_MODEL_DIR="/home/dedey/data"`

## Getting the data
Petridish code assumes datasets are in certain format (e.g. we transform ImageNet raw data to lmdb format).
While one can always download the raw data of standard datasets and use the relevant scripts in `petridishnn/petridish/data` to convert
them Debadeepta Dey <dedey@microsoft.com> maintains an Azure blob with all the data in the converted format. (For Microsoft employees only)
Please email him for access.

## Running a sample search job on cifar
Before doing full scale search on Azure it is common to check everything is running on local machine.
An example job script is at `petridishnn/scripts/test_distributed.sh`. Make sure you have all the
environment variables used in this script. Run this from root folder of `petridishn` as `bash scripts/test_distributed.sh`.
This will output somethings to stdout but will output models and logs to the corresponding folders.
If this succeeds you have a working installation. Yay!

## Post-search Analysis

We provide a number of scripts to analyze and post-process the search results in the directory
[petridish/analysis](./petridish/analysis).
We also provide [a script to generate training scripts](./petridish/cust_exps_gen/generate_train_script.py) to train the found models.
We list them in the order of usage as follows.
Please refer to the header of each linked file for usage.

1. [Inspect the search log](./petridish/analysis/search.py)
2. [Generate scripts to train found models](./petridish/cust_exps_gen/generate_train_script.py)
3. [Check Performance of model training](./petridish/analysis/model.py)

Contacts:

Debadeepta Dey (dedey@microsoft.com)

Hanzhang Hu (hanzhang@cs.cmu.edu)

John Langford (jcl@microsoft.com)

Rich Caruana (rcaruana@microsoft.com)

Eric Horvitz (horvitz@microsoft.com)

# Conduct and Privacy
Petridishnn has adopted the Microsoft [Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct). For more information on this code of conduct, see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact opencode@microsoft.com with any additional questions or comments. Read Microsoft’s statement on [Privacy & Cookies](https://privacy.microsoft.com/en-us/privacystatement/)
