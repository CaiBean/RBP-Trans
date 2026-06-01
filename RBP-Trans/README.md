# PaRPI
## Contacts
Any more questions, please do not hesitate to contact me: [20234227085@stu.suda.edu.cn](mailto:20234227085@stu.suda.edu.cn)

## NOTICE
Due to the capacity limiation of Github, we put the relevant files (including the BERT model and all datasets) in [Zenodo](https://doi.org/10.5281/zenodo.14878562). All source code, data and model are open source and can be downloaded from GitHub.

## Requirements
PaRPI mainly depends on the Python scientific stack.
- python=3.8
- ViennaRNA
- numpy
- pandas
- scikit-learn
- torch
- dgl
- fair-esm
- transformers


You can automatically install all the dependencies with Anaconda using the following command:
```sh
conda env create -f environment.yml
```

## Quick start
Before running the code, please create two empty directories required for data processing:

```sh
mkdir -p ./dataset/bat/
mkdir -p ./dataset/test/
```

The RBP-Trans model and all to be trained datasets should first be download and put into the corresponding folder. Then, you can train a model with a certain dataset using the following command:

```sh
python train_base.py --train --eclip_path data/encode_eclip.h5  --config .configs/base_config.yaml
python train_bc.py --train --eclip_path data/encode_eclip_bc.h5  --config .configs/bc_config.yaml
```

After training, you can validate the model by using :

```sh
python train_base.py --validate --eclip_path data/encode_eclip.h5 --config .configs/base_config.yaml
python train_bc.py --validate --eclip_path data/encode_eclip_bc.h5  --config .configs/bc_config.yaml
```

To utilize a trained model for predicting new RBP, you should first acquire the corresponding sample datas and place them in the designated folder. Subsequently, execute the following code:

```sh
python main.py --prediction_aware --cell H9 --data_name LIN28A_H9
```
