# RBP-Trans
![]([https://rbp-trans.oss-cn-hangzhou.aliyuncs.com/%E6%8A%80%E6%9C%AF%E8%B7%AF%E7%BA%BF%E5%9B%BE.bmp?Expires=1780286061&OSSAccessKeyId=TMP.3KvN3xVzVr1ZigtZXZxepPSJvSkJf1QbzDGDGMACv2xahBRREMoim2K5pkHv6guh4rmGiWf7T7meuPNimo5ejxBTX9vKew&Signature=ZFR6wJONUpcUSXzeggD58Kotppw%3D](https://rbp-trans.oss-cn-hangzhou.aliyuncs.com/36ae410e-bfed-4b39-951f-eed71d095093.png?Expires=1780286399&OSSAccessKeyId=TMP.3KvN3xVzVr1ZigtZXZxepPSJvSkJf1QbzDGDGMACv2xahBRREMoim2K5pkHv6guh4rmGiWf7T7meuPNimo5ejxBTX9vKew&Signature=14hAV0GvmQFXW0L1Yu%2F8vEbYreA%3D))


## Requirements
RBP-Trans mainly depends on the Python scientific stack.
- python=3.9
- RNA-FM
- numpy
- pandas
- scikit-learn
- torch
- esm
- fair-esm
- transformers


You can automatically install all the dependencies with Anaconda using the following command:
```sh
conda env create -f environment.yml
```

## Quick start
Before running the code, please create two empty directories required for data and checkpoint:

```sh
mkdir  ./data/
mkdir  ./checkpoint/
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

To utilize a trained model for denoisy peak calling pipeline, you should first acquire the corresponding sample datas and place them in the designated folder. Subsequently, execute code like this:

```sh
python train_rabk.py --train --parquet_path data/rank_HepG2_RBFOX2_rep1.parquet  --config .configs/rank_config.yaml
```
## NOTICE
Due to the capacity limiation of Github, we put the relevant files (including the BERT model and all datasets) in [Zenodo](https://doi.org/10.5281/zenodo.14878562). All source code, data and model are open source and can be downloaded from GitHub.

## Contacts
Any more questions, please do not hesitate to contact me: [522023300040@smail.nju.edu.cn](mailto:522023300040@smail.nju.edu.cn)
