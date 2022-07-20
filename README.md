# Partial and Asymmetric Contrastive Learning for Out-of-Distribution Detection in Long-Tailed Recognition

This is the official implementation of the [Partial and Asymmetric Contrastive Learning for Out-of-Distribution Detection in Long-Tailed Recognition](https://proceedings.mlr.press/v162/wang22aq/wang22aq.pdf) paper at ICML 22 (Long Presentation).


<p align="center">
<img width="300"  src="https://user-images.githubusercontent.com/22279212/180052064-c9fa0d55-1e3d-44ac-adf2-60ec89536fc6.png">
</p>

## Stage 1 training: Training main branch using PASCL loss

CIFAR10-LT: 

```
python stage1.py --gpu 0 --ds cifar10 --Lambda2 0.1 --T 0.07 \
    --drp <where_you_store_all_your_datasets> --srp <where_to_save_the_ckpt>
```

CIFAR100-LT:

```
python stage1.py --gpu 0 --ds cifar100 --Lambda2 0.02 --T 0.2 \
    --drp <where_you_store_all_your_datasets> --srp <where_to_save_the_ckpt>
```

ImageNet-LT:

```
python stage1.py --gpu 0 --ds imagenet --md ResNet50 -e 100 --opt sgd --decay multisteps --lr 0.1 --wd 5e-5 --tb 100 \
    --ddp --dist_url tcp://localhost:23457 \
    --drp <where_you_store_all_your_datasets> --srp <where_to_save_the_ckpt>
```


## Stage 2 training: Finetune auxiliary classification head (ABF)

CIFAR10-LT:

```
python stage2.py --gpu 0 --ds cifar10 \
    --drp <where_you_store_all_your_datasets> \
    --pretrained_exp_str <the_name_of_your_stage1_training_experiment>
```

CIFAR100-LT:

```
python stage2.py --gpu 0 --ds cifar100 \
    --drp <where_you_store_all_your_datasets> \
    --pretrained_exp_str <the_name_of_your_stage1_training_experiment>
```

ImageNet-LT:

```
python stage2.py --gpu 0 --ds imagenet -e 3 --opt sgd --decay multisteps --lr 0.01 --wd 5e-5 --tb 100 \
    --ddp --dist_url tcp://localhost:23457 \
    --pretrained_exp_str <the_name_of_your_stage1_training_experiment>
```

`--pretrained_exp_str ` should be something like `e200-b256-adam-lr0.001-wd0.0005-cos_Lambda0.5-Lambda20.1-T0.07-sign-k0.5`

## Testing

CIFAR10-LT:

```
for dout in texture svhn cifar tin lsun places365
do
python test.py --gpu 0 --ds cifar10 --dout $dout \
    --drp <where_you_store_all_your_datasets> \
    --ckpt_path <where_you_save_the_ckpt>
done
```

CIFAR100-LT:

```
for dout in texture svhn cifar tin lsun places365
do
python test.py --gpu 0 --ds cifar100 --dout $dout \
    --drp <where_you_store_all_your_datasets> \
    --ckpt_path <where_you_save_the_ckpt>
done
```

ImageNet-LT:

```
python test_imagenet.py --gpu 0 \
    --drp <where_you_store_all_your_datasets> \
    --ckpt_path <where_you_save_the_ckpt>
```

Use stage 1 model to test OOD detection performance and stage 2 model to test in-distribution classification performance. 
Stage 1 and 2 models have identical parameters except those few in BN, the last fully connected layers and  the small convolutions in skip connections on ImageNet models. We save them as two separate models for convenience. 

To train or test our pretrained ImageNet model using ImageNet-10k dataset, you need to download it on your own and place it in the path indicated by `--drp`.

## Pretrained models

Pretrained models are available on [Google Drive](https://drive.google.com/drive/folders/1Z2VkeY8e6XIyEu995bSJBV628MXBH5ZZ?usp=sharing)


## Acknowledgement

Part of our codes are adapted from these repos:

pytorch-cifar - https://github.com/kuangliu/pytorch-cifar - MIT license

SupContrast - https://github.com/HobbitLong/SupContrast - BSD-2-Clause license

outlier-exposure - https://github.com/hendrycks/outlier-exposure - Apache-2.0 license

Long-Tailed-Recognition.pytorch - https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch - GPL-3.0 license

## Citation
```
@inproceedings{wang2022partial,
  title={Partial and Asymmetric Contrastive Learning for Out-of-Distribution Detection in Long-Tailed Recognition},
  author={Wang, Haotao and Zhang, Aston and Zhu, Yi and Zheng, Shuai and Li, Mu and Smola, Alex J and Wang, Zhangyang},
  booktitle={International Conference on Machine Learning},
  pages={23446--23458},
  year={2022},
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
