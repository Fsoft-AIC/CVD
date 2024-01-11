# Continual Variational Dropout
This repository contains all of our code for Continual Variational Dropout: A View of Auxiliary Local Variables in Continual Learning.

## Dataset

To download the [Omniglot dataset](https://drive.google.com/file/d/19UaTcjGYj8YUBlj69mPK7zcVvFUR8bso/view).

## Perform Training

```
$ python3 main.py --experiment [dataset] --approach [approach] --film --KL_weight [KL_weight] --prior_var [prior_var]
```

To perform the CVD on split-mnist, enter the following command:
```
$ python3 main.py --experiment split_mnist --approach gvclf_vd --film --KL_weight 0.01
```

To perform CVD on split-CIFAR100, enter the following command:
```
$ python3 main.py --experiment split_cifar100 --approach gvclf_vd --film --KL_weight 0.01 --conv_Dropout --prior_var 1
```

## Acknowledgement
Our implementation is based on [yolky/gvcl](https://github.com/yolky/gvcl).

## Citation
```bibtex
@article{hainam2023continual,
  title={Continual variational dropout: a view of auxiliary local variables in continual learning},
  author={Hai, Nam Le and Nguyen, Trang and Van, Linh Ngo and Nguyen, Thien Huu and Than, Khoat},
  journal={Machine Learning},
  pages={1--43},
  year={2023},
  publisher={Springer}
}
```
