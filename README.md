# gvclf_vd

#### To run CVD on split-mnist, enter the following command:

```
$ python3 main.py --experiment split_mnist --approach gvclf_vd --film --KL_weight 0.01
```

#### To run CVD on split-CIFAR100, enter the following command:

```
$ python3 main.py --experiment split_cifar100 --approach gvclf_vd --film --KL_weight 0.01 --conv_Dropout --prior_var 1
```
