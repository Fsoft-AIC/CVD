import os

lamb = [400, 4000, 10000, 40000, 100000]

beta = [0.0001, 0.0002, 0.001, 0.002, 0.01, 0.3]
alpha = [0.01, 0.3, 5]
ratio = [0.125, 0.5]

droprate = [0.5, 0.2, 0.1]

KL_weight = [1, 0.1, 0.01, 0.001, 0.0001]
init_alpha = [0.5, 0.2, 0.1]

seed = [0,1,2,3,4]
commands = []

for s in seed:
    # cmd = "python main.py --seed {} --experiment pmnist --approach ewc --drop_type Gauss --droprate 0.5 --KL_weight 1 --lamb 400".format(s)
    # cmd = "python main.py --seed {} --experiment pmnist --approach vcl_vd --drop_type Gauss --KL_weight 0.1 --droprate 0.2 --num_samples 1 --test_samples 100 --KL_weight_theta 1 --local_trick".format(s)
    cmd = "python3 main.py --seed {} --experiment pmnist --approach ucl_vd --drop_type Gauss --droprate 0.5 --KL_weight 0.001 --local_trick --beta 0.03 --ratio 0.5 --lr_rho 0.001 --alpha 0.01 --num_samples 1 --test_sample 100".format(s)
    commands.append(cmd)

for cmd in commands:
    os.system(cmd)
