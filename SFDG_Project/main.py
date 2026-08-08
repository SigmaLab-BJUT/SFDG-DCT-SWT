from config import train_config
from train import distributed_train, main_worker
from evaluation import all_eval
import argparse
import fire
import torch
import subprocess
# torch.autograd.set_detect_anomaly(True)


def pretrain():
    name = 'SFDG_wo_AMRB'
    url = 'tcp://127.0.0.1:27015'
    Config = train_config(name, ['wild', 'efficientnet-b4'], url=url, attention_layer='b5',
                          feature_layer='logits', epochs=20, batch_size=16, AGDA_loss_weight=0)
    Config.mkdirs()
    distributed_train(Config)
    procs = [subprocess.Popen(['/bin/bash', '-c', 'CUDA_VISIBLE_DEVICES={} python main.py test {} {}'.format(i, name, j)])
             for i, j in enumerate(range(-2, 0))]
    for i in procs:
        i.wait()

# do pretrain first!
def aexp():
    name = 'SFDG_wo_AMRB'
    url = 'tcp://127.0.0.1:27018'
    Config = train_config(name, ['ff-all-c40', 'efficientnet-b4'], url=url, attention_layer='b5', feature_layer='b2', epochs=50, batch_size=16,
                          ckpt='', inner_margin=[0.2, -0.8], margin=0.8, AGDA_loss_weight=0)  # checkpoints/Efb4/ckpt_7.pth
    Config.mkdirs()
    distributed_train(Config)
    procs = [subprocess.Popen(['/bin/bash', '-c', 'CUDA_VISIBLE_DEVICES={} python main.py test {}'.format(i, name)])
             for i in enumerate(range(-3,0))]
    for i in procs:
        i.wait()


def resume(name, epochs=0):
    Config = train_config.load(name)
    Config.epochs += epochs
    Config.reload()
    Config.resume_optim = True
    distributed_train(Config)
    for i in range(-2, 0):
        all_eval(name, i)


def test(name, ckpt=None):
    all_eval(name, ckpt)


if __name__ == "__main__":
    fire.Fire()
