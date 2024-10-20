Last login: Tue Apr  4 17:18:45 on ttys002
shuangyanyang@Shuangyans-MacBook-Pro ~ % cd .ssh
shuangyanyang@Shuangyans-MacBook-Pro .ssh % ll
zsh: command not found: ll
shuangyanyang@Shuangyans-MacBook-Pro .ssh % ls -a
.		config		id_ed25519.pub	id_rsa.pub	known_hosts
..		id_ed25519	id_rsa		jie_ren_ssh.key	known_hosts.old
shuangyanyang@Shuangyans-MacBook-Pro .ssh % vim id_rsa.pub 
shuangyanyang@Shuangyans-MacBook-Pro .ssh % cd
shuangyanyang@Shuangyans-MacBook-Pro ~ % cd Desktop 
shuangyanyang@Shuangyans-MacBook-Pro Desktop % ssh cc@192.5.86.188
Enter passphrase for key '/Users/shuangyanyang/.ssh/id_rsa': 
Welcome to Ubuntu 18.04.6 LTS (GNU/Linux 4.15.0-206-generic x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/advantage

 System information disabled due to load higher than 48.0

 * Strictly confined Kubernetes makes edge and IoT secure. Learn how MicroK8s
   just raised the bar for easy, resilient and secure K8s cluster deployment.

   https://ubuntu.com/engage/secure-kubernetes-at-the-edge

23 updates can be applied immediately.
To see these additional updates run: apt list --upgradable

New release '20.04.6 LTS' available.
Run 'do-release-upgrade' to upgrade to it.


*** System restart required ***
Last login: Thu Apr  6 23:19:19 2023 from 169.236.78.22
cc@ysy-rtx-6000-core:~$ cd Betty_baseline/
cc@ysy-rtx-6000-core:~/Betty_baseline$ ll
total 68
drwxrwxr-x  6 cc cc  4096 Apr  6 23:56 ./
drwxr-xr-x 27 cc cc  4096 Apr 19 21:55 ../
drwxrwxr-x  8 cc cc  4096 Apr  6 23:56 .git/
drwxrwxr-x  9 cc cc  4096 Apr  7 02:10 Figures/
-rw-rw-r--  1 cc cc 35149 Apr  6 23:56 LICENSE
-rw-rw-r--  1 cc cc  1490 Apr  6 23:56 README.md
drwxrwxr-x  8 cc cc  4096 Apr  9 03:50 dataset/
-rw-rw-r--  1 cc cc  1556 Apr  6 23:56 install_requirements.sh
drwxrwxr-x  7 cc cc  4096 Apr  7 02:11 pytorch/
cc@ysy-rtx-6000-core:~/Betty_baseline$ cd Figures/
cc@ysy-rtx-6000-core:~/Betty_baseline/Figures$ ll
total 36
drwxrwxr-x 9 cc cc 4096 Apr  7 02:10 ./
drwxrwxr-x 6 cc cc 4096 Apr  6 23:56 ../
drwxrwxr-x 3 cc cc 4096 Apr 19 21:23 Betty_baseline/
drwxrwxr-x 7 cc cc 4096 Apr 19 21:06 bucket/
drwxrwxr-x 3 cc cc 4096 Apr  6 23:56 figure12/
drwxrwxr-x 3 cc cc 4096 Apr 11 21:56 figure13/
drwxrwxr-x 5 cc cc 4096 Apr  6 23:56 figure2_and_figure_10/
drwxrwxr-x 3 cc cc 4096 Apr  6 23:56 figure5/
drwxrwxr-x 4 cc cc 4096 Apr  6 23:56 figure9/
cc@ysy-rtx-6000-core:~/Betty_baseline/Figures$ cd bucket/
cc@ysy-rtx-6000-core:~/Betty_baseline/Figures/bucket$ ll
total 200
drwxrwxr-x 7 cc cc  4096 Apr 19 21:06 ./
drwxrwxr-x 9 cc cc  4096 Apr  7 02:10 ../
-rw-rw-r-- 1 cc cc 12237 Apr 14 20:01 AA_loss.py
-rw-rw-r-- 1 cc cc 12219 Apr 14 19:17 AA_loss_log.py
-rw-rw-r-- 1 cc cc 12657 Apr 19 01:44 AA_loss_mem.py
-rw-rw-r-- 1 cc cc 14485 Apr 19 20:39 AA_loss_time.py
-rw-rw-r-- 1 cc cc 13589 Apr 19 21:28 AA_pure_train_time.py
drwxrwxr-x 2 cc cc  4096 Apr  7 03:51 __pycache__/
drwxrwxr-x 4 cc cc  4096 Apr 19 03:30 arxiv_2_layer/
drwxrwxr-x 2 cc cc  4096 Apr 11 23:51 bak/
drwxrwxr-x 2 cc cc  4096 Apr 13 22:25 compare_res/
-rw-rw-r-- 1 cc cc  2422 Apr  7 03:30 cpu_mem_usage.py
-rw-rw-r-- 1 cc cc  9288 Apr  7 03:31 graphsage_model.py
-rw-rw-r-- 1 cc cc 18958 Apr  7 03:32 load_graph.py
drwxrwxr-x 2 cc cc  4096 Apr 11 22:51 log/
-rw-rw-r-- 1 cc cc 11920 Apr 19 20:43 lstm_arxiv_2_layer_3_nb.log
-rw-rw-r-- 1 cc cc  1258 Apr  7 03:30 memory_usage.py
-rw-rw-r-- 1 cc cc 12809 Apr  9 04:19 multi-layer-bucket.py
-rw-rw-r-- 1 cc cc 12753 Apr 11 22:52 multi_layer_bucket_loss.py
-rw-rw-r-- 1 cc cc  4845 Apr  7 03:30 my_utils.py
-rwxrwxrwx 1 cc cc  1344 Apr 11 23:08 run.sh*
-rw-rw-r-- 1 cc cc  1767 Apr  7 03:31 utils.py
cc@ysy-rtx-6000-core:~/Betty_baseline/Figures/bucket$ vim AA_pure_train_time.py  
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'..')
sys.path.insert(0,'../../pytorch/utils/')
sys.path.insert(0,'../../pytorch/bucketing/')
sys.path.insert(0,'../../pytorch/models/')
import dgl
from dgl.data.utils import save_graphs
import numpy as np
from statistics import mean
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from bucketing_dataloader import generate_dataloader_bucket_block

import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm                    
