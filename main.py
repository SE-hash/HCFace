import torch
import torch.distributed as dist
from models.model import model
import os
from torchsummary import summary
import distutils.version

if __name__ == '__main__':
    parser = model.parser()
    opt = parser.parse_args()
    print(opt)

"""
    train AIFR
    torchrun --nproc_per_node="number of your gpu" --master_port=17647 main.py --train_fr \
        --backbone_name ir50 --head_s 64 --head_m 0.35 --weight_decay 5e-4 \
            --momentum 0.9 --fr_age_loss_weight 0.001 --fr_da_loss_weight 0.002 \
                --age_group 7 --gamma 0.1 --milestone 20000 23000 --warmup 1000 \
                    --learning_rate 0.1 --dataset_name "dataset" --image_size 112 \
                        --num_iter 36000 --batch_size 64 --amp
                        
    train FAS
    torchrun --nproc_per_node="number of your gpu" --master_port=17647 main.py     \
    --train_fas --backbone_name ir50 --age_group 7    \
     --dataset_name "dataset" --image_size 112 --num_iter 36000 --batch_size 64     \
     --d_lr 1e-5 --g_lr 1e-5 --fas_gan_loss_weight 75 --fas_age_loss_weight 10     \
     --fas_id_loss_weight 0.002
"""


    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    model = model(opt)

    model.fit()
