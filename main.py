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

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    model = model(opt)

    model.fit()