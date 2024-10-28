import os
import torch
from torch.utils.tensorboard import SummaryWriter


class BaseWriter(object):
    def __init__(self, opt):
        self.rank = opt.global_rank
    def add_scalar(self, step, key, val):
        pass  # do nothing
    def add_image(self, step, key, image):
        pass  # do nothing
    def close(self): pass


class TensorBoardWriter(BaseWriter):
    def __init__(self, opt, path_to_save="./saved_info/dd_gan", dir_name=None):
        super(TensorBoardWriter, self).__init__(opt)
        if self.rank == 0:

            exp = opt.exp
            if dir_name is None:
                dir_name = opt.dataset
            parent_dir = os.path.join(path_to_save, dir_name)
            print(f"parent_dir = {parent_dir}")
            run_dir = os.path.join(parent_dir, exp)
            print(f"run_dir = {run_dir}")

            os.makedirs(run_dir, exist_ok=True)
            print(f"run dir for tensorboard = {run_dir}")
            self.writer = SummaryWriter(log_dir=run_dir, flush_secs=20)

    def add_scalar(self, global_step, key, val):
        if self.rank == 0:
            self.writer.add_scalar(key, val, global_step=global_step)

    def add_image(self, global_step, key, image):
        if self.rank == 0:
            image = image.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8)
            self.writer.add_image(key, image, global_step=global_step)

    def close(self):
        if self.rank == 0:
            self.writer.close()
