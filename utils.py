import os, shutil
import torch
from tensorboardX import SummaryWriter
import torch.distributed as dist


def make_path(args):
    path = "{}_fix{}_{}_{}_{}_{}_bs{}_lr{}{}_wrm{}_wd{}_gc{}".format(
         args.model_type, args.fix_rate, args.dataset, args.threshold_similar, args.threshold_negative, args.data_ratio, args.BatchSize, args.lr, args.lr_decay_style, args.warmup, args.weight_decay, args.grad_clip)
    return path

def save_model(model, filename="checkpoint", args=None):
    save_path = make_path(args)
    save_path = os.path.join(args.checkpoint_base, save_path)
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)
    model_name = os.path.join(save_path, f'{filename}.pth')
    model_without_ddp = model.module if hasattr(model, "module") else model
    torch.save(model_without_ddp.state_dict(), model_name)
    if get_rank() == 0:
        print('model saved to %s'%model_name)

def load_model(model, filename="best", args=None):
    if args.ckpt_path is not None:
        model_name = args.ckpt_path
    else:
        save_path = make_path(args)
        save_path = os.path.join(args.checkpoint_base, save_path)
        model_name = os.path.join(save_path, f'{filename}.pth')
        
    print('load checkpoint from %s'%model_name)
    checkpoint = torch.load(model_name, map_location='cpu') 
    state_dict = checkpoint
    msg = model.load_state_dict(state_dict,strict=False)
    print("missing keys:", msg.missing_keys)

    return model 

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path, 0o777, exist_ok=True)

def visualizer(args):
    if get_rank() == 0:
        save_path = make_path(args)
        filewriter_path = os.path.join(args.visual_base, save_path)
        if args.clear_visualizer and os.path.exists(filewriter_path):   # 删掉以前的summary，以免重合
            shutil.rmtree(filewriter_path)
        makedir(filewriter_path)
        writer = SummaryWriter(filewriter_path, comment='visualizer')
        return writer
    
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        """
        :param patience: Number of epochs to wait after the last improvement.
        :param min_delta: Minimum change in monitored value to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset patience counter
        else:
            self.counter += 1  # Increase counter if no improvement

        return self.counter >= self.patience  # Stop if patience is exceeded