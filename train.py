"""
Train the reward model.
Adapted from ImageReward (https://github.com/THUDM/ImageReward).
"""
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.backends import cudnn
from torch.distributed import init_process_group, destroy_process_group

from dataset import Dataset, DataCollator
from cyclereward import CycleReward
from utils import save_model, get_rank, visualizer, EarlyStopping
from learning_rates import get_learning_rate_scheduler


def init_seeds(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
       cudnn.deterministic = True
       cudnn.benchmark = False
    else:  # faster, less reproducible
       cudnn.deterministic = False
       cudnn.benchmark = True
       
def loss_fn(reward):
    chosen_reward = reward[:, 0] 
    reject_reward = reward[:, 1]
    logits = chosen_reward - reject_reward
    loss = -F.logsigmoid(logits)
    return loss.mean(), logits

def evaluate(model, dataloader, train_data_ratio, iteration, writer, model_type):
    model.eval()
    loss_stats = {k: 0.0 for k in ["loss", "loss_text", "loss_image"]}
    acc_stats = {k: [] for k in ["text", "image"]}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating"):
            reward_text, reward_image = model(batch)
            if 'combo' in model_type:
                loss_text, logits_text = loss_fn(reward_text)
                loss_image, logits_image = loss_fn(reward_image)
                loss = loss_text * train_data_ratio + loss_image
                acc_text = torch.mean((logits_text > 0).clone().detach().float())
                acc_image = torch.mean((logits_image > 0).clone().detach().float())
            else:
                loss, logits_text = loss_fn(reward_text)
                loss_image = torch.zeros_like(loss)
                loss_text = loss
                acc_text = torch.mean((logits_text > 0).clone().detach().float())
                acc_image = torch.zeros_like(acc_text)

            loss_stats["loss"] += loss.item()
            loss_stats["loss_text"] += loss_text.item()
            loss_stats["loss_image"] += loss_image.item()
            acc_stats["text"].append(acc_text.item())
            acc_stats["image"].append(acc_image.item())

    if get_rank() == 0:
        logs = {
            "loss": loss_stats["loss"] / len(dataloader),
            "loss_text": loss_stats["loss_text"] / len(dataloader),
            "loss_image": loss_stats["loss_image"] / len(dataloader),
            "acc_text": sum(acc_stats["text"]) / len(dataloader),
            "acc_image": sum(acc_stats["image"]) / len(dataloader),
        }
        print(f"Iter {iteration} | Loss {logs['loss']:.5f} | Text Loss {logs['loss_text']:.5f} | Image Loss {logs['loss_image']:.5f} | Text Acc {logs['acc_text']:.4f} | Image Acc {logs['acc_image']:.4f}")
        for k, v in logs.items():
            writer.add_scalar(f'train_{k}', v, global_step=iteration)
    
    return logs["loss"]

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # basic arguments
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the config file.')
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--checkpoint_base', type=str, default='checkpoints', help='Path to save checkpoints.')
    parser.add_argument('--visual_base', type=str, default='visualizer', help='Path to save visualizations.')
    parser.add_argument('--clear_visualizer', action='store_true', help='Clear the visualizer directory before training.')

    # dataset arguments
    parser.add_argument('--dataset', type=str, default=None, help='Name of the dataset to use.')
    parser.add_argument('--i2t_data_path', type=str, default=None, help='Path to the image-to-text dataset.')
    parser.add_argument('--t2i_data_path', type=str, default=None, help='Path to the text-to-image dataset.')
    parser.add_argument('--threshold_similar', type=float, default=0.005, help='Threshold for similar samples in the dataset.')
    parser.add_argument('--threshold_negative', type=float, default=0.7, 
                        help='Threshold for negative samples in the dataset. We use 0.7 for dreamsim and 0.4 for sbert_score.')

    # model arguments
    parser.add_argument('--model_type', type=str, default='CycleReward-Combo')
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument("--fix_rate", type=float, default=0.7)

    # training arguments
    parser.add_argument('--data_ratio', type=int, default=1, help='ratio between image and text loss')
    parser.add_argument('--batch_size', type=int, default=256, help='')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='')
    parser.add_argument('--save_steps', type=int, default=10000, help='')
    parser.add_argument('--epochs', type=int, default=10, help='')
    parser.add_argument('--train-iters', type=int, default=None,
                        help='total number of iterations to train over all training runs')
    parser.add_argument('--grad_clip', type=float, default=0, help='clip the gradient')
    parser.add_argument('--patience', type=int, default=5, help='early stopping in training')
    parser.add_argument('--lr', type=float, default=5e-06,
                        help='initial learning rate')
    parser.add_argument('--lr-decay-iters', type=int, default=None,
                        help='number of iterations to decay LR over,'
                            ' If None defaults to `--train-iters`*`--epochs`')
    parser.add_argument('--lr-decay-style', type=str, default='cosine',
                        choices=['constant', 'linear', 'cosine', 'exponential', 'inverse_square_root'],
                        help='learning rate decay function')
    parser.add_argument('--lr-decay-ratio', type=float, default=0.0)
    parser.add_argument('--warmup', type=float, default=0.0,
                        help='percentage of data to warmup on (.01 = 1% of all '
                            'training iters). Default 0.01')
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.999)
    parser.add_argument('--adam-eps', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--valid_per_epoch', type=int, default=1)

    # device arguments
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    gpu_num = torch.cuda.device_count()
    args.BatchSize = args.batch_size * args.accumulation_steps * gpu_num

    if args.distributed:
        # initialize distributed environment
        init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        init_seeds(args.seed + local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        init_seeds(args.seed)

    writer = visualizer(args)

    # load model
    model = CycleReward(
        device=device, 
        model_type=args.model_type, 
        max_length=args.max_length, 
        fix_rate=args.fix_rate
    ).to(device)
    
    # load dataset
    train_dataset = Dataset(args.i2t_data_path, args.t2i_data_path, "train", args.model_type, args.threshold_similar, args.threshold_negative)
    valid_dataset = Dataset(args.i2t_data_path, args.t2i_data_path, "valid", args.model_type, args.threshold_similar, args.threshold_negative)
    test_dataset = Dataset(args.i2t_data_path, args.t2i_data_path, "test", args.model_type, args.threshold_similar, args.threshold_negative)
    collate_fn = DataCollator(args.max_length, args.model_type)
    train_data_ratio = args.data_ratio

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=DistributedSampler(train_dataset) if args.distributed else None,
        shuffle=True,
        collate_fn=collate_fn
    )
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    # set the training iterations
    args.train_iters = args.epochs * len(train_loader)
    steps_per_valid = len(train_loader) // args.valid_per_epoch
    if get_rank() == 0:
        print(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps)
    scheduler = get_learning_rate_scheduler(optimizer, args)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)

    early_stopping = EarlyStopping(patience=args.patience)
    best_loss = 1e9
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        loss_stats = {k: 0.0 for k in ["loss", "loss_text", "loss_image"]}
        acc_stats = {k: [] for k in ["text", "image"]}
        
        for step, batch in enumerate(tqdm(train_loader)):
            model.train()
            reward_text, reward_image = model(batch)
            if 'Combo' in args.model_type:
                loss_text, logits_text = loss_fn(reward_text)
                loss_image, logits_image = loss_fn(reward_image)
                loss = loss_text * train_data_ratio + loss_image
                acc_text = torch.mean((logits_text > 0).clone().detach().float())
                acc_image = torch.mean((logits_image > 0).clone().detach().float())       
            else:
                loss, logits_text = loss_fn(reward_text)
                loss_image = torch.zeros_like(loss)
                loss_text = loss
                acc_text = torch.mean((logits_text > 0).clone().detach().float())
                acc_image = torch.zeros_like(acc_text)

            loss = loss / args.accumulation_steps # scale the loss to account for gradient accumulation
            loss.backward()

            if args.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            loss_stats["loss"] += loss.item() * args.accumulation_steps  # undo normalization for logging
            loss_stats["loss_text"] += loss_text.item() 
            loss_stats["loss_image"] += loss_image.item()
            acc_stats["text"].append(acc_text.item())
            acc_stats["image"].append(acc_image.item())

            iterations = epoch * len(train_loader) + step + 1
            
            # update parameters
            if (iterations % args.accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                # logging 
                if get_rank() == 0:
                    logs = {
                        "loss": loss_stats["loss"] / args.accumulation_steps,
                        "loss_text": loss_stats["loss_text"] / args.accumulation_steps,
                        "loss_image": loss_stats["loss_image"] / args.accumulation_steps,
                        "acc_text": sum(acc_stats["text"]) / len(acc_stats["text"]),
                        "acc_image": sum(acc_stats["image"]) / len(acc_stats["image"]),
                    }
                    print(f"Iter {iterations // args.accumulation_steps} | Loss {logs['loss']:.5f} | Text Loss {logs['loss_text']:.5f} | Image Loss {logs['loss_image']:.5f} | Text Acc {logs['acc_text']:.4f} | Image Acc {logs['acc_image']:.4f}")
                    for k, v in logs.items():
                        writer.add_scalar(f'train_{k}', v, global_step=iterations // args.accumulation_steps)
                
                # clear stats for next iteration
                loss_stats = {k: 0.0 for k in loss_stats}
                acc_stats = {k: [] for k in acc_stats}
            
            # validation
            if (iterations % steps_per_valid) == 0:
                if get_rank() == 0:
                    valid_loss = evaluate(model, valid_loader, train_data_ratio, iterations, writer, args.model_type)
                    if valid_loss < best_loss:
                        print("Best Val loss so far. Saving model")
                        best_loss = valid_loss
                        save_model(model, filename="best")
                    
                    if early_stopping(valid_loss):
                        print(f"Early stopping triggered at epoch {epoch}")
                        break 

        if early_stopping.counter >= args.patience:
            break  

    # explicitly clean up before exiting
    if args.distributed:
        destroy_process_group()

if __name__ == "__main__":
    main()