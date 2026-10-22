import os
import yaml
import torch
import wandb
import random
import torch_geometric
import torch.nn.functional as F
import time
from datetime import timedelta

from torch.utils.data import ConcatDataset, Subset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from typing import Optional
from basemodels.utils.scalegmn.data import dataset
from basemodels.utils.scalegmn.scalegmn.models import ScaleGMN
from basemodels.utils.scalegmn.utils.loss import select_criterion
from basemodels.utils.scalegmn.utils.optim import setup_optimization
from basemodels.utils.scalegmn.utils.helpers import count_parameters, set_seed, mask_input, mask_hidden
from basemodels.utils.scalegmn.utils.setup_arg_parser import setup_arg_parser
from utils.wandb_manager import WandbManager


class SimpleTimer:
    def __init__(self, total_steps, start_step=0):
        self.total = total_steps
        self.start_step = start_step
        self.start = time.time()

    def get_eta(self, current_step):
        elapsed = time.time() - self.start
        steps_done = current_step - self.start_step
        speed = steps_done / elapsed if elapsed > 0 and steps_done > 0 else 0
        eta = (self.total - current_step) / speed if speed > 0 else 0
        return str(timedelta(seconds=int(eta)))


def main(conf, *, use_ddp=False, rank=0, world_size=1, local_rank=0, wandb_manager: Optional[WandbManager] = None):
    torch.set_float32_matmul_precision('high')

    # print(yaml.dump(conf, default_flow_style=False))
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    is_main = not use_ddp or rank == 0

    # Initialize wandb manager if not provided
    owns_wandb = wandb_manager is None
    if owns_wandb:
        wandb_manager = WandbManager(enabled=conf.get("wandb", False))

    # Initialize wandb run
    if wandb_manager.enabled and wandb_manager.current_run is None:
        dataset_name = conf['data'].get('dataset', 'unknown').replace('_inr', '')
        run_name = wandb_manager.generate_run_name(
            dataset=dataset_name,
            model="scalegmn",
            training_type="classifier",
        )
        wandb_manager.init_run(
            name=run_name,
            group=f"scalegmn_{dataset_name}",
            tags=wandb_manager.generate_tags(
                dataset=dataset_name,
                model="scalegmn",
                training_type="classifier",
            ),
            config=conf,
            job_type="training",
        )

    set_seed(conf['train_args']['seed'])

    # set up save directory for CIFAR datasets
    save_dir = conf['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if is_main: print(f"Saving models to {save_dir}")
    
    # =============================================================================================
    #   SETUP DATASET AND DATALOADER
    # =============================================================================================
    equiv_on_hidden = mask_hidden(conf)
    get_first_layer_mask = mask_input(conf)
    train_set = dataset(conf['data'],
                        split='train',
                        prefix=conf['prefix'],
                        direction=conf['scalegmn_args']['direction'],
                        equiv_on_hidden=equiv_on_hidden,
                        get_first_layer_mask=get_first_layer_mask)
    conf['scalegmn_args']["layer_layout"] = train_set.get_layer_layout()

    if conf['extra_aug']:
        aug = []
        for i in range(1, conf['extra_aug']):
            aug.append(dataset(conf['data'],
                            split='train',
                            prefix=f"{conf['prefix']}_{i}",
                            direction=conf['scalegmn_args']['direction'],
                            equiv_on_hidden=equiv_on_hidden,
                            get_first_layer_mask=get_first_layer_mask)
            )
        train_set = ConcatDataset([train_set] + aug)

    val_set = dataset(conf['data'],
                      split='val',
                      prefix=conf['prefix'],
                      direction=conf['scalegmn_args']['direction'],
                      equiv_on_hidden=equiv_on_hidden,
                      get_first_layer_mask=get_first_layer_mask)

    test_set = dataset(conf['data'],
                       split='test',
                       prefix=conf['prefix'],
                       direction=conf['scalegmn_args']['direction'],
                       equiv_on_hidden=equiv_on_hidden,
                       get_first_layer_mask=get_first_layer_mask)
    
    if is_main: print(f'Len train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}, batch size: {conf["batch_size"]}')

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    
    train_loader = torch_geometric.loader.DataLoader(
        dataset=train_set,
        batch_size=conf["batch_size"],
        shuffle=not use_ddp,
        num_workers=conf["num_workers"],
        pin_memory=True,
        sampler=train_sampler,
    )
    if use_ddp:
        indices = random.sample(range(len(train_set)), len(val_set))
        train_subset = Subset(train_set, indices)
        train_loader_for_val = torch_geometric.loader.DataLoader(
            dataset=train_subset, 
            batch_size=conf["batch_size"], 
            shuffle=True,             
            num_workers=conf["num_workers"],
            pin_memory=True,
        )

        Subset(train_set, len(val_set))
    else: 
        train_loader_for_val = train_loader

    val_loader = torch_geometric.loader.DataLoader(
        dataset=val_set,
        batch_size=conf["batch_size"],
        shuffle=False,
    )
    test_loader = torch_geometric.loader.DataLoader(
        dataset=test_set,
        batch_size=conf["batch_size"],
        shuffle=False,
        num_workers=conf["num_workers"],
        pin_memory=True,
    )

    # =============================================================================================
    #   DEFINE MODEL
    # =============================================================================================
    base_model = ScaleGMN(conf['scalegmn_args'])
    base_model = base_model.to(device)
    # print(net)

    if use_ddp:
        net = DDP(base_model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        net = base_model

    cnt_p = count_parameters(net=base_model)
    wandb_manager.log({'number of parameters': cnt_p}, step=0)

    for p in net.parameters():
        p.requires_grad = True

    # =============================================================================================
    #   DEFINE LOSS
    # =============================================================================================
    criterion = select_criterion(conf['train_args']['loss'], {})

    # =============================================================================================
    #   DEFINE OPTIMIZATION
    # =============================================================================================
    conf_opt = conf['optimization']
    model_params = [p for p in net.parameters() if p.requires_grad]
    optimizer, scheduler = setup_optimization(model_params, optimizer_name=conf_opt['optimizer_name'], optimizer_args=conf_opt['optimizer_args'], scheduler_args=conf_opt['scheduler_args'])
    # =============================================================================================

    # =============================================================================================
    # RESUME FROM CHECKPOINT
    # =============================================================================================
    start_epoch = 0
    ckpt_path = os.path.join(save_dir, "checkpoint.pt")
    best_path = os.path.join(save_dir, "best.pt")
    
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        base_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler[0] is not None and "scheduler" in ckpt:
            scheduler[0].load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", -1)
        if is_main: print(f"Resumed from checkpoint epoch {start_epoch}")

    elif os.path.exists(best_path):
        base_model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
        start_epoch = conf.get("resume_epoch", 0)
        if is_main: print(f"Loaded best.pt, starting from epoch {start_epoch}")
        best_val_acc = -1
    else:
        best_val_acc = -1
    # =============================================================================================
    # TRAINING LOOP
    # =============================================================================================

    best_train_acc = -1
    best_test_results, best_val_results, best_train_results, best_train_results_TRAIN = None, None, None, None
    last_val_accs = []
    patience = conf['train_args']['patience']

    should_stop = torch.tensor(0, device=device)  # bool 대신 tensor
    total_steps = conf['train_args']['num_epochs'] * len(train_loader)
    timer = SimpleTimer(total_steps, start_step=start_epoch * len(train_loader))
    for epoch in range(start_epoch, conf['train_args']['num_epochs']):
        net.train()
        curr_loss = 0
        len_dataloader = len(train_loader)
        if use_ddp:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            step = epoch * len_dataloader + i
            batch = batch.to(device)

            optimizer.zero_grad()
            out = net(batch)

            loss = criterion(out, batch.label)
            loss.backward()
            log = {}
            if conf['optimization']['clip_grad']:
                log['grad_norm'] = torch.nn.utils.clip_grad_norm_(net.parameters(), conf['optimization']['clip_grad_max_norm'])

            optimizer.step()

            log[f"train/{conf['train_args']['loss']}"] = loss.item()
            log["epoch"] = epoch

            if scheduler[1] is not None and scheduler[1] != 'ReduceLROnPlateau':
                log["lr"] = scheduler[0].get_last_lr()[0]
                scheduler[0].step()

            if is_main: wandb_manager.log(log, step=step)

            if i % 100 == 0 and i > 0:
                print(f"Epoch {epoch}/{conf['train_args']['num_epochs']}, Step {i}/{len_dataloader} | ETA: {timer.get_eta(step)} | train_loss: {loss.item():.3f}", flush=True)
        #############################################
        # VALIDATION
        #############################################
        if conf["validate"] and is_main:
            print(f"\nValidation after epoch {epoch}:")
            val_loss_dict = evaluate(base_model, val_loader, device=device)
            test_loss_dict = evaluate(base_model, test_loader, device=device)
            val_loss = val_loss_dict["avg_loss"]
            val_acc = val_loss_dict["avg_acc"]
            test_loss = test_loss_dict["avg_loss"]
            test_acc = test_loss_dict["avg_acc"]

            train_loss_dict = evaluate(base_model, train_loader_for_val, device=device)

            best_val_criteria = val_acc >= best_val_acc

            if best_val_criteria:
                best_val_acc = val_acc
                best_test_results = test_loss_dict
                best_val_results = val_loss_dict
                best_train_results = train_loss_dict
                if save_dir is not None:
                    torch.save(base_model.state_dict(), os.path.join(save_dir, "best.pt"))

            best_train_criteria = train_loss_dict["avg_acc"] >= best_train_acc
            if best_train_criteria:
                best_train_acc = train_loss_dict["avg_acc"]
                best_train_results_TRAIN = train_loss_dict

            log = {
                "train/avg_loss": train_loss_dict["avg_loss"],
                "train/acc": train_loss_dict["avg_acc"],
                "train/best_loss": best_train_results["avg_loss"],
                "train/best_acc": best_train_results["avg_acc"],
                "train/best_loss_TRAIN_based": best_train_results_TRAIN["avg_loss"],
                "train/best_acc_TRAIN_based": best_train_results_TRAIN["avg_acc"],
                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/best_loss": best_val_results["avg_loss"],
                "val/best_acc": best_val_results["avg_acc"],
                "test/loss": test_loss,
                "test/acc": test_acc,
                "test/best_loss": best_test_results["avg_loss"],
                "test/best_acc": best_test_results["avg_acc"],
                "epoch": epoch,
            }
            # Log confusion matrices only if wandb is active
            if wandb_manager.is_active:
                log["train/conf_mat"] = wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=train_loss_dict["gt"],
                    preds=train_loss_dict["predicted"],
                    class_names=range(10),
                )
                log["val/conf_mat"] = wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=val_loss_dict["gt"],
                    preds=val_loss_dict["predicted"],
                    class_names=range(10),
                )
                log["test/conf_mat"] = wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=test_loss_dict["gt"],
                    preds=test_loss_dict["predicted"],
                    class_names=range(10),
                )

            wandb_manager.log(log)
            print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

            last_val_accs.append(val_acc)
            # Keep only the last accuracies
            if len(last_val_accs) > patience:
                last_val_accs.pop(0)
            # Check if the accuracies are decreasing
            if len(last_val_accs) == patience and all(x > y for x, y in zip(last_val_accs, last_val_accs[1:])):
                print(f"Validation accuracy has been dropping for {patience} consecutive epochs:\n{last_val_accs}\nExiting.")
                if save_dir is not None:
                    torch.save(base_model.state_dict(), os.path.join(save_dir, "final.pt"))
                should_stop.fill_(1)

        if use_ddp:
            dist.barrier()  # Ensure all processes are synced before next validation phase
            dist.broadcast(should_stop, src=0)

        if is_main: # save every epoch
            torch.save({
                    "model": base_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler[0].state_dict() if scheduler[0] is not None else None,
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
            }, os.path.join(save_dir, "checkpoint.pt"))

        if should_stop.item():
            break

    # save final model after finishing all epochs
    if save_dir is not None and is_main:
        torch.save(base_model.state_dict(), os.path.join(save_dir, "final.pt"))

@torch.no_grad()
def evaluate(model, loader, device=None):

    model.eval()
    loss = 0.0
    correct = 0.0
    total = 0.0
    predicted, gt = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss += F.cross_entropy(out, batch.label, reduction="sum")
        total += len(batch.label)
        pred = out.argmax(1)
        correct += pred.eq(batch.label).sum()
        predicted.extend(pred.cpu().numpy().tolist())
        gt.extend(batch.label.cpu().numpy().tolist())

    model.train()
    avg_loss = loss / total
    avg_acc = correct / total

    return dict(avg_loss=avg_loss, avg_acc=avg_acc, predicted=predicted, gt=gt)


if __name__ == '__main__':
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()

    if isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]

    conf = yaml.safe_load(open(args.conf))

    main(args=args)
