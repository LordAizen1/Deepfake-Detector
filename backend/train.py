# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
import wandb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import numpy as np

from dataset import get_dataloaders

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "model_name"    : "efficientnet_b4",
    "pretrained"    : True,
    "num_epochs"    : 20,
    "batch_size"    : 32,
    "lr"            : 1e-4,
    "weight_decay"  : 1e-4,
    "freeze_epochs" : 3,        # freeze backbone for first N epochs
    "num_workers"   : 4,
    "face_dir"      : "faces",
    "save_dir"      : "checkpoints",
    "wandb_project" : "deepfake-detection",
    "wandb_run"     : "efficientnet-b4-multi-adv",
    "device"        : "cuda" if torch.cuda.is_available() else "cpu",
    "adv_epsilon"   : 0.02,     # FGSM perturbation strength (ε)
}


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def build_model(model_name: str, pretrained: bool) -> nn.Module:
    """
    Load EfficientNet-B4 from timm, replace classifier head
    with a single sigmoid output for binary classification.
    """
    model = timm.create_model(model_name, pretrained=pretrained)

    # replace final classifier — EfficientNet-B4 has 'classifier' as head
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 1)   # single logit → BCEWithLogitsLoss
    )
    return model


def freeze_backbone(model: nn.Module):
    """Freeze everything except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    print("  Backbone frozen — training head only")


def unfreeze_backbone(model: nn.Module):
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    print("  Backbone unfrozen — full fine-tuning")


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_metrics(all_labels, all_probs):
    all_preds = (np.array(all_probs) >= 0.5).astype(int)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0

    return {"accuracy": acc, "f1": f1, "auc": auc}


# ─────────────────────────────────────────────
# FGSM ADVERSARIAL PERTURBATION
# ─────────────────────────────────────────────
def fgsm_perturb(model, imgs, labels, criterion, epsilon):
    """Return FGSM-perturbed batch. Grad computation is isolated."""
    model.eval()
    imgs_adv = imgs.clone().detach().requires_grad_(True)
    logits = model(imgs_adv)
    loss   = criterion(logits, labels)
    loss.backward()
    imgs_adv = (imgs_adv + epsilon * imgs_adv.grad.sign()).detach()
    model.train()
    return imgs_adv


# ─────────────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    all_labels, all_probs = [], []
    epsilon = CONFIG["adv_epsilon"]

    pbar = tqdm(loader, desc=f"  Train epoch {epoch}", leave=False)
    for imgs, labels in pbar:
        imgs   = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)   # [B] → [B, 1]

        # adversarial examples (separate forward/backward, no gradient leak)
        imgs_adv = fgsm_perturb(model, imgs, labels, criterion, epsilon)

        optimizer.zero_grad()
        logits_clean = model(imgs)
        loss_clean   = criterion(logits_clean, labels)
        loss_adv     = criterion(model(imgs_adv), labels)
        loss = 0.5 * loss_clean + 0.5 * loss_adv
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits_clean).detach().cpu().squeeze().tolist()
        if isinstance(probs, float):
            probs = [probs]

        all_probs  += probs
        lbls = labels.cpu().squeeze().tolist()
        if isinstance(lbls, float):
            lbls = [lbls]
        all_labels += lbls
        running_loss += loss.item()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / len(loader)
    metrics  = compute_metrics(all_labels, all_probs)
    return avg_loss, metrics


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device, split="val"):
    model.eval()
    running_loss = 0.0
    all_labels, all_probs = [], []

    for imgs, labels in tqdm(loader, desc=f"  {split.capitalize()}", leave=False):
        imgs   = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)

        logits = model(imgs)
        loss   = criterion(logits, labels)

        probs = torch.sigmoid(logits).cpu().squeeze().tolist()
        if isinstance(probs, float):
            probs = [probs]

        all_probs  += probs
        lbls = labels.cpu().squeeze().tolist()
        if isinstance(lbls, float):
            lbls = [lbls]
        all_labels += lbls
        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    metrics  = compute_metrics(all_labels, all_probs)
    return avg_loss, metrics


# ─────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────
def train():
    # ── setup ──
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    device = torch.device(CONFIG["device"])
    print(f"\nUsing device: {device}")

    # ── W&B ──
    wandb.init(
        project = CONFIG["wandb_project"],
        name    = CONFIG["wandb_run"],
        config  = CONFIG,
    )

    # ── data ──
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        face_dir    = CONFIG["face_dir"],
        batch_size  = CONFIG["batch_size"],
        num_workers = CONFIG["num_workers"],
    )

    # ── model ──
    print(f"\nBuilding {CONFIG['model_name']}...")
    model = build_model(CONFIG["model_name"], CONFIG["pretrained"])
    model = model.to(device)
    wandb.watch(model, log="gradients", log_freq=100)

    # ── loss — weighted BCE to handle real/fake imbalance ──
    # fake:real = 6:1 in FF++ so we down-weight fake slightly
    pos_weight = torch.tensor([1.0]).to(device)   # adjust if needed
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── optimizer + scheduler ──
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = CONFIG["lr"],
        weight_decay = CONFIG["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["num_epochs"], eta_min=1e-6)

    # ── training loop ──
    best_val_auc = 0.0
    best_ckpt    = os.path.join(CONFIG["save_dir"], "best_model.pth")

    for epoch in range(1, CONFIG["num_epochs"] + 1):

        # freeze backbone for first N epochs, then unfreeze
        if epoch == 1:
            freeze_backbone(model)
            # re-init optimizer with only head params
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=CONFIG["lr"] * 10,    # higher LR for head-only phase
                weight_decay=CONFIG["weight_decay"],
            )
        elif epoch == CONFIG["freeze_epochs"] + 1:
            unfreeze_backbone(model)
            # lower LR for full fine-tuning
            optimizer = optim.AdamW(
                model.parameters(),
                lr           = CONFIG["lr"],
                weight_decay = CONFIG["weight_decay"],
            )
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max   = CONFIG["num_epochs"] - CONFIG["freeze_epochs"],
                eta_min = 1e-6,
            )

        print(f"\nEpoch {epoch}/{CONFIG['num_epochs']}")

        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device, split="val"
        )
        scheduler.step()

        # ── logging ──
        print(f"  Train  loss={train_loss:.4f}  acc={train_metrics['accuracy']:.4f}"
              f"  f1={train_metrics['f1']:.4f}  auc={train_metrics['auc']:.4f}")
        print(f"  Val    loss={val_loss:.4f}    acc={val_metrics['accuracy']:.4f}"
              f"  f1={val_metrics['f1']:.4f}  auc={val_metrics['auc']:.4f}")

        wandb.log({
            "epoch"          : epoch,
            "train/loss"     : train_loss,
            "train/accuracy" : train_metrics["accuracy"],
            "train/f1"       : train_metrics["f1"],
            "train/auc"      : train_metrics["auc"],
            "val/loss"       : val_loss,
            "val/accuracy"   : val_metrics["accuracy"],
            "val/f1"         : val_metrics["f1"],
            "val/auc"        : val_metrics["auc"],
            "lr"             : optimizer.param_groups[0]["lr"],
        })

        # ── save best checkpoint ──
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "val_auc"    : best_val_auc,
                "config"     : CONFIG,
            }, best_ckpt)
            print(f"  ✓ New best model saved (val AUC={best_val_auc:.4f})")
            wandb.save(best_ckpt)

    # ── final test evaluation ──
    print("\n=== Final Test Evaluation ===")
    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_loss, test_metrics = evaluate(
        model, test_loader, criterion, device, split="test"
    )
    print(f"  Test   loss={test_loss:.4f}  acc={test_metrics['accuracy']:.4f}"
          f"  f1={test_metrics['f1']:.4f}  auc={test_metrics['auc']:.4f}")

    wandb.log({
        "test/loss"     : test_loss,
        "test/accuracy" : test_metrics["accuracy"],
        "test/f1"       : test_metrics["f1"],
        "test/auc"      : test_metrics["auc"],
    })
    wandb.finish()
    print(f"\nDone. Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    train()