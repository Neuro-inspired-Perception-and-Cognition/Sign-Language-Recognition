import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,  WeightedRandomSampler

import os
import numpy as np
import matplotlib.pyplot as plt

class EarlyStopping:
    """
    Early stopping + best-checkpoint saving.
    """
    def __init__(self, monitor="val_acc", patience=10, min_delta=0.0,
                 ckpt_path="best_model.pt", verbose=True):
        assert monitor in ("val_acc", "val_loss")
        self.monitor = monitor
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.ckpt_path = ckpt_path
        self.verbose = verbose

        self.best = None
        self.best_epoch = -1
        self.num_bad_epochs = 0

    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.monitor == "val_acc":
            return value > (self.best + self.min_delta)
        else:  # val_loss
            return value < (self.best - self.min_delta)

    def step(self, value: float, epoch: int, model, optimizer=None, scheduler=None, extra: dict | None = None) -> bool:
        """
        Returns True if training should stop.
        """
        value = float(value)
        if self._is_improvement(value):
            self.best = value
            self.best_epoch = epoch
            self.num_bad_epochs = 0

            ckpt = {
                "epoch": epoch,
                "monitor": self.monitor,
                "best_value": self.best,
                "model_state_dict": model.state_dict(),
            }
            if optimizer is not None:
                ckpt["optimizer_state_dict"] = optimizer.state_dict()
            if scheduler is not None:
                ckpt["scheduler_state_dict"] = scheduler.state_dict()
            if extra is not None:
                ckpt["extra"] = extra

            os.makedirs(os.path.dirname(self.ckpt_path) or ".", exist_ok=True)
            torch.save(ckpt, self.ckpt_path)

            if self.verbose:
                print(f"[EarlyStopping] New best {self.monitor}={self.best:.6f} at epoch {epoch}. Saved -> {self.ckpt_path}")
            return False

        self.num_bad_epochs += 1
        if self.verbose:
            print(f"[EarlyStopping] No improvement in {self.monitor}. bad_epochs={self.num_bad_epochs}/{self.patience}")

        return self.num_bad_epochs >= self.patience

    def restore_best(self, model, device=None):
        ckpt = torch.load(self.ckpt_path, map_location=device if device is not None else "cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        return ckpt

class EnhancedFocalLoss(nn.Module):
    """
    Combines:
    - Focal Loss 
    - Class weights 
    - Label smoothing 
    - Online hard example mining 
    """
    def __init__(self, gamma=2.5, alpha=None, label_smoothing=0.1, 
                 online_mining=True, mining_threshold=0.7):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Class weights
        self.label_smoothing = label_smoothing
        self.online_mining = online_mining
        self.mining_threshold = mining_threshold
        
    def forward(self, inputs, targets):
        n_classes = inputs.size(-1)
        
        # Label smoothing
        targets_one_hot = F.one_hot(targets, n_classes).float()
        if self.label_smoothing > 0:
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + \
                            self.label_smoothing / n_classes
        
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        
        # Focal weight
        focal_weight = (1 - probs) ** self.gamma
        
        # Base loss
        loss = -(targets_one_hot * focal_weight * log_probs).sum(dim=-1)
        
        # Class weights
        if self.alpha is not None:
            alpha_weight = self.alpha[targets]
            loss = alpha_weight * loss
        
        # Online hard example mining
        if self.online_mining:
            true_class_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            
            hard_weight = torch.where(
                true_class_probs < self.mining_threshold,
                torch.ones_like(true_class_probs) * 2.0, 
                torch.ones_like(true_class_probs)
            )
            loss = loss * hard_weight
        
        return loss.mean()
    
def analyze_class_distribution(dataset, n_classes=24):
    """Get detailed class statistics"""
    labels = [label for _, label in dataset]
    class_counts = np.bincount(labels, minlength=n_classes)
    
    print("\nClass distribution:")
    class_names = ['A','B','C','D','E','F','G','H','I','K','L','M',
                   'N','O','P','Q','R','S','T','U','V','W','X','Y']
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        print(f"{name}: {count:4d} ({100*count/len(labels):5.2f}%)")
    
    # Calculate balanced weights
    total = len(labels)
    weights = torch.FloatTensor([total / (n_classes * max(c, 1)) for c in class_counts])
    
    return weights, class_counts

def create_balanced_dataloader(dataset, batch_size=16, num_workers=0):
    """DataLoader with class balancing"""
    labels = [label for _, label in dataset]
    class_sample_counts = np.bincount(labels)
    
    # Weight for each sample (inverse frequency)
    weights = [1.0 / class_sample_counts[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  
    )


def plot_cm_inline(cm, class_names, normalize=True, title="Confusion matrix"):
    cm_np = cm.numpy().astype(np.float64)
    if normalize:
        row_sums = cm_np.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_np = cm_np / row_sums

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_np, aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    fig.tight_layout()
    return fig

@torch.no_grad()
def confusion_matrix_snn(model, loader, device, n_letters):
    """
    Confusion matrix for SNN model that returns spk_out with shape [T,B,C].
    """
    model.eval()
    cm = torch.zeros((n_letters, n_letters), dtype=torch.int64)

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        if labels.ndim > 1:
            labels = labels.argmax(dim=1)
        labels = labels.long()

        spk_out = model(imgs)          # [T,B,C]
        spk_sum = spk_out.sum(0)       # [B,C]
        logits = spk_sum[:, :n_letters]
        pred = logits.argmax(dim=1)    # 0..n_letters-1

        # vectorized accumulate
        idx = labels.cpu() * n_letters + pred.cpu()
        cm.view(-1).index_add_(0, idx, torch.ones_like(idx, dtype=torch.int64))

    return cm


def load_spike_tuples(path):
    spikes = np.load(path, allow_pickle=True)
    # spikes is like array([(t,nid), ...], dtype=object) or list
    t = np.array([s[0] for s in spikes], dtype=np.int64)
    nid = np.array([s[1] for s in spikes], dtype=np.int64)
    return t, nid

def load_spike_tuples_npz(path):
    """
    Load spike tuples from .npz produced by np.savez_compressed(out_file, spikes)
    """
    data = np.load(path, allow_pickle=True)
    spikes = data["arr_0"]

    # spikes: array([(t_ms, neuron_id), ...], dtype=object)
    t = np.array([s[0] for s in spikes], dtype=np.float64)
    nid = np.array([s[1] for s in spikes], dtype=np.int64)

    return t, nid

def spikes_to_tonic_events(t, nid, H, W):
    """
    Convert (t, nid) spikes into tonic structured event array.
    """
    y = nid // W
    x = nid % W
    p = np.ones_like(t, dtype=np.int8)

    ev = np.zeros(len(t), dtype=[
        ('t', np.int64),
        ('x', np.int16),
        ('y', np.int16),
        ('p', np.int8),
    ])
    ev['t'] = t.astype(np.int64)
    ev['x'] = x.astype(np.int16)
    ev['y'] = y.astype(np.int16)
    ev['p'] = p
    return ev

def tonic_events_to_spikes(ev, H, W):
    t = ev['t']
    nid = ev['y'] * W + ev['x']
    return t, nid

def roi_to_spike_array(roi_events_down, roi_size=48):
    """
    Convert downscaled ROI events to spike array for SpiNNaker
    Returns:
        List of tuples (timestamp_ms, neuron_id)
    """
    if len(roi_events_down) == 0:
        return []

    timestamps = roi_events_down['t']
    x_coords = roi_events_down['x']
    y_coords = roi_events_down['y']

    t_min = timestamps.min()
    timestamps_ms = ((timestamps - t_min) / 1000.0).astype(np.float64)

    # Map coordinates to 1D neuron IDs 
    # neuron_id = y * roi_size + x
    neuron_ids = (y_coords * roi_size + x_coords).astype(np.int32)
    spikes = [(timestamps_ms[i], neuron_ids[i]) for i in range(len(roi_events_down))]

    spikes.sort(key=lambda x: x[0])
    return spikes

def collate_timefirst(batch):
    """ 
    Handle variable-length spike trains.
    Pads all samples in the batch to the length of the longest sample.
    """
    xs, ys = zip(*batch)  
    
    # Find the max number of timesteps 
    max_T = max(x.shape[0] for x in xs)
    
    # Pad each spike train 
    xs_padded = []
    for x in xs:
        T_current = x.shape[0]
        if T_current < max_T:
            padding = torch.zeros(max_T - T_current, x.shape[1], dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=0)  
        else:
            x_padded = x
        xs_padded.append(x_padded)
    
    xs = torch.stack(xs_padded, dim=0)
    xs = xs.permute(1, 0, 2).contiguous() 
    ys = torch.tensor(ys, dtype=torch.long)
    
    return xs, ys