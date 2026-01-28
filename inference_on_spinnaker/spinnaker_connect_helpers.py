import numpy as np
import pyNN.spiNNaker as sim
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import DataLoader
import queue
import threading
import time
import string
import torch

def weights_to_exc_inh_from_matrix_prune0(W, fixed_delay=1.0, threshold=1e-4):
    """
    Add threshold to skip near-zero weights
    """
    W_exc = W.clip(min=0)
    W_inh = (-W).clip(min=0)
    
    exc_conns = []
    inh_conns = []
    
    npost, npre = W.shape
    for i in range(npost):
        for j in range(npre):
            w_e = W_exc[i, j]
            w_i = W_inh[i, j]
            
            # Skip connections below threshold
            if w_e > threshold:
                exc_conns.append((j, i, float(w_e), fixed_delay))
            if w_i > threshold:
                inh_conns.append((j, i, float(w_i), fixed_delay))
    
    return exc_conns, inh_conns

def weights_to_exc_inh_from_matrix(W_post_pre, fixed_delay=1.0):
    """
    W_post_pre: numpy [N_post, N_pre] with signed weights.
    """
    exc = []
    inh = []
    N_post, N_pre = W_post_pre.shape
    for post in range(N_post):
        row = W_post_pre[post]
        # iterate nonzeros only (optional); for dense weights, this is still fine but slower
        for pre in range(N_pre):
            w = float(row[pre])
            if w == 0.0:
                continue
            if w >= 0:
                exc.append((pre, post, w, fixed_delay))
            else:
                inh.append((pre, post, abs(w), fixed_delay))
    return exc, inh

def pytorch_conv_to_spinnaker_connections(
    conv_weight, input_h, input_w, output_h, output_w,
    stride, padding=0, weight_scale=1.0, eps=0.0, delay=1.0
):
    out_ch, in_ch, kH, kW = conv_weight.shape
    assert in_ch == 1

    exc, inh = [], []
    for oc in range(out_ch):
        for oy in range(output_h):
            for ox in range(output_w):
                post = oc * (output_h * output_w) + oy * output_w + ox

                base_y = oy * stride - padding
                base_x = ox * stride - padding

                for ky in range(kH):
                    iy = base_y + ky
                    if iy < 0 or iy >= input_h:
                        continue
                    for kx in range(kW):
                        ix = base_x + kx
                        if ix < 0 or ix >= input_w:
                            continue

                        pre = iy * input_w + ix
                        w = float(conv_weight[oc, 0, ky, kx]) * weight_scale
                        if abs(w) <= eps:
                            continue
                        if w >= 0:
                            exc.append((pre, post, w, delay))
                        else:
                            inh.append((pre, post, -w, delay))
    return exc, inh

def running_accuracy(preds_list, labels_list):
    n = len(labels_list)
    if n == 0:
        return 0.0, 0, 0
    correct = sum(int(p == y) for p, y in zip(preds_list, labels_list))
    return correct / n, correct, n

def print_live_acc(preds_list, labels_list, every=1, idx=None, pred=None, true=None):
    # idx is 0-based sample index
    n = len(labels_list)
    if n == 0:
        return
    if every is None or every <= 0:
        every = 1
    if (n % every) != 0:
        return
    acc, correct, total = running_accuracy(preds_list, labels_list)
    if idx is None:
        print(f"[live] acc={acc:.3f} ({correct}/{total})")
    else:
        print(f"[live] after {idx+1} samples: acc={acc:.3f} ({correct}/{total}) | last pred={pred} true={true}")

def create_conv_connections_with_weights(input_h, input_w, output_h, output_w, 
                                         kernel_size, stride, n_output_channels,
                                         trained_weights=None, weight_scale=1.0):
    """
    Create convolutional connectivity pattern.
    """
    connections_exc = []
    connections_inh = []
    
    offset = kernel_size // 2
    
    for out_chan in range(n_output_channels):
        for out_y in range(output_h):
            for out_x in range(output_w):
                post_idx = out_chan * (output_h * output_w) + out_y * output_w + out_x
                
                center_y = out_y * stride
                center_x = out_x * stride
                
                for ky in range(kernel_size):
                    for kx in range(kernel_size):
                        inp_y = center_y + ky - offset
                        inp_x = center_x + kx - offset
                        
                        if 0 <= inp_y < input_h and 0 <= inp_x < input_w:
                            pre_idx = inp_y * input_w + inp_x
                            
                            # Get weight from trained model or random
                            if trained_weights is not None and post_idx < trained_weights.shape[0]:
                                # Extract weight for this connection
                                kernel_idx = ky * kernel_size + kx
                                if pre_idx < trained_weights.shape[1]:
                                    w = trained_weights[post_idx, pre_idx] * weight_scale
                                else:
                                    w = np.random.randn() * 0.01
                            else:
                                w = np.random.randn() * 0.01
                            
                            if w >= 0:
                                connections_exc.append((pre_idx, post_idx, abs(w), 1.0))
                            else:
                                connections_inh.append((pre_idx, post_idx, abs(w), 1.0))
    
    return connections_exc, connections_inh

MAX_ROW = 256  # SpiNNaker row limit

def split_sizes(total, block=MAX_ROW):
    sizes = []
    left = int(total)
    while left > 0:
        sizes.append(min(block, left))
        left -= sizes[-1]
    return sizes

def slice_fc_post_pre(W_post_pre, start, size):
    # W_post_pre: [N_post, N_pre]
    return W_post_pre[start:start+size, :]

def slice_readout_post_pre(W_post_pre, start, size):
    # readout W_post_pre: [N_out, N_hidden]
    return W_post_pre[:, start:start+size]


def bucket_by_ms(spike_times_per_neuron, T_ms=None):
    # spike_times_per_neuron: list-of-lists (ms float/int), assumed already rebased to 0
    buckets = {}  # rel_ms -> list[int nid]
    for nid, times in enumerate(spike_times_per_neuron):
        for t in times:
            tm = int(t)
            if T_ms is not None and (tm < 0 or tm >= T_ms):
                continue
            buckets.setdefault(tm, []).append(nid)

    for tm in list(buckets.keys()):
        buckets[tm] = list(set(buckets[tm]))
    return buckets

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]



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