import os
import time
import snntorch as snn
import torch.nn as nn
from snntorch import surrogate
from spinnaker_connect_helpers import *
from IPython.display import clear_output
from sklearn.metrics import confusion_matrix, accuracy_score
from datetime import datetime

import matplotlib.pyplot as plt

fig = None
ax = None

fc_hidden = 512

device = torch.device("cuda")

# Snntorch parameters
beta = 0.92
num_steps = 35

# Spinnaker pyNN parameters
timestep = 1.0
delay = 1.0
tau_m = float(-timestep / np.log(beta))
v_rest = -65
v_reset =  -65
v_thresh = -61
tau_syn_E = 5.0
tau_syn_I = 3.0
tau_refrac = 1.0

w_scale_fc = 0.3
w_scale_out = 2.0

inh_weight = 1.3


class SignDataset_spinnaker:
    def __init__(self, root, H=28, W=28, max_t=None, transform=None):
        self.LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                        'U', 'V', 'W', 'X', 'Y']
        self.label_map = {'A': 0,
                          'B': 1,
                          'C': 2,
                          'D': 3,
                          'E': 4,
                          'F': 5,
                          'G': 6,
                          'H': 7,
                          'I': 8,
                          'K': 9,
                          'L': 10,
                          'M': 11,
                          'N': 12,
                          'O': 13,
                          'P': 14,
                          'Q': 15,
                          'R': 16,
                          'S': 17,
                          'T': 18,
                          'U': 19,
                          'V': 20,
                          'W': 21,
                          'X': 22,
                          'Y': 23}

        self.H = H
        self.W = W
        self.D = H * W
        self.max_t = max_t
        self.transform = transform

        self.samples = []
        for f in os.listdir(root):
            if not (f.endswith(".npy") or f.endswith(".npz")):
                continue
            label_char = f[0].upper()
            if label_char not in self.label_map:
                continue
            path = os.path.join(root, f)
            self.samples.append((path, self.label_map[label_char]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        if self.transform is not None:
            ev = np.load(path, allow_pickle=False).copy()
            ev["t"] = ev["t"] - ev["t"].min()
            ev = self.transform(ev)
            t_ms = np.floor((ev["t"] - ev["t"].min()) / 1000.0).astype(np.int64)
            x = np.clip(ev["x"].astype(np.int64), 0, self.W - 1)
            y = np.clip(ev["y"].astype(np.int64), 0, self.H - 1)
            nid = y * self.W + x

        else:
            if path.endswith(".npy"):
                t_ms, nid = load_spike_tuples(path)
            else:
                t_ms, nid = load_spike_tuples_npz(path)

            t_ms = (t_ms - t_ms.min()).astype(np.int64)

        if t_ms.size:
            T = int(self.max_t) if self.max_t is not None else int(t_ms.max()) + 1
        else:
            T = int(self.max_t) if self.max_t is not None else 0

        if T > 0:
            t_ms = np.clip(t_ms, 0, T - 1)

        m = (nid >= 0) & (nid < self.D)
        t_ms = t_ms[m]
        nid = nid[m]

        spike_times = [[] for _ in range(self.D)]
        for t, n in zip(t_ms, nid):
            spike_times[int(n)].append(float(t))
        return spike_times, int(label), int(T)


input_spatial_size = 48
test_root = r"D:\gesture-recognition-paper\data\ASL_DVS\test_5_ROI_spikes_35ms_mini\test_5_ROI_spikes_35ms_minimini"
test_dataset_spinn = SignDataset_spinnaker(test_root, H=input_spatial_size, W=input_spatial_size, transform = None)
val_loader = DataLoader(test_dataset_spinn, batch_size=1, shuffle=False, collate_fn=lambda batch: batch[0])
print(f"Val loader size: {len(val_loader)}")

spike_grad = surrogate.fast_sigmoid(slope=25)


class TinyConvSNN_24_FC(nn.Module):
    def __init__(self, H=48, W=48, n_classes=24, beta=0.95, num_steps=35, dropout_p=0.0, fc_hidden=512):
        super().__init__()
        self.H, self.W = H, W
        self.n_classes = n_classes
        self.num_steps = num_steps

        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, stride=6, padding=0, bias=False)
        self.fc1 = nn.Linear(256, fc_hidden, bias=False)
        self.lif_fc1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=False)
        self.dropout_fc1 = nn.Dropout(p=dropout_p)
        self.readout = nn.Linear(fc_hidden, n_classes, bias=False)

        # LIF neurons
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=False)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=False)

        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout_readout = nn.Dropout(p=dropout_p)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight, gain=1.0)

    def forward(self, x):
        T, B, _ = x.shape
        x = x.view(T, B, 1, self.H, self.W)

        mem1 = self.lif1.init_leaky().to(x.device)
        mem_fc1 = self.lif_fc1.init_leaky().to(x.device)
        memo = self.lif_out.init_leaky().to(x.device)

        spk_out_rec = []
        for t in range(T):
            xt = x[t]

            # Conv layer
            cur1 = self.conv1(xt)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.dropout1(spk1)

            feat = spk1.flatten(1)  # [B, 256]

            # Hidden FC layer
            cur_fc1 = self.fc1(feat)
            spk_fc1, mem_fc1 = self.lif_fc1(cur_fc1, mem_fc1)
            spk_fc1 = self.dropout_fc1(spk_fc1)

            # Readout layer
            spk_fc1_drop = self.dropout_readout(spk_fc1)
            cur_out = self.readout(spk_fc1_drop)
            spk_out, memo = self.lif_out(cur_out, memo)

            spk_out_rec.append(spk_out)

        return torch.stack(spk_out_rec, dim=0)

    @torch.no_grad()
    def predict(self, x):
        # x: [T,B,2304]
        spk = self.forward(x)
        counts = spk.sum(0)  # [B,26]
        return counts.argmax(dim=1)


model_snn_cpu = TinyConvSNN_24_FC(
    H=48, W=48,
    n_classes=24,
    beta=beta,
    num_steps=num_steps,
    dropout_p=0.0,
    fc_hidden=fc_hidden
)

# ----------------------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------------------
model_name = "tiny_conv_up_08_ASLDVS_finetune_cleaninit.pt"
checkpoint = torch.load(
    f"D:\\gesture-recognition-paper\\finetuning-spinnaker\\{model_name}", map_location=device)
model_snn_cpu.load_state_dict(checkpoint["model_state_dict"], strict=False)

# Copy pretrained weights
model_snn_cpu.conv1.weight.data = checkpoint['model_state_dict']['conv1.weight']
model_snn_cpu.fc1.weight.data = checkpoint['model_state_dict']['fc1.weight']
model_snn_cpu.readout.weight.data = checkpoint['model_state_dict']['readout.weight']

model_snn_cpu.eval()

W1 = model_snn_cpu.conv1.weight.detach().cpu().numpy()  # (4,1,5,5)
W2 = model_snn_cpu.fc1.weight.detach().cpu().numpy()  # (fc_hidden,256)
W3 = model_snn_cpu.readout.weight.detach().cpu().numpy()  # (24,fc_hidden)

class_names = test_dataset_spinn.LETTERS  # length 24

def plot_spike_counts(idx, spinn_counts, true_label=None, pred=None):
    global fig, ax
    spinn_counts = np.asarray(spinn_counts).ravel()

    assert spinn_counts.ndim == 1 and spinn_counts.size == 24

    clear_output(wait=True)

    x = np.arange(24)
    width = 0.40

    if fig is None:
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(111)
        plt.show(block=False)
    ax.clear()

    if true_label is not None:
        tl = int(true_label)
        plt.axvspan(tl - 0.5, tl + 0.5, color="gold", alpha=0.25, zorder=0)

    plt.bar(x + width/2, spinn_counts, width=width, label="SpiNNaker", zorder=2)

    plt.xticks(x, class_names, rotation=90)
    plt.ylabel("Spike count")

    title = f"Sample {idx}"
    if true_label is not None:
        tl = int(true_label)
        title += f" | true={tl}({class_names[tl]})"
    if pred is not None:
        pd = int(pred)
        title += f" | pred={pd}({class_names[pd]})"
    plt.title(title)

    plt.legend()
    plt.tight_layout()

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)

# ----------------------------------------------------------------------
# LAYERS: set up connection and neuron populations
# ----------------------------------------------------------------------
injector_label = "injector"
output_label = "output"
timer_label = "timer"

fc_labels = ["fc1_block_0", "fc1_block_1"]
receive_labels = list([output_label, timer_label, "conv1" ] + fc_labels)
connection = sim.external_devices.SpynnakerLiveSpikesConnection(
    local_port=None,
    send_labels=[injector_label],
    receive_labels=receive_labels
)

time.sleep(2)
sim.setup(timestep=1.0, time_scale_factor=20)

input_pop = sim.Population(
    48*48,
    sim.external_devices.SpikeInjector(
        database_notify_port_num=connection.local_port
    ),
    label=injector_label
)
conv_out_h = (48 - 5) // 6 + 1  # = 8
conv_out_w = (48 - 5) // 6 + 1  # = 8
conv1_size = 4 * conv_out_h * conv_out_w  # = 256

conv1 = sim.Population(
    conv1_size,
    sim.IF_curr_exp(
        v_rest=v_rest, v_reset=v_reset, v_thresh=v_thresh,
        tau_m=tau_m, tau_syn_E=tau_syn_E, tau_syn_I=tau_syn_I,tau_refrac = tau_refrac,
    ),
    label="conv1"
)

fc_block_sizes = split_sizes(fc_hidden, block=MAX_ROW)
print("fc_hidden:", fc_hidden, "blocks:", fc_block_sizes)

fc_pops = []
for bi, bsz in enumerate(fc_block_sizes):
    fc_pop = sim.Population(
        bsz,
        sim.IF_curr_exp(
            v_rest=v_rest, v_reset=v_reset, v_thresh=v_thresh,
            tau_m=tau_m, tau_syn_E=tau_syn_E, tau_syn_I=tau_syn_I, tau_refrac=tau_refrac,
        ),
        label=f"fc1_block_{bi}")
    fc_pop.record("spikes")
    fc_pops.append(fc_pop)

output_pop = sim.Population(
    24,
    sim.IF_curr_exp(
        v_rest=v_rest, v_reset=v_reset, v_thresh=v_thresh,
        tau_m=tau_m, tau_syn_E=tau_syn_E, tau_syn_I=tau_syn_I, tau_refrac = tau_refrac,
    ),
    label=output_label
)

conv1.record("spikes")
output_pop.record("spikes")


def print_layer_sizes(input_pop, conv1, fc_pops, output_pop):
    print("\n===== NEURON COUNTS =====")
    print(f"input_pop      : {input_pop.size}")
    print(f"conv1          : {conv1.size}")

    fc_sizes = [p.size for p in fc_pops]
    print(f"fc1 blocks     : {len(fc_pops)}")
    print(f"fc1 block sizes: {fc_sizes}")
    print(f"fc1 total      : {sum(fc_sizes)}")

    print(f"output_pop     : {output_pop.size}")
    print(f"timer_pop      : {timer_pop.size}")
    print(f"TOTAL neurons  : {input_pop.size + conv1.size + sum(fc_sizes) + output_pop.size + timer_pop.size}")
    print("=========================\n")

# ----------------------------------------------------------------------
# TIMER: continuous sim-time reference (independent of classifier output)
# ----------------------------------------------------------------------
timer_pop = sim.Population(
    1,
    sim.SpikeSourceArray(spike_times=[list(range(0, 100000))]),
    label=timer_label
)

sim.external_devices.activate_live_output_for(
    timer_pop,
    database_notify_port_num=connection.local_port
)

# ----------------------------------------------------------------------
# CONNECTIONS
# ----------------------------------------------------------------------
conv1_exc, conv1_inh = pytorch_conv_to_spinnaker_connections(
    conv_weight=W1,
    input_h=48, input_w=48,
    output_h=8, output_w=8,
    stride=6,
    padding=0,
    weight_scale=1.0,
    eps=1e-16,
    delay=1.0
)
sim.Projection(input_pop, conv1, sim.FromListConnector(conv1_exc), receptor_type="excitatory")
sim.Projection(input_pop, conv1, sim.FromListConnector(conv1_inh), receptor_type="inhibitory")

eps = 1e-16

fc_block_lists = []
post_offset = 0
for bi, fc_pop in enumerate(fc_pops):
    bsz = fc_pop.size
    W2_block = slice_fc_post_pre(W2, post_offset, bsz)
    exc, inh = weights_to_exc_inh_from_matrix_prune0(W2_block * w_scale_fc, fixed_delay=1.0)
    fc_block_lists.append((exc, inh))

    sim.Projection(conv1, fc_pop, sim.FromListConnector(exc), receptor_type="excitatory")
    sim.Projection(conv1, fc_pop, sim.FromListConnector(inh), receptor_type="inhibitory")
    post_offset += bsz

out_block_lists = []
pre_offset = 0
for bi, fc_pop in enumerate(fc_pops):
    bsz = fc_pop.size
    W3_block = slice_readout_post_pre(W3, pre_offset, bsz)
    exc, inh = weights_to_exc_inh_from_matrix_prune0(W3_block * w_scale_out, fixed_delay=1.0)
    out_block_lists.append((exc, inh))

    sim.Projection(fc_pop, output_pop, sim.FromListConnector(exc), receptor_type="excitatory")
    sim.Projection(fc_pop, output_pop, sim.FromListConnector(inh), receptor_type="inhibitory")
    pre_offset += bsz

delay = 1.0

inh_conns = []
for i in range(24):
    for j in range(24):
        if i != j:
            inh_conns.append((i, j, inh_weight, delay))

sim.Projection(
    output_pop,
    output_pop,
    sim.FromListConnector(inh_conns),
    receptor_type="inhibitory"
)

def count_synapses_from_lists(conv1_exc, conv1_inh, fc_block_lists, out_block_lists, inh_conns):
    """
    fc_block_lists: list of (exc_list, inh_list) for each fc block
    out_block_lists: list of (exc_list, inh_list) for each readout block
    Each *_list is the exact list passed into FromListConnector.
    """

    def n(lst): return 0 if lst is None else len(lst)

    conv_total = n(conv1_exc) + n(conv1_inh)

    fc_exc_total = sum(n(exc) for exc, inh in fc_block_lists)
    fc_inh_total = sum(n(inh) for exc, inh in fc_block_lists)
    fc_total = fc_exc_total + fc_inh_total

    out_exc_total = sum(n(exc) for exc, inh in out_block_lists)
    out_inh_total = sum(n(inh) for exc, inh in out_block_lists)
    out_total = out_exc_total + out_inh_total

    wta_total = n(inh_conns)

    grand_total = conv_total + fc_total + out_total + wta_total

    print("\n===== SYNAPSE COUNTS (FromListConnector) =====")
    print(f"conv  exc: {n(conv1_exc):>10} | inh: {n(conv1_inh):>10} | total: {conv_total:>10}")
    print(f"fc    exc: {fc_exc_total:>10} | inh: {fc_inh_total:>10} | total: {fc_total:>10}")
    print(f"out   exc: {out_exc_total:>10} | inh: {out_inh_total:>10} | total: {out_total:>10}")
    print(f"wta inhib: {wta_total:>10}")
    print(f"GRAND TOTAL synapses: {grand_total}")
    print("=============================================\n")

    return {
        "conv_exc": n(conv1_exc), "conv_inh": n(conv1_inh),
        "fc_exc": fc_exc_total, "fc_inh": fc_inh_total,
        "out_exc": out_exc_total, "out_inh": out_inh_total,
        "wta_inh": wta_total,
        "total": grand_total,
    }


syn_counts = count_synapses_from_lists(conv1_exc, conv1_inh, fc_block_lists, out_block_lists, inh_conns)

# ============================================================================
# LIVE INPUT/OUTPUT
# ============================================================================
sim.external_devices.activate_live_output_for(
    output_pop,
    database_notify_port_num=connection.local_port
)
sim.external_devices.activate_live_output_for(
    conv1,
    database_notify_port_num=connection.local_port
)

for bi, fc_pop in enumerate(fc_pops):
    sim.external_devices.activate_live_output_for(
        fc_pop,
        database_notify_port_num=connection.local_port
    )

sim_t_now = -1
sim_t_lock = threading.Lock()

def timer_callback(label, t, neuron_ids):
    # t is absolute simulation time in ms (integer ticks)
    global sim_t_now
    with sim_t_lock:
        sim_t_now = int(t)

def get_sim_t_now():
    with sim_t_lock:
        return int(sim_t_now)

def wait_until_sim_time(target_t, sleep_s=0.0005):
    # blocks until sim time >= target_t
    while True:
        now = get_sim_t_now()
        if now >= target_t:
            return now
        time.sleep(sleep_s)

def bucket_by_ms(spike_times_per_neuron, T_ms=None):
    buckets = {}
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

current_window = {"t0": 0, "t1": 0}
output_counts = np.zeros(24, dtype=np.int64)
out_lock = threading.Lock()

n_spikes_per_layer = {}
for label in receive_labels:
    n_spikes_per_layer[label] = 0

def live_output_callback(label, t, neuron_ids):
    tt = int(t)
    with out_lock:

        n_spikes_per_layer[label] += len(neuron_ids)

        if current_window["t0"] <= tt < current_window["t1"] and label == output_label:
            for nid in neuron_ids:
                if 0 <= nid < output_counts.size:
                    output_counts[nid] += 1

def send_sample(
    spike_times_per_neuron,
    T_ms,
    guard_ms=30,
    quiet_ms=50,
    lead_in_ms=20,
    chunk_size=128
):
    """
      1) chooses a future window [t0_abs, t1_abs)
      2) gates output counts to that window
      3) at each ms tick, sends bucketed spikes for that relative ms
    """
    T_ms = int(T_ms)
    guard_ms = int(guard_ms)
    quiet_ms = int(quiet_ms)
    lead_in_ms = int(lead_in_ms)

    now = get_sim_t_now()
    if now < 0:
        raise RuntimeError("sim_t_now not initialized; timer callback not firing")

    t0_abs = now + lead_in_ms
    t1_abs = t0_abs + T_ms

    buckets_rel = bucket_by_ms(spike_times_per_neuron, T_ms=T_ms)

    n_in = sum(len(v) for v in buckets_rel.values())
    max_bin = max((len(v) for v in buckets_rel.values()), default=0)
    print(f"[send_sample] input spikes (dedup): {n_in}, max spikes in a 1ms bin: {max_bin}")

    with out_lock:
        current_window["t0"] = t0_abs
        current_window["t1"] = t1_abs
        output_counts[:] = 0

    wait_until_sim_time(t0_abs)

    for rel_ms in range(T_ms):
        t_abs = t0_abs + rel_ms
        wait_until_sim_time(t_abs)

        ids = buckets_rel.get(rel_ms, [])
        if ids:
            for part in chunks(ids, chunk_size):
                connection.send_spikes(injector_label, part)

    wait_until_sim_time(t1_abs + guard_ms)

    with out_lock:
        counts = output_counts.copy()

    wait_until_sim_time(t1_abs + guard_ms + quiet_ms)

    return counts, (t0_abs, t1_abs)

spinnaker_ready = False

def start_callback(label, conn):
    global spinnaker_ready
    spinnaker_ready = True
    print("START CALLBACK FIRED")

connection.add_start_callback(injector_label, start_callback)

for label in receive_labels:
    connection.add_receive_callback(label, live_output_callback)

connection.add_receive_callback(timer_label, timer_callback)


# ============================================================================
# SIMULATION: run simulation on the background thread
# ============================================================================

sim_thread = (threading.Thread(target=lambda: sim.run(1000000), daemon=True))
sim_thread.start()

print("Waiting for start callback...")
while not spinnaker_ready:
    time.sleep(0.05)

while get_sim_t_now() < 0:
    time.sleep(0.05)

print("SpiNNaker ready! Starting evaluation...")

# ============================================================================
# EVALUATE ALL TEST SAMPLES
# ============================================================================
preds = []
labels = []

start_time = time.perf_counter()
for idx, (spike_times, label, T) in enumerate(val_loader):
    label_int = int(label.item()) if hasattr(label, "item") else int(label)
    T_ms = int(T.item()) if hasattr(T, "item") else int(T)

    if isinstance(spike_times, np.ndarray):
        spike_times = spike_times.tolist()
    spike_times = [list(map(float, st)) for st in spike_times]


    def max_bin_occupancy(spike_times_per_neuron, T_ms=None, dedup=True):
        counts = {}
        for nid, times in enumerate(spike_times_per_neuron):
            for t in times:
                tm = int(t)
                if T_ms is not None and (tm < 0 or tm >= T_ms):
                    continue
                if dedup:
                    counts[(tm, nid)] = 1
                else:
                    counts[tm] = counts.get(tm, 0) + 1

        if dedup:
            per_tm = {}
            for (tm, _nid) in counts.keys():
                per_tm[tm] = per_tm.get(tm, 0) + 1
        else:
            per_tm = counts

        if not per_tm:
            return 0, None, {}
        t_max = max(per_tm, key=per_tm.get)
        return per_tm[t_max], t_max, per_tm


    def count_duplicates_same_ms(spike_times_per_neuron, T_ms=None):
        dup = 0
        total = 0
        for nid, times in enumerate(spike_times_per_neuron):
            ms = [int(t) for t in times if (T_ms is None or (0 <= int(t) < T_ms))]
            total += len(ms)
            dup += (len(ms) - len(set(ms)))
        return total, dup


    total_sp, dup_sp = count_duplicates_same_ms(spike_times, T_ms=T_ms)
    print(f"[check] total spikes={total_sp}, duplicates_same_neuron_same_ms={dup_sp}")

    max_bin, t_at, per_tm = max_bin_occupancy(spike_times, T_ms=T_ms, dedup=True)
    print(f"[check] max unique neuron spikes in 1ms bin = {max_bin} at t={t_at} ms (T={T_ms})")

    counts, (t0_abs, t1_abs) = send_sample(
        spike_times_per_neuron=spike_times,
        T_ms=T_ms,
        guard_ms=30,
        quiet_ms=50,
        lead_in_ms=20,
        chunk_size=128
    )

    spinn_counts = counts.astype(np.int64)
    true_label = int(label) if not hasattr(label, "item") else int(label.item())
    pred = int(np.argmax(spinn_counts))
    total_spikes = int(counts.sum())

    preds.append(pred)
    labels.append(label_int)
    plot_spike_counts(idx, spinn_counts, true_label=true_label, pred=pred)
    print_live_acc(preds, labels, every=1, idx=idx, pred=pred, true=label_int)
    print(f"\nSample {idx + 1}/{len(test_dataset_spinn)}")
    print(f"Window [{t0_abs},{t1_abs}) ms | Pred: {pred} True: {label_int} Total output spikes: {total_spikes}")
    print_live_acc(preds, labels, every=1, idx=idx, pred=pred, true=label_int)
end_time = time.perf_counter()

# ============================================================================
# RESULTS
# ============================================================================
print(start_time, end_time, "total time:", end_time - start_time)
print(n_spikes_per_layer)

preds = np.array(preds, dtype=np.int64)
labels = np.array(labels, dtype=np.int64)

acc = accuracy_score(labels, preds)
print(f"\n{'=' * 60}")
print(f"FINAL RESULTS")
print(f"{'=' * 60}")
print(f"Overall accuracy: {acc:.3f} ({acc * 100:.1f}%)")
print(f"Correct: {(preds == labels).sum()}/{len(labels)}")

cm = confusion_matrix(labels, preds, labels=np.arange(24))
print(f"\nConfusion matrix:")
print(cm)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = r"D:\gesture-recognition-paper\finetuning-spinnaker\results"
cm_filename = os.path.join(output_dir, f"confusion_matrix_{timestamp}.npy")
np.save(cm_filename, cm)
print(f"\nConfusion matrix saved to: {cm_filename}")

results_dict = {
    'confusion_matrix': cm,
    'predictions': preds,
    'labels': labels,
    'accuracy': acc,
    'spike_counts': n_spikes_per_layer,
    'class_names': class_names
}

results_filename = os.path.join(output_dir, f"results_{timestamp}.npz")
np.savez(results_filename, **results_dict)
print(f"Full results saved to: {results_filename}")
sim.end()