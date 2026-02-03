import dv_processing as dv
from datetime import timedelta
import os
import numpy as np
import cv2 as cv
import tonic.transforms as T

def get_field(evs_np, candidates):
    for k in candidates:
        if k in evs_np.dtype.names:
            return k
    raise RuntimeError(f"Missing fields {candidates} in dtype names: {evs_np.dtype.names}")

def normalize_event_fields(evs_np):
    """
    Normalize dv_processing numpy field names to what your pipeline expects:
    returns a structured array with fields: ['timestamp','x','y','polarity'].
    """
    ts_k = get_field(evs_np, ("timestamp", "t", "ts"))
    x_k  = get_field(evs_np, ("x", "xs"))
    y_k  = get_field(evs_np, ("y", "ys"))
    p_k  = get_field(evs_np, ("polarity", "p", "pol"))

    out = np.empty(evs_np.shape[0],
                   dtype=[("timestamp", np.int64),
                          ("x", np.int16),
                          ("y", np.int16),
                          ("polarity", np.bool_)])
    out["timestamp"] = evs_np[ts_k].astype(np.int64, copy=False)
    out["x"]         = evs_np[x_k].astype(np.int16, copy=False)
    out["y"]         = evs_np[y_k].astype(np.int16, copy=False)
    out["polarity"]  = evs_np[p_k].astype(np.bool_, copy=False)
    return out


def adjust_timestamps(eventbins_per_frame):
    normalized_eventbins = []
    for frame_idx, frame_events in enumerate(eventbins_per_frame):
        normalized_frame = []
        for bin_idx, eventbin in enumerate(frame_events):
            if len(eventbin) > 0:  # eventbin is a list of chunks
                # Concatenate all chunks in this bin
                bin_events = np.concatenate(eventbin, axis=0)
                
                if len(bin_events) > 0:
                    # Find minimum timestamp in this bin
                    min_timestamp = bin_events["timestamp"].min()
                    
                    # Create normalized events
                    normalized_events = np.zeros(
                        len(bin_events),
                        dtype=[("x", "i2"), ("y", "i2"), ("timestamp", "i8"), ("polarity", "i1")]
                    )
                    normalized_events["x"] = bin_events["x"]
                    normalized_events["y"] = bin_events["y"]
                    normalized_events["timestamp"] = bin_events["timestamp"] - min_timestamp
                    normalized_events["polarity"] = bin_events["polarity"]
                    
                    # Store as a single-element list (to maintain structure)
                    normalized_frame.append([normalized_events])
                else:
                    normalized_frame.append([])
            else:
                normalized_frame.append([])
        normalized_eventbins.append(normalized_frame)
    return normalized_eventbins

def np_events_to_store(ev_np, timestamp_name = "timestamp", polarity_name = "polarity"):
    store = dv.EventStore()
    if ev_np.size == 0:
        return store

    ev_sorted = np.sort(ev_np, order=timestamp_name)    # sort by timestamp, ensure strictly increasing timestamps
    last_t = None
    for e in ev_sorted:
        t = int(e[timestamp_name])
        if last_t is not None and t <= last_t:
            continue  # drop duplicates / out-of-order
        store.push_back(
            t,
            int(e["x"]),
            int(e["y"]),
            bool(e[polarity_name]),
        )
        last_t = t

    return store

def roi_bounds(cx, cy, size, H, W):
    half = size // 2
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(W, cx + half)
    y1 = min(H, cy + half)
    return x0, y0, x1, y1

def draw_roi(img, x0, y0, x1, y1, color=(0,255,0)):
    if img.ndim == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    out = img.copy()
    cv.rectangle(out, (x0, y0), (x1, y1), color, 2)
    return out

def draw_cross(img, x, y, color=(0, 255, 0), size=12, thickness=2):
    if img.ndim == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    out = img.copy()
    cv.drawMarker(out, (int(x), int(y)), color,
                  markerType=cv.MARKER_CROSS,
                  markerSize=size, thickness=thickness)
    return out

def resize_h(img, H):
    h, w = img.shape[:2]
    if h == H:
        return img
    new_w = int(w * H / h)
    return cv.resize(img, (new_w, H), interpolation=cv.INTER_NEAREST)


def tonic_downsample_ROIevents(event_frame, cx,cy, ev_np, SIZE_ROI,downsample_size=28): # ROI extraction
    H, W = event_frame.shape[:2]
    x0, y0, x1, y1 = roi_bounds(cx, cy, SIZE_ROI, H, W)
    # crop out from events
    mask = ((ev_np["x"] >= x0) & (ev_np["x"] < x1) & (ev_np["y"] >= y0) & (ev_np["y"] < y1))
    roi_events_np = ev_np[mask].copy() 
    roi_events_np["x"] -= x0 
    roi_events_np["y"] -= y0

    W_roi = x1 - x0 
    H_roi = y1 - y0
    sx = float(downsample_size) / W_roi
    sy = float(downsample_size) / H_roi

    events_tonic = np.empty(roi_events_np.shape[0], dtype=[("t","i8"),("x","i2"),("y","i2"),("p","b")]) 
    events_tonic["t"] = roi_events_np["timestamp"]
    events_tonic["x"] = roi_events_np["x"]
    events_tonic["y"] = roi_events_np["y"]
    events_tonic["p"] = roi_events_np["polarity"].astype(bool)

    downsample_only = T.Downsample(spatial_factor=(sx, sy))
    events_28 = downsample_only(events_tonic)  # (t,x,y,p) downsampled events
    return events_28

def roi_to_spike_array(roi_events_down, roi_size=28):
    """
    Convert downscaled ROI events to spike array for SpiNNaker
    """
    if len(roi_events_down) == 0:
        return []

    # Extract fields 
    timestamps = roi_events_down['t']
    x_coords = roi_events_down['x']
    y_coords = roi_events_down['y']

    t_min = timestamps.min()
    timestamps_ms = ((timestamps - t_min) / 1000.0).astype(np.float64)

    # Map 2D coordinates to 1D neuron IDs using row-major ordering
    # neuron_id = y * roi_size + x
    neuron_ids = (y_coords * roi_size + x_coords).astype(np.int32)
    spikes = [(timestamps_ms[i], neuron_ids[i]) for i in range(len(roi_events_down))]

    # Sort by timestamp, maintain temporal order
    spikes.sort(key=lambda x: x[0])
    return spikes


def extract_intervals(rec, intervals_us):
    """
    Returns list of dv.EventStore for each [start_us, end_us).
    Works even if batches are not strictly time-sorted.
    """
    intervals_us = sorted([(int(s), int(e)) for s, e in intervals_us], key=lambda x: x[0])

    merged = []
    for s, e in intervals_us:
        if not merged or s >= merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    intervals_us = merged

    n = len(intervals_us)

    # collect per-interval arrays
    buf = [dict(x=[], y=[], t=[], p=[]) for _ in range(n)]

    i = 0
    while True:
        batch = rec.getNextEventBatch()
        if batch is None:
            break
        if batch.isEmpty():
            continue

        bt = batch.timestamps()

        # x/y
        if hasattr(batch, "coordinates"):
            xy = batch.coordinates()
            bx = xy[:, 0]
            by = xy[:, 1]
        else:
            bx = batch.xs()
            by = batch.ys()

        bp = batch.polarities() if hasattr(batch, "polarities") else None

        while i < n and bt[0] >= intervals_us[i][1]:
            i += 1
        if i >= n:
            break

        # If batch ends before current interval starts, skip
        if bt[-1] < intervals_us[i][0]:
            continue

        # batch may overlap multiple intervals
        j = i
        while j < n and intervals_us[j][0] <= bt[-1]:
            s, e = intervals_us[j]
            m = (bt >= s) & (bt < e)
            if np.any(m):
                buf[j]["x"].append(bx[m])
                buf[j]["y"].append(by[m])
                buf[j]["t"].append(bt[m])
                if bp is not None:
                    buf[j]["p"].append(bp[m])
            j += 1

    # build EventStores 
    stores = []
    for j in range(n):
        if len(buf[j]["t"]) == 0:
            stores.append(dv.EventStore())
            continue

        x = np.concatenate(buf[j]["x"]).astype(np.int32, copy=False)
        y = np.concatenate(buf[j]["y"]).astype(np.int32, copy=False)
        t = np.concatenate(buf[j]["t"]).astype(np.int64, copy=False)

        if len(buf[j]["p"]) > 0:
            p = np.concatenate(buf[j]["p"]).astype(np.bool_, copy=False)
        else:
            p = np.ones_like(t, dtype=np.bool_)

        order = np.argsort(t, kind="stable")
        x, y, t, p = x[order], y[order], t[order], p[order]

        es = dv.EventStore()
        for xi, yi, ti, pi in zip(x, y, t, p):
            es.push_back(int(ti), int(xi), int(yi), bool(pi))
        stores.append(es)

    return stores


def build_event_histogram(rec, bin_width_ms = 1):
    timestamps_us = []
    total_events = 0
    n_batches = 0

    for _ in range(1_000_000):
        batch = rec.getNextEventBatch()

        if batch is None:
            if n_batches > 0:
                break
            continue
        n_batches += 1

        if batch.isEmpty():
            continue

        ts = batch.timestamps()
        timestamps_us.append(ts)
        total_events += ts.size

    if total_events == 0:
        raise RuntimeError(
            "No events were read. Either the file has no event stream, "
            "or this dv-processing build uses a different reader API for AEDAT4."
        )

    ts_us = np.concatenate(timestamps_us).astype(np.float64)
    ts_ms = (ts_us - ts_us[0]) * 1e-3

    bins = np.arange(0, ts_ms[-1] + bin_width_ms, bin_width_ms)
    counts, _ = np.histogram(ts_ms, bins=bins)
    return bins, counts, ts_ms

def select_eventful_intervals(bins, counts, thr = 1000, target_ms = 35.0, dt_ms = 1):
    W = int(round(target_ms / dt_ms))   # 50

    t0 = bins[:-1]  # left edge of each 0.1 ms bin, shape (N,)
    c = counts.astype(np.int32)

    # --- sliding sum over 5 ms windows ---
    cs = np.concatenate([[0], np.cumsum(c)])
    win_sum = cs[W:] - cs[:-W]  # shape (N-W+1,)
    t_win0 = t0[:len(win_sum)]  # window start times (ms)


    # produce non-overlapping 5 ms crops (greedy by peak activity) ---
    min_gap_ms = 10.0  # set >0 if you want spacing between crops
    min_gap = int(round(min_gap_ms / dt_ms))

    mask = win_sum > thr
    cand_idx = np.where(mask)[0]

    cand_idx = cand_idx[np.argsort(win_sum[cand_idx])[::-1]]

    selected = []
    occupied = np.zeros_like(mask, dtype=bool)

    for j in cand_idx:
        # window covers [j, j+W)
        lo = max(0, j - W - min_gap)
        hi = min(len(mask), j + W + min_gap)
        if occupied[lo:hi].any():
            continue
        selected.append((t_win0[j], t_win0[j] + target_ms, int(win_sum[j])))
        occupied[j:j+W] = True

    # sort selected by time
    selected.sort(key=lambda x: x[0])
    return selected


def save_eventstore_npz(ev, out_path):
    ts = ev.timestamps()
    if hasattr(ev, "coordinates"):
        xy = ev.coordinates()
        xs = xy[:, 0]
        ys = xy[:, 1]
    else:
        xs = ev.xs()
        ys = ev.ys()
    ps = ev.polarities() if hasattr(ev, "polarities") else np.ones_like(ts, dtype=np.bool_)

    np.savez_compressed(out_path, x=xs, y=ys, t=ts, p=ps)

import bisect

def closest_frame_with_offset(frames, frame_ts, t_us):
    i = bisect.bisect_left(frame_ts, t_us)
    if i <= 0:
        t, img = frames[0]
        return t, img, abs(t - t_us)
    if i >= len(frames):
        t, img = frames[-1]
        return t, img, abs(t - t_us)

    t0, img0 = frames[i-1]
    t1, img1 = frames[i]
    if (t_us - t0) <= (t1 - t_us):
        return t0, img0, (t_us - t0)
    else:
        return t1, img1, (t1 - t_us)
    
def eventstore_to_count_image(ev, H, W):
    img = np.zeros((H, W), dtype=np.uint16)  # uint32 if very dense
    xy = ev.coordinates()                    # (N, 2) [x, y]
    xs = xy[:, 0].astype(np.int64, copy=False)
    ys = xy[:, 1].astype(np.int64, copy=False)

    # accumulate counts
    np.add.at(img, (ys, xs), 1)
    return img

def eventstore_to_structured(ev, t0_us=None):
    """
    Convert dv.EventStore -> numpy structured array with fields:
    ('timestamp','x','y','polarity').
    """
    t = ev.timestamps().astype(np.int64, copy=False)
    xy = ev.coordinates()
    x = xy[:, 0].astype(np.int16, copy=False)
    y = xy[:, 1].astype(np.int16, copy=False)

    if hasattr(ev, "polarities"):
        p = ev.polarities().astype(np.uint8, copy=False)
    else:
        p = np.ones_like(t, dtype=np.uint8)

    if t0_us is not None:
        t = t - np.int64(t0_us)
        if t.size and t.min() < 0:
            t = t - t.min()

    ev_np = np.empty(t.shape[0], dtype=[("timestamp", "<i8"),
                                       ("x", "<i2"),
                                       ("y", "<i2"),
                                       ("polarity", "u1")])
    ev_np["timestamp"] = t
    ev_np["x"] = x
    ev_np["y"] = y
    ev_np["polarity"] = p

    # Ensure time-sorted 
    ev_np.sort(order="timestamp")
    return ev_np