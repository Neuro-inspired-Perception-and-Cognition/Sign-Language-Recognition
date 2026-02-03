""" 
Open an .aedat4 file, access the events, apply the sVA attention, select a square ROI of a given size SIZE_ROI around 
the most salient point. Convert the ROI events into spikes and save in the input format for the fingerspelling recognition SNN.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import dv_processing as dv
from dv_processing import visualization as dv_vis

import cv2 as cv
import cv2
import numpy as np
import string
import tonic.transforms as T
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm

import torch
torch.set_num_threads(1) 
torch.set_num_interop_threads(1)

from prepare_data_functions import *
from attention import *

#___________________________________________________________________________________

filepath = Path("data/path_to_the_file.aedat4") # source DVS recording .aedat4

SIZE_ROI = 250 # size of region of interest to be cropped out
DOWNSAMPLED_RES = 48 # size after downsampling

# directories to save the 
OUT_DIR = f"data/ROI_events" 
OUT_DIR_SPIKES = f"data/ROI_spikes"
OUT_FILENAME = "file1" # will be saved as file1.npy
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_DIR_SPIKES, exist_ok=True)

# length of time windows (samples) from aedat4 video
WINDOW_US = 35000 # microsecs (35 ms)    

# How many time windows (samples) will be selected per .aedat4 file
N_SAMPLES = 200

VISUALIZE = True # True for realtime event chunk + saliency map + downsampled ROI visualisation

#___________________________________________________________________________________

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = Config()
net_attention = initialise_attention(device, config.ATTENTION_PARAMS)


# ==== pick non-overlapping windows of specified length ====
rec = dv.io.MonoCameraRecording(str(filepath))
resolution = rec.getEventResolution()
slicer = dv.EventStreamSlicer()

state = {
    "windows": [],
    "win_idx": 0,
    "done": False,
}

def on_window(ev: dv.EventStore):
    if state["done"]:
        return

    n = int(ev.size())
    ev_np = normalize_event_fields(ev.numpy())

    state["windows"].append((n, state["win_idx"], ev_np))
    state["win_idx"] += 1

    if state["win_idx"] >= N_SAMPLES:
        state["done"] = True

slicer.doEveryTimeInterval(timedelta(microseconds=WINDOW_US), on_window)

while rec.isRunning() and not state["done"]:
    batch = rec.getNextEventBatch()
    if batch is None:
        break
    if batch.size() == 0:
        continue
    slicer.accept(batch)

windows = state["windows"]
print(f"Collected {len(windows)} windows.")


# ==== for each kept window -> attention -> ROI -> downsample -> save ==== 

vis_full = dv_vis.EventVisualizer(resolution)
vis_full.setBackgroundColor(dv_vis.colors.black())
vis_full.setPositiveColor(dv_vis.colors.white())
vis_full.setNegativeColor(dv_vis.colors.white())

if VISUALIZE:
    cv.namedWindow("Sample", cv.WINDOW_NORMAL)

vis_ds = dv.visualization.EventVisualizer((DOWNSAMPLED_RES, DOWNSAMPLED_RES))
vis_ds.setBackgroundColor(dv.visualization.colors.black())
vis_ds.setPositiveColor(dv.visualization.colors.white())
vis_ds.setNegativeColor(dv.visualization.colors.white())

for rank, (n_events, idx, ev_np) in enumerate(
        tqdm(windows, total=len(windows), desc="Processing windows")):
    # 1) window events -> EventStore
    events_store = np_events_to_store(
        ev_np,
        timestamp_name="timestamp",
        polarity_name="polarity"
    )

    # 2) event image for attention input
    ev_img = vis_full.generateImage(events_store)  # RGB
    ev_gray = cv2.cvtColor(ev_img, cv2.COLOR_RGB2GRAY)

    # 3) run attention -> (cx, cy) of most salient point
    frame_u8 = np.ascontiguousarray(ev_gray, dtype=np.uint8)
    frame_tensor = torch.tensor(frame_u8, dtype=torch.float32).unsqueeze(0).contiguous()

    with torch.no_grad():
        salmap, salmax_coords = run_attention(
            frame_tensor, net_attention, device,
            resolution, resolution,
            config.ATTENTION_PARAMS["num_pyr"]
        )
    cy, cx = int(salmax_coords[0]), int(salmax_coords[1])
    att_viz = visualize_saliency(ev_gray, salmap, salmax_coords)

    ev_vis = draw_cross(ev_gray, cx, cy)
    att_vis = draw_cross(att_viz, cx, cy)

    # 4) ROI crop + downsample events to 48x48 
    roi_size = SIZE_ROI
    events_down = tonic_downsample_ROIevents(
        ev_gray, cx, cy, ev_np, roi_size,
        downsample_size=DOWNSAMPLED_RES
    )
    store_ds = np_events_to_store(events_down, timestamp_name="t", polarity_name="p")
    
    if VISUALIZE:
        ds_img = vis_ds.generateImage(store_ds)                    # RGB small
        ds_bgr = cv.cvtColor(ds_img, cv.COLOR_RGB2BGR)

        # ---- compose side-by-side view ----
        ev_bgr  = cv.cvtColor(ev_vis,  cv.COLOR_GRAY2BGR) if ev_vis.ndim == 2 else ev_vis
        att_bgr = cv.cvtColor(att_vis, cv.COLOR_GRAY2BGR) if att_vis.ndim == 2 else att_vis

        H0, W0 = ev_bgr.shape[:2]
        att_bgr = cv.resize(att_bgr, (W0, H0), interpolation=cv.INTER_NEAREST)

        ds_big = cv.resize(ds_bgr, (W0, H0), interpolation=cv.INTER_NEAREST)

        combined = np.hstack([ev_bgr, att_bgr, ds_big])

        cv.imshow("Sample", combined)
    
        key = cv.waitKey(30) & 0xFF   # 30 ms per window; increase if still too fast
        if key in (27, ord("q")):     # ESC or q
            break
        elif key == ord(" "):         # space = pause
            while True:
                k2 = cv.waitKey(0) & 0xFF
                if k2 in (ord(" "), ord("s"), 27, ord("q")):
                    key = k2
                    break
            if key in (27, ord("q")):
                break
                
    # 5) save downsampled ROI events
    np.save(str(OUT_DIR + f"/{OUT_FILENAME}.npy"), events_down)

    # transform downsampled input to tuple structure
    spikes = roi_to_spike_array(events_down, roi_size=DOWNSAMPLED_RES)
    np.save(str(OUT_DIR + f"/{OUT_FILENAME}.npy"), spikes)
    
print("Done.")

