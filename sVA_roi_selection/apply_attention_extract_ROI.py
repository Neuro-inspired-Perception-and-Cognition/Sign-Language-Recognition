import dv_processing as dv
import cv2 as cv
from datetime import timedelta
import os
import numpy as np
from prepare_data_functions import *

from attention import *
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import tonic
import tonic.transforms as T

import string
import time

import torch
torch.set_num_threads(1) 
torch.set_num_interop_threads(1)
#___________________________________________________________________________________

letters = list(string.ascii_uppercase)
LETTER = "F"
PERSON = 5




FILENAME = f"data/raw_dataset_recordings_adjusted/{LETTER}_person{PERSON}_take000.aedat4" # source DVS recording .aedat4

HAND_SELECTION_ON = True
SIZE_ROI = [150,200] #[100,120] 
NUM_BINS = 4 #approximately 505 us time slots
DOWNSAMPLED_RES = 48
SAVE_STRIDE = 1 # save only each {SAVE_STRIDE}th timebin
# aiming for 5000 samples per letter per signer
OUT_DIR = f"data/dataset_by_signers_5ms/signer_{PERSON}" 
OUT_DIR_SPIKES = f"data/dataset_by_signers_5ms/signer_{PERSON}_spikes" 

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_DIR_SPIKES, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = Config()
net_attention = initialise_attention(device, config.ATTENTION_PARAMS)

#___________________________________________________________________________________
# Open recording 
rec = dv.io.MonoCameraRecording(f"{FILENAME}")
if not rec.isEventStreamAvailable():
    raise RuntimeError("Recording does not contain an event stream.")
if not rec.isFrameStreamAvailable():
    print("Warning: recording has no APS frames.")
resolution = rec.getEventResolution()

#___________________________________________________________________________________
# PASS 1: read GRAYSCALE frames and their timestamps (the whole video) 
frames = []  # list of (timestamp, image)
while rec.isRunning():
    f = rec.getNextFrame()
    if f is None:
        break                   
    frames.append((f.timestamp, f.image.copy()))
if not frames:
    raise RuntimeError("No frames read from recording.")

print(f"Loaded {len(frames)} frames.")

#___________________________________________________________________________________
# PASS 2: read EVENTS and bucket them by frame intervals
rec = dv.io.MonoCameraRecording(f"{FILENAME}") # reopen the recording to start events from the beginning

# each RBG frame has multiple event segments, each described by list of event chunks
eventbins_per_frame = [[[] for _ in range(NUM_BINS)] for _ in frames] # Each RGB frame has 66 event bins, each bin is a list of event chunks


frame_idx = 0
cur_frame_start = frames[0][0]
cur_frame_end = frames[1][0] if len(frames) > 1 else np.iinfo(np.int64).max

# Calculate bin boundaries for current frame
interval_duration = cur_frame_end - cur_frame_start
bin_duration = interval_duration / NUM_BINS
bin_boundaries = [cur_frame_start + i * bin_duration for i in range(NUM_BINS + 1)]


while rec.isRunning() and frame_idx < len(frames):
    batch = rec.getNextEventBatch()
    if batch is None or batch.isEmpty():
        continue

    evs = batch.numpy()            # structured array, fields like ['t', 'x', 'y', 'p']
    ts = (evs["timestamp"])                  # adjust key if your dv version uses a different name

    start_idx = 0     # events in a batch are time-sorted
    while start_idx < len(evs) and frame_idx < len(frames):
    
        if ts[start_idx] >= cur_frame_end:  # Check if we've moved past the current frame
            frame_idx += 1  # Move to next frame
            if frame_idx >= len(frames):
                break
            
            # Update frame boundaries
            cur_frame_start = frames[frame_idx][0]
            cur_frame_end = frames[frame_idx + 1][0] if frame_idx + 1 < len(frames) else np.iinfo(np.int64).max
            
            # Recalculate bin boundaries for new frame
            interval_duration = cur_frame_end - cur_frame_start
            bin_duration = interval_duration / NUM_BINS
            bin_boundaries = [cur_frame_start + i * bin_duration for i in range(NUM_BINS + 1)]
            continue
        
        # Find which events belong to current frame
        frame_end_idx = np.searchsorted(ts[start_idx:], cur_frame_end, side="left") + start_idx
        frame_events = evs[start_idx:frame_end_idx] # Get events for current frame
        
        if len(frame_events) > 0:
            # Distribute events into NUM_BINS bins
            frame_ts = frame_events["timestamp"]
            
            for bin_idx in range(NUM_BINS):
                bin_start = bin_boundaries[bin_idx]
                bin_end = bin_boundaries[bin_idx + 1]
                
                # Find events in this bin
                bin_mask = (frame_ts >= bin_start) & (frame_ts < bin_end)
                bin_events = frame_events[bin_mask]
                
                if len(bin_events) > 0:
                    eventbins_per_frame[frame_idx][bin_idx].append(bin_events)
        
        # Move to next batch segment
        start_idx = frame_end_idx
        # If we've processed all events in this batch, break to get next batch
        if start_idx >= len(evs):
            break

print("Managed event bins per frame.")
# Now we have populated the eventbins_per_frame structure: 
# each RGB frame has 66 event bins, each bin contains a list of event chunks [x,y,t,p]

#___________________________________________________________________________________
# PASS 3: 
# For each frame process the event chunks in time bins, apply attention -> extract ROI
# save: event_chunk.npy, corresponding_rgb_frame.png, events_inside_roi.npy
# filename format: {label}_frame{f}_bin{b}_ROI{T/F}.npy
# filename format: {label}_frame{f}.png

if not HAND_SELECTION_ON:
    # Adjust the timestamps in each bin so that the values start from 0 but still remain in original units (microsecs)
    eventbins_normalized = adjust_timestamps(eventbins_per_frame)
    num_saved_instances = 0
    num_empty_eventbins = 0
    for f, (ts_frame, img) in enumerate(frames):
        bincount = 0
        for b, eventbin in enumerate(eventbins_normalized[f]):
            if bincount%SAVE_STRIDE == 0:
            # Check if bin has events
                if len(eventbin) == 0:
                    #print(f"Frame {f}, Bin {b}: No events, skipping")
                    num_empty_eventbins += 1
                    continue

                # save event chunk
                ev_np = np.sort(eventbins_normalized[f][b], order="timestamp") 
                ev_np = ev_np[0]
            
                # apply attention to get ROI on events
                events = np_events_to_store(ev_np) # events in a native dv EventStore formatevents = np_events_to_store(ev_np) # events in a native dv EventStore format
                
                visualizer = dv.visualization.EventVisualizer(resolution)
                visualizer.setBackgroundColor(dv.visualization.colors.black())
                visualizer.setPositiveColor(dv.visualization.colors.white())
                visualizer.setNegativeColor(dv.visualization.colors.white())

                event_frame = visualizer.generateImage(events) # prepare events for the attention model
                event_frame = cv.cvtColor(event_frame, cv.COLOR_RGB2GRAY)
                event_frame_u8 = np.ascontiguousarray(event_frame, dtype=np.uint8)  # forces contiguous buffer
                frame_tensor = torch.tensor(event_frame_u8, dtype=torch.float32)  # forces a copy (safe)
                frame_tensor = frame_tensor.unsqueeze(0).contiguous()  

                cv.namedWindow("Sample", cv.WINDOW_NORMAL)
                cv.imshow("Sample", event_frame) 
                cv.waitKey(30)

                # Run attention
                with torch.no_grad():
                    salmap, salmax_coords = run_attention(
                        frame_tensor, net_attention, device,
                        resolution, resolution,
                        config.ATTENTION_PARAMS['num_pyr'])
                cy, cx = int(salmax_coords[0]), int(salmax_coords[1]) # most salient point

                # extract ROI and downsample events to DOWNSAMPLED_RESxDOWNSAMPLED_RES via TONIC
                roi_size = np.random.choice(SIZE_ROI)
                events_downsampled = tonic_downsample_ROIevents(event_frame, cx,cy, ev_np, roi_size, downsample_size = DOWNSAMPLED_RES) # still events (t,x,y,p)
                np.save(str(OUT_DIR + f"/{LETTER}_frame{f:04d}_bin{b:02d}.npy"), events_downsampled)
                num_saved_instances += 1
                #transform downsampled input to tuple structure
                spikes = roi_to_spike_array(events_downsampled, roi_size=DOWNSAMPLED_RES)
                np.save(str(OUT_DIR_SPIKES + f"/{LETTER}_frame{f:04d}_bin{b:02d}.npy"), events_downsampled)
                
                #print(f"Frame {f}/{len(frames)}: processed, ROI center: {cx},{cy}")
            
            bincount +=1
        filled = int(50 * (f + 1) / len(frames))
        bar = "|" * filled + "." * (50 - filled)
        print(f"\r{LETTER} [{bar}]", end="", flush=True)
    print(f"{LETTER} - Saved instances: {num_saved_instances}")
    print(f"{LETTER} - Empty bins: {num_empty_eventbins}")

elif HAND_SELECTION_ON:
    #___________________________________________________________________________________
    # VISUALIZE + decide if keeping
    # First: show just the accumulated events (all 66 bin combined) with attention
    # -> let user decide if we keep this frame

    # KEEP: normalize timestamps in bins
    #       run attention on each bin, save events, frame and downsampled ROI events

    vis_full = dv.visualization.EventVisualizer(resolution)
    vis_full.setBackgroundColor(dv.visualization.colors.black())
    vis_full.setPositiveColor(dv.visualization.colors.white())
    vis_full.setNegativeColor(dv.visualization.colors.white())

    cv.namedWindow("Sample", cv.WINDOW_NORMAL)

    vis_ds = dv.visualization.EventVisualizer((DOWNSAMPLED_RES, DOWNSAMPLED_RES))
    vis_ds.setBackgroundColor(dv.visualization.colors.black())
    vis_ds.setPositiveColor(dv.visualization.colors.white())
    vis_ds.setNegativeColor(dv.visualization.colors.white())

    for f, (ts_frame, img_rgb) in enumerate(frames):

        for b in range(NUM_BINS):
            chunks = eventbins_per_frame[f][b]
            if len(chunks) == 0:
                continue

            # ---- build structured array for THIS bin ----
            ev_np = np.concatenate(chunks, axis=0)
            ev_np = np.sort(ev_np, order="timestamp")
            n_events_bin = len(ev_np)

            # optional: skip tiny bins
            if n_events_bin < 600:
                continue

            # ---- visualize events for THIS bin ----
            events_store = np_events_to_store(ev_np)
            ev_img = vis_full.generateImage(events_store)              # RGB
            ev_gray = cv.cvtColor(ev_img, cv.COLOR_RGB2GRAY)

            # ---- attention on THIS bin ----
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

            #ev_vis = draw_cross(ev_gray, cx, cy)
            att_vis = draw_cross(att_viz, cx, cy)

            # ---- downsample ROI events (48x48) for THIS bin ----
            roi_size = np.random.choice(SIZE_ROI)  # or fixed, or cycle sizes
            events_down = tonic_downsample_ROIevents(
                ev_gray, cx, cy, ev_np, roi_size,
                downsample_size=DOWNSAMPLED_RES
            )  # expects structured with t/x/y/p (your function)

            store_ds = np_events_to_store(events_down, timestamp_name="t", polarity_name="p")
            ds_img = vis_ds.generateImage(store_ds)                    # RGB small
            ds_bgr = cv.cvtColor(ds_img, cv.COLOR_RGB2BGR)

            # ---- compose side-by-side view ----
            # ensure BGR and same height
            ev_bgr  = cv.cvtColor(ev_vis,  cv.COLOR_GRAY2BGR) if ev_vis.ndim == 2 else ev_vis
            att_bgr = cv.cvtColor(att_vis, cv.COLOR_GRAY2BGR) if att_vis.ndim == 2 else att_vis

            H = ev_bgr.shape[0]
            ds_big = cv.resize(ds_bgr, (H, H), interpolation=cv.INTER_NEAREST)

            combined = np.hstack([ev_bgr, att_bgr, ds_big])

            # ---- overlay text ----
            text = f"frame {f}/{len(frames)-1}  bin {b}/{NUM_BINS-1}  events {n_events_bin}  ROI {roi_size}"
            cv.putText(combined, text, (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)

            cv.imshow("Sample", combined)

            # controls: Enter/space next bin, s save, q/esc quit
            key = cv.waitKey(0)
            if key in (27, ord("q")):
                cv.destroyAllWindows()
                raise SystemExit
            elif key == ord("s"):
                # save bin-level artifacts (adjust names as needed)
                np.save(str(OUT_DIR + f"/{LETTER}_frame{f:04d}_bin{b:02d}.npy"), events_down)
                #transform downsampled input to tuple structure
                spikes = roi_to_spike_array(events_down, roi_size=DOWNSAMPLED_RES)
                np.save(str(OUT_DIR_SPIKES + f"/{LETTER}_frame{f:04d}_bin{b:02d}.npy"), spikes)
                print(spikes)
            else:
                pass  # continue to next bin


cv.destroyAllWindows()


