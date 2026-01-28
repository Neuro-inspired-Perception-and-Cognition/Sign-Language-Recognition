from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import sys

IEBCS_ROOT = Path("IEBCS")  # add correct path to IEBCS 
sys.path.append(str(IEBCS_ROOT / "src"))

from dvs_sensor import DvsSensor
from event_buffer import EventBuffer



# Load Sign Language MNIST CSV
# ----------------------------
def load_sign_mnist_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    labels = df["label"].to_numpy(np.int64)
    images = df.drop(columns=["label"]).to_numpy(np.uint8).reshape(-1, 28, 28)
    return images, labels

# Upsample to 48x48
# ----------------------------
def upsample_to_48(gray28: np.ndarray) -> np.ndarray:
    return cv2.resize(gray28, (48, 48), interpolation=cv2.INTER_NEAREST)

# Generate shaking frames (RGB)
# ----------------------------
def make_shake_frames_rgb(
    rgb: np.ndarray,
    n_frames: int = 30,
    max_px: int = 3,
    seed: int | None = None,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    h, w = rgb.shape[:2]
    frames = []
    for _ in range(n_frames):
        dx, dy = rng.integers(-max_px, max_px + 1, size=2)
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        f = cv2.warpAffine(
            rgb, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REFLECT_101
        )
        frames.append(f)
    return frames


# IEBCS
# ----------------------------

class EventFrameRendererWindow:
    # rendering just the latest timewindow of events
    def __init__(self, width, height, tau):
        self.width, self.height = width, height
        self.window_us = tau
        self.now = 0

    def update(self, events, dt_us):
        self.now += dt_us

        img = np.full((self.height, self.width), 0, dtype=np.uint8)

        if events.i > 0:
            mask = events.ts[:events.i] >= (self.now - self.window_us)
            x = events.x[:events.i][mask]
            y = events.y[:events.i][mask]
            p = events.p[:events.i][mask]

            # do not distinguish polarity of events
            img[y[p == 1], x[p == 1]] = 255   # ON
            img[y[p == 0], x[p == 0]] = 255    

        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def to_luminance(rgb_uint8):
    luv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LUV)
    return (luv[:,:,0].astype(np.float32) / 255.0) * 1e4  

def rgb_to_iebcs_irradiance(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)  # 0..255
    return (gray / 255.0) * 1e4

def extract_events(ev) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if ev is None:
        return (np.empty((0,), np.uint64),
                np.empty((0,), np.int16),
                np.empty((0,), np.int16),
                np.empty((0,), np.uint8))

    # Get first existing attribute
    def get_attr(obj, names):
        for n in names:
            if hasattr(obj, n):
                return getattr(obj, n)
        return None

    if isinstance(ev, np.ndarray):
        if ev.dtype.fields is not None:
            x = ev["x"] if "x" in ev.dtype.fields else (ev["xs"] if "xs" in ev.dtype.fields else None)
            y = ev["y"] if "y" in ev.dtype.fields else (ev["ys"] if "ys" in ev.dtype.fields else None)
            t = ev["t"] if "t" in ev.dtype.fields else (ev["ts"] if "ts" in ev.dtype.fields else None)
            p = ev["p"] if "p" in ev.dtype.fields else (ev["polarity"] if "polarity" in ev.dtype.fields else None)
            if x is not None and y is not None and t is not None and p is not None:
                x = np.asarray(x)
                y = np.asarray(y)
                t = np.asarray(t)
                p = np.asarray(p)
                if p.min() < 0:
                    p = (p > 0).astype(np.uint8)
                else:
                    p = (p > 0).astype(np.uint8)
                return (t.astype(np.uint64),
                        x.astype(np.int16),
                        y.astype(np.int16),
                        p.astype(np.uint8))

    x = get_attr(ev, ["x", "xs", "X"])
    y = get_attr(ev, ["y", "ys", "Y"])
    t = get_attr(ev, ["t", "ts", "timestamp", "timestamps", "T"])
    p = get_attr(ev, ["p", "polarity", "pol", "P"])

    if x is None or y is None or t is None or p is None:
        for m in ["to_arrays", "as_arrays", "get_arrays", "numpy", "to_numpy"]:
            if hasattr(ev, m):
                out = getattr(ev, m)()
                if isinstance(out, dict):
                    x = out.get("x", out.get("xs"))
                    y = out.get("y", out.get("ys"))
                    t = out.get("t", out.get("ts"))
                    p = out.get("p", out.get("polarity"))
                elif isinstance(out, (tuple, list)) and len(out) >= 4:
                    t, x, y, p = out[0], out[1], out[2], out[3]
                break

    if x is None or y is None or t is None or p is None:
        raise TypeError(
            "Cannot extract events from IEBCS return object. "
            "Print(type(ev), dir(ev)) once and map the field names in extract_events()."
        )

    x = np.asarray(x)
    y = np.asarray(y)
    t = np.asarray(t)
    p = np.asarray(p)

    # Normalize polarity to {0,1}
    if p.dtype != np.bool_:
        if p.size > 0 and np.min(p) < 0:
            p = (p > 0)
        else:
            p = (p > 0)
    p = p.astype(np.uint8)

    return (t.astype(np.uint64),
            x.astype(np.int16),
            y.astype(np.int16),
            p.astype(np.uint8))


def frames_to_iebcs_event_stream(
    frames_rgb: list[np.ndarray],
    dt_us: int = 1000,
    sensor_params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns concatenated arrays: t_us, x, y, p
    """
    assert len(frames_rgb) > 0
    H, W = frames_rgb[0].shape[:2]

    sp = sensor_params or {}
    dvs = DvsSensor("RealTimeDVS")
    dvs.initCamera(
        W, H,
        lat=sp.get("lat", 100),
        jit=sp.get("jit", 10),
        ref=sp.get("ref", 100),
        tau=sp.get("tau", 10),
        th_pos=sp.get("th_pos", 0.4),
        th_neg=sp.get("th_neg", 0.4),
        th_noise=sp.get("th_noise", 0.1),
        bgnp=sp.get("bgnp", 0.01),
        bgnn=sp.get("bgnn", 0.01),
    )

    all_t, all_x, all_y, all_p = [], [], [], []
    for fr in frames_rgb:
        img = rgb_to_iebcs_irradiance(fr)
        ev = dvs.update(img, int(dt_us))
        t, x, y, p = extract_events(ev)
        if t.size:
            all_t.append(t); all_x.append(x); all_y.append(y); all_p.append(p)

    if len(all_t) == 0:
        return (np.empty((0,), np.uint64),
                np.empty((0,), np.int16),
                np.empty((0,), np.int16),
                np.empty((0,), np.uint8))

    return (np.concatenate(all_t),
            np.concatenate(all_x),
            np.concatenate(all_y),
            np.concatenate(all_p))


def save_event_stream_npz(path: str | Path, t_us, x, y, p, *, label: int, H: int, W: int, dt_us: int):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        t_us=t_us.astype(np.uint64),
        x=x.astype(np.int16),
        y=y.astype(np.int16),
        p=p.astype(np.uint8),
        label=int(label),
        H=int(H),
        W=int(W),
        dt_us=int(dt_us),
    )

# ----------------------------
# Conversion that also returns per-frame event packets
# ----------------------------
def frames_to_iebcs_events_and_stream(frames_rgb, dt_us, sensor_params, dvs_ctor, init_camera_fn,
                                     rgb_to_irradiance_fn, extract_events_fn):
    """
    dvs_ctor: e.g. lambda: DvsSensor("RealTimeDVS")
    init_camera_fn: function(dvs, W, H, sensor_params) that calls dvs.initCamera(...)

    """
    H, W = frames_rgb[0].shape[:2]

    dvs = dvs_ctor()
    init_camera_fn(dvs, W, H, sensor_params)

    per_frame_events = []
    all_t, all_x, all_y, all_p = [], [], [], []

    for fr in frames_rgb:
        img = rgb_to_irradiance_fn(fr)
        ev = dvs.update(img, int(dt_us))
        per_frame_events.append(ev)

        t, x, y, p = extract_events_fn(ev)
        if t.size:
            all_t.append(t); all_x.append(x); all_y.append(y); all_p.append(p)

    if len(all_t) == 0:
        t_us = np.empty((0,), np.uint64)
        x = np.empty((0,), np.int16)
        y = np.empty((0,), np.int16)
        p = np.empty((0,), np.uint8)
    else:
        t_us = np.concatenate(all_t).astype(np.uint64)
        x = np.concatenate(all_x).astype(np.int16)
        y = np.concatenate(all_y).astype(np.int16)
        p = np.concatenate(all_p).astype(np.uint8)

    return per_frame_events, (t_us, x, y, p)

def keep_positive_events(t_us, x, y, p):
    mask = (p == 1)
    return t_us[mask], x[mask], y[mask], p[mask]

def roi_to_spike_array(roi_events_down, roi_size=28):
    """
    Convert downscaled ROI events to spike array for SpiNNaker
    """
    if len(roi_events_down) == 0:
        return []

    # Extract fields - note the order in your data is (t, x, y, p)
    timestamps = roi_events_down['t']
    x_coords = roi_events_down['x']
    y_coords = roi_events_down['y']

    t_min = timestamps.min()
    timestamps_ms = ((timestamps - t_min) / 1000.0).astype(np.float64)

    # Map 2D coordinates to 1D neuron IDs using row-major ordering
    # neuron_id = y * roi_size + x
    neuron_ids = (y_coords * roi_size + x_coords).astype(np.int32)
    spikes = [(timestamps_ms[i], neuron_ids[i]) for i in range(len(roi_events_down))]

    # Sort by timestamp to maintain temporal order
    spikes.sort(key=lambda x: x[0])
    return spikes

def crop_events_first_n_ms(roi_events, n_ms):
    """
    Crop structured event array to the first n milliseconds.

    Args:
        roi_events: structured numpy array with fields ['t','x','y','p']
                    timestamps in microseconds
        n_ms: window length in milliseconds

    Returns:
        cropped structured array
    """
    if len(roi_events) == 0:
        return roi_events

    t0 = roi_events["t"].min()
    t_max = t0 + int(n_ms * 1000)

    mask = roi_events["t"] < t_max
    return roi_events[mask]

def events_to_structured_array(t_us, x, y, p):
    """
    Pack separate event arrays into a structured array
    compatible with roi_to_spike_array().
    """
    assert len(t_us) == len(x) == len(y) == len(p)

    dtype = np.dtype([
        ("t", np.uint64),
        ("x", np.int16),
        ("y", np.int16),
        ("p", np.uint8),
    ])

    ev = np.empty(len(t_us), dtype=dtype)
    ev["t"] = t_us
    ev["x"] = x
    ev["y"] = y
    ev["p"] = p
    return ev



def crop_events_skip_first_n_ms(events, skip_ms=1):
    """
    Skip the first skip_ms milliseconds of events
    Useful for removing initialization artifacts
    
    Args:
        events: structured array with 't' field in microseconds
        skip_ms: how many milliseconds to skip (default 1ms)
    
    Returns:
        events with first skip_ms removed and time reset to 0
    """
    skip_us = skip_ms * 1000
    
    # Keep only events after skip_us
    mask = events['t'] >= skip_us
    cropped = events[mask]
    
    # Reset time to start at 0
    if len(cropped) > 0:
        cropped['t'] = cropped['t'] - skip_us
    
    return cropped
# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    csv_path="/sign_mnist_test/sign_mnist_test.csv"

    imgs28, labels = load_sign_mnist_csv(csv_path) #imgs and labels
    out_root = Path("dvs_spikes_ieBCS") #set ouput repository name
    out_root.mkdir(parents=True, exist_ok=True)

    n_samples = 27455 #number of samples in directory
    n_variations = 2 #how many variations (samples) from each RGB image
    n_frames = 20 # how many frames will be used
    max_shake_px = 2 # max number of pixels in shaking/shifts

    dt_us = np.random.randint(2500, 3500)  # 10 ms per frame
    skip_initial_ms = 1  # Skip first 1ms - usually artifacts
    letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y',]

    total_generated = 0
    for idx in tqdm(range(min(n_samples, len(imgs28)))):
        gray48 = upsample_to_48(imgs28[idx])                   # upsample 
        rgb48 = np.repeat(gray48[:, :, None], 3, axis=2)       # RGB
        for var_idx in range(n_variations):
            seed = idx * n_variations + var_idx
            np.random.seed(seed)
            max_shake_px = np.random.randint(1, 3)  
            dt_us = np.random.randint(2500, 3500)
            frames = make_shake_frames_rgb(
                rgb48, n_frames=n_frames, max_px=max_shake_px, seed=seed
            )

            sensor_params = {'lat': 100,
                'jit': np.random.randint(5, 12),
                'ref': 100,
                'tau': 10,
                'th_pos': 0.4,
                'th_neg': 0.4,
                'th_noise': np.random.uniform(0.05, 0.15),
                'bgnp': 0.01,
                'bgnn': 0.01}
        
            def dvs_ctor():
                return DvsSensor("RealTimeDVS")

            def init_camera(dvs, W, H, sp):
                dvs.initCamera(
                    W, H,
                    lat=sp.get("lat", 100),
                    jit=sp.get("jit", 10),
                    ref=sp.get("ref", 100),
                    tau=sp.get("tau", 10),
                    th_pos=sp.get("th_pos", 0.4),
                    th_neg=sp.get("th_neg", 0.4),
                    th_noise=sp.get("th_noise", 0.1),
                    bgnp=sp.get("bgnp", 0.01),
                    bgnn=sp.get("bgnn", 0.01),
                )

            per_frame_events, (t_us, x, y, p) = frames_to_iebcs_events_and_stream(
                frames_rgb=frames,
                dt_us=dt_us,
                sensor_params=sensor_params,
                dvs_ctor=dvs_ctor,
                init_camera_fn=init_camera,
                rgb_to_irradiance_fn=rgb_to_iebcs_irradiance,   
                extract_events_fn=extract_events               
            )
        
            t_us, x, y, p = keep_positive_events(t_us, x, y, p)
            roi_events = events_to_structured_array(t_us, x, y, p)

            # skip first "skip_initial_ms" milliseconds to remove artifacts
            roi_events_clean = crop_events_skip_first_n_ms(roi_events, skip_ms=skip_initial_ms)
            
            # crop out a sequence of desided length
            roi_35ms  = crop_events_first_n_ms(roi_events_clean, 35) #events

            # convert to spikes
            spikes_35ms  = roi_to_spike_array(roi_35ms, roi_size=48) #spikes

            total_generated += 1

            # save 
            filename = f"{letters[labels[idx]]}_sample_{idx:05d}_var_{var_idx:02d}_spikes.npz"
            out_dir = out_root / f"test_35ms_smooth_2"
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(f"{out_dir}/{filename}", spikes_35ms)
        