import cv2
import base64
import numpy as np
import json
import time
from datetime import datetime, timezone, timedelta
from statistics import mean
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import VideoFileSource, StreamConfig, VideoMetadata

# ---------------------------------------------------
# Roboflow Client
# ---------------------------------------------------

client = InferenceHTTPClient.init(
    api_url="https://serverless.roboflow.com",
    api_key="ROBOFLOW_API_KEY"
)

source = VideoFileSource("pallet.mp4", realtime_processing=False)

VIDEO_OUTPUT = "zone_time"

config = StreamConfig(
    stream_output=[],
    data_output=["zone_time", "zone_output"],
    requested_plan="webrtc-gpu-medium",
    requested_region="us",
)

session = client.webrtc.stream(
    source=source,
    workflow="pallet-ac",
    workspace="tim-4ijf0",
    image_input="image",
    config=config
)

# ---------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------

active_tracks = {}
events = []
frames = []

MISSING_TIMEOUT_SEC = 1.0

run_start_dt = datetime.now(timezone.utc)

# ---------------------------------------------------
# LIVE DISPLAY CONFIG
# ---------------------------------------------------

DISPLAY_LIVE = True
DISPLAY_WINDOW = "Pallet Workflow Live"
THROTTLE_TO_REALTIME = True

_last_video_t = None
_last_wall_t = None

# ---------------------------------------------------
# JSON OUTPUT FILES
# ---------------------------------------------------

EVENT_FILE = "pallet_zone_events.json"
WAREHOUSE_FILE = "warehouse_metrics.json"
PALLET_FILE = "pallet_metrics.json"

# ---------------------------------------------------
# TIME HELPERS
# ---------------------------------------------------

def video_time_seconds(metadata: VideoMetadata):
    return float(metadata.pts) * float(metadata.time_base)

def to_datetime_str(sec):
    dt = run_start_dt + timedelta(seconds=sec)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------------------------------
# PARSE PREDICTIONS
# ---------------------------------------------------

def parse_zone_predictions(data):
    if "zone_output" not in data:
        return []
    return data["zone_output"].get("timed_detections", {}).get("predictions", [])

# ---------------------------------------------------
# LIVE VIDEO DISPLAY
# ---------------------------------------------------

def show_live_frame(frame, t_sec):

    global _last_video_t, _last_wall_t

    if not DISPLAY_LIVE:
        return True

    if THROTTLE_TO_REALTIME:

        now_wall = time.time()

        if _last_video_t is None:
            _last_video_t = t_sec
            _last_wall_t = now_wall
        else:

            dv = t_sec - _last_video_t
            dw = now_wall - _last_wall_t

            sleep_time = max(0, dv - dw)

            if sleep_time > 0:
                time.sleep(sleep_time)

    cv2.imshow(DISPLAY_WINDOW, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        return False

    return True

# ---------------------------------------------------
# FINALIZE PALLET EVENT
# ---------------------------------------------------

def finalize_track(tracker_id, out_time):

    st = active_tracks.get(tracker_id)
    if not st:
        return

    duration = round(out_time - st["in_time_sec"], 3)

    event = {
        "tracker_id": tracker_id,
        "class": st.get("class", "pallet"),
        "in_time": st["in_time"],
        "out_time": to_datetime_str(out_time),
        "in_time_sec": round(st["in_time_sec"],3),
        "out_time_sec": round(out_time,3),
        "duration_sec": duration,
        "max_time_in_zone_sec": round(st.get("max_time_in_zone_sec",0),3),
        "last_confidence": st.get("last_confidence"),
        "note": "Removed from zone"
    }

    events.append(event)

    del active_tracks[tracker_id]

    print(f"[EXIT] pallet {tracker_id}  duration={duration}s")

    update_reports()

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------

def compute_warehouse_metrics(events):

    if not events:
        return {}

    events_sorted = sorted(events, key=lambda e: e["in_time_sec"])

    latencies = [e["duration_sec"] for e in events_sorted]

    pickup_times = [e["out_time_sec"] for e in events_sorted]

    cycle_times = [
        pickup_times[i] - pickup_times[i-1]
        for i in range(1,len(pickup_times))
    ]

    start = events_sorted[0]["in_time_sec"]
    end = max(e["out_time_sec"] for e in events_sorted)

    duration = max(0.001, end-start)

    throughput = len(events_sorted)/(duration/3600)

    return {
        "total_pallets_completed": len(events_sorted),

        "avg_pickup_latency_sec": round(mean(latencies),3),
        "max_pickup_latency_sec": round(max(latencies),3),
        "min_pickup_latency_sec": round(min(latencies),3),

        "avg_forklift_cycle_time_sec": round(mean(cycle_times),3) if cycle_times else None,
        "max_forklift_cycle_time_sec": max(cycle_times) if cycle_times else None,
        "min_forklift_cycle_time_sec": min(cycle_times) if cycle_times else None,

        "throughput_pallets_per_hour": round(throughput,2)
    }

def compute_pallet_metrics(events):

    if not events:
        return {}

    ev = sorted(events, key=lambda x: x["in_time_sec"])

    durations = [e["duration_sec"] for e in ev]

    start = ev[0]["in_time_sec"]
    end = max(e["out_time_sec"] for e in ev)

    window = max(0.001,end-start)

    throughput = len(ev)/(window/3600)

    return {
        "total_pallet_events": len(ev),
        "unique_pallets": len(set(e["tracker_id"] for e in ev)),
        "avg_dwell_sec": round(mean(durations),3),
        "max_dwell_sec": round(max(durations),3),
        "min_dwell_sec": round(min(durations),3),
        "throughput_pallets_per_hour": round(throughput,2)
    }

# ---------------------------------------------------
# SAVE REPORTS
# ---------------------------------------------------

def update_reports():

    with open(EVENT_FILE,"w") as f:
        json.dump(events,f,indent=2)

    wm = compute_warehouse_metrics(events)

    with open(WAREHOUSE_FILE,"w") as f:
        json.dump(wm,f,indent=2)

    pm = compute_pallet_metrics(events)

    with open(PALLET_FILE,"w") as f:
        json.dump(pm,f,indent=2)

    print("\n--- Warehouse Metrics ---")
    print(json.dumps(wm,indent=2))

# ---------------------------------------------------
# STREAM CALLBACK
# ---------------------------------------------------

@session.on_data()
def on_data(data, metadata):

    t_sec = video_time_seconds(metadata)
    now_str = to_datetime_str(t_sec)

    preds = parse_zone_predictions(data)

    present_ids = set()

    for p in preds:

        tid = p.get("tracker_id")

        if tid is None:
            continue

        present_ids.add(tid)

        conf = p.get("confidence")
        cls = p.get("class","pallet")

        time_in_zone = p.get("time_in_zone")

        if tid not in active_tracks:

            active_tracks[tid] = {
                "tracker_id": tid,
                "class": cls,
                "in_time_sec": t_sec,
                "in_time": now_str,
                "last_seen_sec": t_sec,
                "last_confidence": conf,
                "max_time_in_zone_sec": float(time_in_zone or 0)
            }

            print(f"[ENTER] pallet {tid}")

        else:

            st = active_tracks[tid]

            st["last_seen_sec"] = t_sec
            st["last_confidence"] = conf

            if time_in_zone:
                st["max_time_in_zone_sec"] = max(
                    st["max_time_in_zone_sec"],
                    float(time_in_zone)
                )

    # detect exits

    to_finalize = []

    for tid,st in active_tracks.items():

        missing = t_sec - st["last_seen_sec"]

        if missing >= MISSING_TIMEOUT_SEC and tid not in present_ids:

            to_finalize.append((tid,st["last_seen_sec"]))

    for tid,out_t in to_finalize:
        finalize_track(tid,out_t)

    # VIDEO FRAME

    if VIDEO_OUTPUT in data:

        img = cv2.imdecode(
            np.frombuffer(base64.b64decode(data[VIDEO_OUTPUT]["value"]),np.uint8),
            cv2.IMREAD_COLOR
        )

        keep_running = show_live_frame(img,t_sec)

        if not keep_running:
            session.close()
            return

        frames.append((t_sec,metadata.frame_id,img))

    print(f"frame {metadata.frame_id} active={len(active_tracks)} events={len(events)}")

# ---------------------------------------------------
# RUN STREAM
# ---------------------------------------------------

session.run()

if DISPLAY_LIVE:
    cv2.destroyAllWindows()

# ---------------------------------------------------
# FINALIZE REMAINING TRACKS
# ---------------------------------------------------

for tid in list(active_tracks.keys()):
    finalize_track(tid, active_tracks[tid]["last_seen_sec"])

update_reports()

# ---------------------------------------------------
# OPTIONAL VIDEO OUTPUT
# ---------------------------------------------------

if frames:

    frames.sort(key=lambda x: x[1])

    fps = len(frames)/(frames[-1][0]-frames[0][0])

    h,w = frames[0][2].shape[:2]

    out = cv2.VideoWriter(
        "output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w,h)
    )

    for _,_,f in frames:
        out.write(f)

    out.release()

    print("output video saved")