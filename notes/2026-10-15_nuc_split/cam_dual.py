import time
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
cfgs={"side":dict(serial_number_or_name="243122071795",width=640,height=480,fps=60,exposure_us=15000,gain=128),
      "ee":dict(serial_number_or_name="315122271876",width=640,height=480,fps=60,exposure_us=15000,gain=70)}
for rnd in range(3):
    cams={k:RealSenseCamera(RealSenseCameraConfig(**kw)) for k,kw in cfgs.items()}
    for c in cams.values(): c.connect()
    prev={k:None for k in cams}; uniq={k:0 for k in cams}; t1=time.time()
    while time.time()-t1<10.0:
        for k,c in cams.items():
            img=c.async_read(timeout_ms=500); h=hash(img[::40,::40].tobytes())
            if h!=prev[k]: uniq[k]+=1; prev[k]=h
        time.sleep(0.002)
    for c in cams.values(): c.disconnect()
    print(f"round{rnd} both open 10s: " + ", ".join(f"{k} {uniq[k]/10:.1f} fr/s" for k in cams), flush=True)
print("ALL_DONE", flush=True)
