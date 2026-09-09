#!/usr/bin/env python3
"""Bounded, read-only BOX feedback continuity test; never arms the actuator."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import time
import numpy as np
from box_transport import BoxTransport


def run(bundle,duration):
    from runtime import verify_bundle
    verify_bundle(bundle)
    c=json.loads((bundle/'config.json').read_text())
    box=BoxTransport(c['box_sdk_dir'],c['gripper_device_id'],c['gripper_ip'],allow_commands=False)
    report=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),read_only=True,motion_commands_sent=0,
                mode_commands_sent=0,requested_duration_s=duration)
    readings=[]
    with (bundle/'session.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            box.open()
            report.update(device_id=box.device_id,ip=box.ip,mode_before=box.mode,discovery=box.discovery)
            start=time.monotonic();previous=start;stale=False
            while time.monotonic()-start<duration:
                sample=box.fresh();now=time.monotonic()
                if sample is not None:
                    seq,width=sample
                    readings.append([now-start,seq,box.decoder.last_timestamp,width])
                    previous=now
                if now-previous>.2:
                    stale=True
                time.sleep(.005)
            rc,mode=box.box.get_mode(box.device_id,timeout_ms=500)
            report.update(mode_after=mode,mode_query_rc=rc,host_elapsed_s=time.monotonic()-start)
            if len(readings)<2:raise RuntimeError('Fewer than two fresh measurements')
            rows=np.asarray(readings)
            gaps=np.diff(np.r_[0.,rows[:,0],duration])
            report.update(new_measurements=len(readings),first_measurement_delay_s=readings[0][0],
                          maximum_host_gap_s=float(gaps.max()),median_host_gap_s=float(np.median(gaps)),
                          width_range_m=[float(rows[:,3].min()),float(rows[:,3].max())],
                          no_200ms_staleness=bool(not stale and gaps.max()<=.2),
                          mode_unchanged=rc==0 and mode==box.mode,
                          fresh_read_rate_hz=(len(readings)-1)/(readings[-1][0]-readings[0][0]),
                          note='Host observation rate, not a claim of device sampling frequency; identical widths can still be fresh timestamped measurements.')
        except Exception as exc:report['error']=str(exc)
        finally:box.close()
    report['measurements_t_seq_mcu_us_width_m']=readings
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--duration',type=float,default=25.)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not 1<=a.duration<=60 or a.output.exists():raise ValueError('Invalid duration / existing output')
    r=run(a.bundle,a.duration)
    text=json.dumps(r,indent=2,allow_nan=False)
    with a.output.open('x') as f:f.write(text)
    print(json.dumps({k:v for k,v in r.items() if k!='measurements_t_seq_mcu_us_width_m'},indent=2))
    raise SystemExit(1 if 'error' in r or not r.get('no_200ms_staleness') or not r.get('mode_unchanged') else 0)
