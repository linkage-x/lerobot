"""Use the existing offline renderer with isolated inputs/outputs only."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPT = ROOT / 'tools/fr3/render_contact_replay_20260907.py'


if __name__ == '__main__':
    if any(HERE.glob('episode_*_mujoco_2x_slow.mp4')):
        raise FileExistsError('Output video already exists')
    manifest = json.loads((HERE / 'ik_manifest.json').read_text())
    if any(e['ik_pass'] != e['frames'] for e in manifest['episodes']):
        raise RuntimeError('Refuse to render a failed IK trajectory as a successful replay')
    spec = importlib.util.spec_from_file_location('offline_renderer', SCRIPT)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    renderer.OUT = HERE
    renderer.MODEL = HERE / 'model/fr3_v2_p0_recovered.urdf'
    renderer.main()
    with (HERE / 'render_provenance.json').open('x') as f:
        json.dump(dict(script=str(SCRIPT), sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
                       model=str(renderer.MODEL), hardware_ready=False,
                       note='Corrected P0 candidate. Kinematic visualisation only, not dynamic tracking or safety certification.'), f, indent=2)
