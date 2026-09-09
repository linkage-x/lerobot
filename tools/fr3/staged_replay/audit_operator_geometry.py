"""Offline only: full recordings plus proposed transit from a saved snapshot."""
import argparse
import json
from pathlib import Path
from plans import geometry, start_check, transit, validate
from replay_timed_candidate import evaluate
from runtime import save

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--prepared-snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False, parents=True)
    root = a.bundle
    plan = json.loads((root / 'timed_plan.json').read_text())
    previous = json.loads(a.prepared_snapshot.read_text())
    scene = json.loads((root / 'scene.json').read_text())
    q, width = previous['snapshot']['q'], previous['initial_measured_width_m']
    for ep in plan['episodes']:
        first = evaluate(ep, 0.)
        stages = {'replay': ep, 'start-only': transit(q, width, first[:7], first[7]),
                  'start-check': start_check(q, width, first[:7])}
        for phase, stage in stages.items():
            print(f'Checking episode {ep["episode"]} / {phase}', flush=True)
            report = dict(episode=ep['episode'], phase=phase, derivatives=validate(stage),
                          geometry=geometry(stage, root / 'model.urdf', scene),
                          source='Saved snapshot; not an authorization for current hardware state')
            save(a.output / f'ep{ep["episode"]}_{phase}.json', report)
            if not report['geometry']['sampled_geometry_pass']:
                raise RuntimeError('Geometry gate failed; see saved report')
