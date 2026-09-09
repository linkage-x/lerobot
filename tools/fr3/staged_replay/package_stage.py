"""Create a NEW isolated package; never install into production site-packages."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def dump(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def package(repo,out):
    stage=repo/'outputs/p0_recovery_replay_20260907'
    out.mkdir(parents=True,exist_ok=False)
    for p in Path(__file__).parent.glob('*.py'):
        shutil.copy2(p,out/p.name)
    for name in ['replay_timed_candidate.py','replay_ik_trajectory_guarded.py','check_measured_table_offline.py','test_timed_candidate_offline.py']:
        shutil.copy2(repo/'tools/fr3'/name,out/name)
    for src,dst in [('player_collision_stage/timed_plan.json','timed_plan.json'),('measured_table_stage/scene_measurement.json','scene.json'),('measured_table_stage/robot_snapshot.json','tool_reference.json')]:
        shutil.copy2(stage/src,out/dst)
    tree=ET.parse(stage/'model/fr3_v2_p0_recovered.urdf')
    (out/'meshes').mkdir()
    for mesh in tree.findall('.//mesh'):
        path=Path(mesh.get('filename'))
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        dst=out/'meshes'/(digest+path.suffix)
        if not dst.exists():shutil.copy2(path,dst)
        mesh.set('filename',str(dst.relative_to(out)))
    compiler=tree.find('.//compiler')
    if compiler is not None:compiler.set('meshdir','.')
    tree.write(out/'model.urdf',encoding='utf-8',xml_declaration=True)
    shutil.copytree(stage/'hardware_stage/native',out/'native')
    shutil.copy2(Path(__file__).with_name('run.sh'),out/'run.sh')
    shutil.copy2(Path(__file__).with_name('run_replay.sh'),out/'run_replay.sh')
    shutil.copy2(Path(__file__).with_name('run_replay_once.sh'),out/'run_replay_once.sh')
    shutil.copy2(Path(__file__).with_name('README.md'),out/'README.md')
    (out/'python/panda_py').mkdir(parents=True)
    (out/'python/panda_py/__init__.py').write_text('"""Isolated staged extension. No production Panda import."""\n')
    dump(out/'config.json',dict(robot_ip='192.168.11.102',gripper_ip='192.168.1.119',
        gripper_device_id=596523097,box_sdk_dir='/home/nvidia/lerobot/tools/thor/box_sdk'))
    dump(out/'physical_test_release.json',dict(approved=False,prepared_sha256=None,
        reason='Deployment and read-only checks only; unresolved geometry contacts and physical testing require separate review.'))


def seal(out):
    files={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*')
           if p.is_file() and not set(p.relative_to(out).parts)&{'build','__pycache__','logs'}
           and p.name not in {'manifest.json','physical_test_release.json','session.lock'}
           and not p.name.startswith('box_sdk.log')}
    sdk=Path('/home/nvidia/lerobot/tools/thor/box_sdk')
    external=[sdk/'python/box_collection_sdk-0.1.0-py3-none-any.whl',sdk/'box_sdk.conf',sdk/'share/monte_gripper.urdf',
              Path('/usr/local/lib/libfranka.so.0.15.0')]+list((sdk/'lib').glob('*.so'))
    dump(out/'manifest.json',dict(schema='fr3-staged-bundle-v1',files=files,
        external_dependencies={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in external},
        physical_validation_completed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['package','seal'])
    p.add_argument('--repo',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    package(a.repo,a.output) if a.action=='package' else seal(a.output)
