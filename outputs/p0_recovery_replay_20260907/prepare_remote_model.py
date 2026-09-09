"""Rebase mesh references for read-only Thor geometry queries."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT / 'src/lerobot/robots/franka_research3/assets/franka_fr3'
THOR_ASSETS = Path('/home/nvidia/box_api/models/fr3_p0_v2')

if __name__ == '__main__':
    tree = ET.parse(HERE / 'model/fr3_v2_p0_recovered.urdf')
    meshes = {}
    for mesh in tree.findall('.//mesh'):
        src = Path(mesh.attrib['filename'])
        dst = THOR_ASSETS / src.relative_to(ASSETS)
        mesh.set('filename', str(dst))
        meshes[str(dst)] = hashlib.sha256(src.read_bytes()).hexdigest()
    with (HERE / 'model/fr3_v2_p0_recovered.thor.urdf').open('xb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)
    with (HERE / 'model/mesh_hashes_thor.json').open('x') as f:
        json.dump(meshes, f, indent=2)
