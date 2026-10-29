# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The FreeCAD half of ``dual_fr3_compliant_handle``: the parts, the assembly and the drawings.

Runs inside FreeCAD's own Python (``freecad freecadcmd``), not the repo's, with the environment variable
``COMPLIANT_HANDLE_PARAMS`` naming the ``params.json`` that ``dual_fr3_compliant_handle --cad`` writes, and
``COMPLIANT_HANDLE_STAGE``: ``geometry`` (the parts, their files and ``meshes/`` for the renders) or
``drawings`` (the same parts again, then the sheets, with the renders the repo side made in between). It
imports nothing from the repo. freecadcmd swallows a script's exceptions and its output, so everything goes
to ``cad_<stage>.log`` in the output directory, and ``cad_<stage>_done.json`` is written last.

Frames, in mm. The handle frame has its origin at the bar centre, x along the bar, y across the grip faces
and z up; the table is at z = -grip height. The tray frame has its origin on the table under the plate
centre; the right hand's handle is at y = -handle_y and the left hand's at +handle_y.

Writes:

* STEP for each part, and the assembly with the off-the-shelf parts drawn as stand-ins;
* STL for the printed parts, and ``meshes/`` (every assembly part in the tray frame) for the renders;
* ``parts.json``: volumes and masses;
* ``drawing.pdf`` (one page per sheet) and a PNG per sheet: the handle assembly, the two printed parts, the
  plate, and the renders.
"""

import json
import math
import os
import traceback
from pathlib import Path

PARAMS = Path(os.environ.get("COMPLIANT_HANDLE_PARAMS", "params.json"))
STAGE = os.environ.get("COMPLIANT_HANDLE_STAGE", "drawings")
P = json.loads(PARAMS.read_text(encoding="utf-8"))
OUT = Path(P["out"])
LOG = open(OUT / f"cad_{STAGE}.log", "w", encoding="utf-8")  # noqa: SIM115

#: g/cm3; the printed parts at the fill a slicer gives them (4 walls, 40 % gyroid: ~0.6 of solid)
DENSITY = {"aluminium": 2.70, "petg": 1.27}
PRINT_FILL = 0.6


def log(*args):
    print(*args, file=LOG, flush=True)


try:
    import FreeCAD as App
    import Import
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import Part
    import TechDraw
    from matplotlib.backends.backend_pdf import PdfPages

    V = App.Vector
    H, T = P["handle"], P["plate"]
    B = H["buffer"]
    L, W, HB = H["bar_length_mm"], H["grip_width_mm"], H["bar_height_mm"]
    PITCH, PIN, TRAVEL = H["buffer_pitch_mm"], H["pin_diameter_mm"], H["travel_mm"]
    CHEEK, WC, DECK, BASE = H["cheek_mm"], H["cradle_width_mm"], H["deck_mm"], H["base_mm"]
    BD, BH, STUD = B["diameter_mm"], B["height_mm"], B["stud_mm"]
    GRIP = T["grip_height_mm"]
    TP = T["thickness_mm"]

    # ---------------------------------------------------------------- geometry (handle frame) ---
    Z_TABLE = -GRIP
    Z_PLATE_TOP = Z_TABLE + TP
    Z_BASE_TOP = Z_PLATE_TOP + BASE
    Z_DECK_TOP = -HB / 2 - BH
    Z_DECK_BOT = Z_DECK_TOP - DECK
    X_IN = L / 2 + TRAVEL  # cheek inner face
    X_OUT = X_IN + CHEEK
    HOLE_R = PIN / 2 + TRAVEL
    Z_CHEEK_TOP = HOLE_R + 8.0
    SHOULDER = TRAVEL + CHEEK + 2.0
    GRIP_ZONE, RECESS = H["grip_zone_mm"], H["knuckle_recess_mm"]
    INSERT_M6 = (8.0, 14.0)  # heat-set insert hole: diameter, depth
    INSERT_M8 = (10.0, 16.0)
    INSERT_M4 = (5.6, 7.0)
    BASE_HOLES = [(sx * 45.0, sy * 15.0) for sx in (-1, 1) for sy in (-1, 1)]
    if Z_DECK_BOT - Z_BASE_TOP < 25.0:
        raise ValueError(f"no room under the deck for the nuts: {Z_DECK_BOT - Z_BASE_TOP:.1f} mm")

    def box(x0, x1, y0, y1, z0, z1):
        return Part.makeBox(x1 - x0, y1 - y0, z1 - z0, V(x0, y0, z0))

    def cyl(r, h, base, axis):
        return Part.makeCylinder(r, h, V(*base), V(*axis))

    def make_bar():
        bar = box(-L / 2, L / 2, -W / 2, W / 2, -HB / 2, HB / 2)
        for s in (-1, 1):
            bar = bar.cut(cyl(INSERT_M6[0] / 2, INSERT_M6[1], (s * PITCH / 2, 0, -HB / 2), (0, 0, 1)))
            bar = bar.cut(cyl(INSERT_M8[0] / 2, INSERT_M8[1], (s * L / 2, 0, 0), (-s, 0, 0)))
        # the grip zone: the top sunk under the gripper's inner knuckles (the MJCF leaves them 0.5 mm over a
        # 32 mm bar, as over the pivot insert); its ends show where to grip
        bar = bar.cut(box(-GRIP_ZONE / 2, GRIP_ZONE / 2, -W / 2 - 1, W / 2 + 1, HB / 2 - RECESS, HB / 2 + 1))
        return bar

    def make_cradle():
        c = box(-X_OUT, X_OUT, -WC / 2, WC / 2, Z_PLATE_TOP, Z_BASE_TOP)
        for s in (-1, 1):
            x0, x1 = sorted((s * X_IN, s * X_OUT))
            c = c.fuse(box(x0, x1, -WC / 2, WC / 2, Z_BASE_TOP, Z_CHEEK_TOP))
        c = c.fuse(box(-X_IN, X_IN, -WC / 2, WC / 2, Z_DECK_BOT, Z_DECK_TOP))
        for s in (-1, 1):
            c = c.cut(cyl(HOLE_R, CHEEK + 2, (s * (X_IN - 1), 0, 0), (s, 0, 0)))
            c = c.cut(cyl(3.25, DECK + 2, (s * PITCH / 2, 0, Z_DECK_BOT - 1), (0, 0, 1)))
        for x, y in BASE_HOLES:
            c = c.cut(cyl(INSERT_M4[0] / 2, INSERT_M4[1], (x, y, Z_PLATE_TOP), (0, 0, 1)))
        return c.removeSplitter()

    def make_buffer():
        rubber = cyl(BD / 2, BH, (0, 0, Z_DECK_TOP), (0, 0, 1))
        top = cyl(3.0, STUD, (0, 0, -HB / 2), (0, 0, 1))
        bottom = cyl(3.0, STUD, (0, 0, Z_DECK_TOP - STUD), (0, 0, 1))
        return rubber.fuse([top, bottom])

    def make_screw(s):
        shoulder = cyl(PIN / 2, SHOULDER, (s * L / 2, 0, 0), (s, 0, 0))
        head = cyl(8.0, 7.0, (s * (L / 2 + SHOULDER), 0, 0), (s, 0, 0))
        thread = cyl(4.0, 13.0, (s * L / 2, 0, 0), (-s, 0, 0))
        return shoulder.fuse([head, thread])

    def make_nut(s):
        hexagon = Part.makePolygon(
            [V(s * PITCH / 2 + 5.77 * math.cos(a), 5.77 * math.sin(a), Z_DECK_BOT - 5.0)
             for a in np.radians(np.arange(0, 361, 60))]
        )  # fmt: skip
        return Part.Face(hexagon).extrude(V(0, 0, 5.0))

    def make_plate():
        x, y = T["length_mm"] / 2, T["width_mm"] / 2
        r = T["corner_mm"]
        plate = box(-x, x, -y, y, 0, TP)
        for sx in (-1, 1):
            for sy in (-1, 1):
                plate = plate.cut(_corner(sx * x, sy * y, sx, sy, r))
        for hy in (-T["handle_y_mm"], T["handle_y_mm"]):
            for hx, dy in BASE_HOLES:
                plate = plate.cut(cyl(2.25, TP + 2, (hx, hy + dy, -1), (0, 0, 1)))
                # 90 deg countersink from below: the screws go up into the cradle, the plate sits flat
                plate = plate.cut(Part.makeCone(4.2, 0.0, 4.2, V(hx, hy + dy, 0.0), V(0, 0, 1)))
        return plate.removeSplitter()

    def _corner(cx, cy, sx, sy, r):
        """The material outside a radius-r round at a plate corner (cx, cy)."""
        square = box(cx - (r if sx > 0 else 0), cx + (r if sx < 0 else 0), cy - (r if sy > 0 else 0),
                     cy + (r if sy < 0 else 0), -1, TP + 1)  # fmt: skip
        return square.cut(cyl(r, TP + 4, (cx - sx * r, cy - sy * r, -2), (0, 0, 1)))

    bar, cradle, plate = make_bar(), make_cradle(), make_plate()
    buffers = [make_buffer().translate(V(s * PITCH / 2, 0, 0)) for s in (-1, 1)]
    screws = [make_screw(s) for s in (-1, 1)]
    nuts = [make_nut(s) for s in (-1, 1)]
    for name, shape in (("bar", bar), ("cradle", cradle), ("plate", plate)):
        if not shape.isValid() or len(shape.Solids) != 1:
            raise ValueError(f"{name}: {len(shape.Solids)} solids, valid {shape.isValid()}")
    # the bar touches only the buffers: its travel to the cheeks and over the deck is free
    gap_cheek = X_IN - L / 2
    gap_deck = -HB / 2 - Z_DECK_TOP
    if bar.common(cradle).Volume > 1e-6:
        raise ValueError("bar and cradle overlap")
    log(
        f"bar {bar.Volume / 1e3:.1f} cm3, cradle {cradle.Volume / 1e3:.1f} cm3, plate {plate.Volume / 1e3:.1f} cm3"
    )
    log(
        f"gap bar end -> cheek {gap_cheek} mm, bar -> deck {gap_deck} mm, under the deck {Z_DECK_BOT - Z_BASE_TOP} mm"
    )

    def placed(shape, y, turn):
        """A handle-frame shape in the tray frame: at y, the bar centre at grip height, turned about z."""
        s = shape.copy()
        if turn:
            s.rotate(V(0, 0, 0), V(0, 0, 1), 180)
        s.translate(V(0, y, GRIP))
        return s

    handle_parts = [("bar", bar, "petg"), ("cradle", cradle, "petg")]
    stand_ins = (
        [("buffer", b) for b in buffers]
        + [("shoulder_screw", s) for s in screws]
        + [("nut", n) for n in nuts]
    )
    doc = App.newDocument("compliant_handle")
    objs = []

    def add(label, shape):
        o = doc.addObject("Part::Feature", label)
        o.Shape = shape
        objs.append(o)
        return o

    add("plate", plate)
    for side, y, turn in (("right", -T["handle_y_mm"], False), ("left", T["handle_y_mm"], True)):
        for name, shape, _ in handle_parts:
            add(f"{side}_{name}", placed(shape, y, turn))
        for i, (name, shape) in enumerate(stand_ins):
            add(f"{side}_{name}_{i}", placed(shape, y, turn))
    doc.recompute()
    Import.export(objs, str(OUT / "tray_assembly.step"))
    for name, shape in (("bar", bar), ("cradle", cradle), ("plate", plate)):
        shape.exportStep(str(OUT / f"{name}.step"))
    for name, shape in (("bar", bar), ("cradle", cradle)):
        # printed: the handle frame turned so the part sits on its largest flat face
        mesh_shape = shape.copy()
        if name == "bar":
            mesh_shape.translate(V(0, 0, HB / 2))
        else:
            mesh_shape.translate(V(0, 0, -Z_PLATE_TOP))
        mesh_shape.exportStl(str(OUT / f"{name}.stl"))

    # ---------------------------------------------------------------- masses ---
    def grams(shape, material, fill=1.0):
        return shape.Volume / 1e3 * DENSITY[material] * fill

    masses = {
        "plate": grams(plate, "aluminium"),
        "bar": grams(bar, "petg", PRINT_FILL),
        "cradle": grams(cradle, "petg", PRINT_FILL),
        "buffer": 20.0,
        "shoulder_screw": 14.0,
        "inserts_nuts_screws": 25.0,
    }
    per_handle = (
        masses["bar"]
        + masses["cradle"]
        + 2 * (masses["buffer"] + masses["shoulder_screw"])
        + masses["inserts_nuts_screws"]
    )
    total_g = masses["plate"] + 2 * per_handle
    log(f"masses g: {json.dumps({k: round(v, 1) for k, v in masses.items()})}, tray {total_g:.0f} g")
    parts = {
        "volumes_cm3": {
            n: round(s.Volume / 1e3, 2) for n, s in (("bar", bar), ("cradle", cradle), ("plate", plate))
        },
        "masses_g": {k: round(v, 1) for k, v in masses.items()},
        "per_handle_g": round(per_handle, 1),
        "tray_g": round(total_g, 1),
        "print_fill": PRINT_FILL,
        "gaps_mm": {
            "bar_end_to_cheek": gap_cheek,
            "bar_to_deck": gap_deck,
            "under_deck": Z_DECK_BOT - Z_BASE_TOP,
        },
    }
    (OUT / "parts.json").write_text(json.dumps(parts, indent=2) + "\n", encoding="utf-8")

    def write_stl(shape, path, deflection=0.05):
        """Binary STL from FreeCAD's own tessellation (finer than exportStl's default)."""
        verts, faces = shape.tessellate(deflection)
        vv = np.array([[v.x, v.y, v.z] for v in verts], dtype=np.float32)
        tri = vv[np.asarray(faces)]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        rec = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        rec["n"], rec["v"] = nrm, tri
        with open(path, "wb") as f:
            f.write(b"dual_fr3_compliant_handle".ljust(80, b" "))
            f.write(np.uint32(len(tri)).tobytes())
            f.write(rec.tobytes())

    if STAGE == "geometry":
        meshes = OUT / "meshes"
        meshes.mkdir(exist_ok=True)
        for old in meshes.glob("*.stl"):
            old.unlink()
        for o in objs:
            write_stl(o.Shape, meshes / f"{o.Label}.stl")
        (OUT / "cad_geometry_done.json").write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")
        log("geometry done")

    if STAGE == "drawings":
        # ---------------------------------------------------------------- views ---
        def view_edges(shape, right, up):
            """Visible and hidden 2D polylines of a shape seen with ``right`` to the right and ``up`` up (the
            viewer on right x up). TechDraw projects along +z with the shape's x, y as the view's x, y, so the
            shape is turned into that frame first."""
            r, u = np.asarray(right, float), np.asarray(up, float)
            n = np.cross(r, u)
            turn = App.Matrix(*r, 0, *u, 0, *n, 0, 0, 0, 0, 1)
            s = shape.copy()
            s.transformShape(turn, True)
            res = TechDraw.projectEx(s, V(0, 0, 1))
            out = {"visible": [], "hidden": []}
            for key, idx in (("visible", (0, 1, 3)), ("hidden", (5, 6, 8))):
                for i in idx:
                    comp = res[i]
                    if comp is None or comp.isNull():
                        continue
                    for e in comp.Edges:
                        pts = (
                            e.discretize(Deflection=0.05)
                            if e.Curve.TypeId != "Part::GeomLine"
                            else [e.Vertexes[0].Point, e.Vertexes[-1].Point]
                        )
                        out[key].append(np.array([[p.x, p.y] for p in pts]))
            return out

        VIEWS = {
            "front": ((1, 0, 0), (0, 0, 1)),
            "top": ((1, 0, 0), (0, 1, 0)),
            "side": ((0, 1, 0), (0, 0, 1)),
        }

        def draw(ax, edges, color="k"):
            for poly in edges["hidden"]:
                ax.plot(poly[:, 0], poly[:, 1], color="0.55", lw=0.5, ls=(0, (3, 2)))
            for poly in edges["visible"]:
                ax.plot(poly[:, 0], poly[:, 1], color=color, lw=0.9)

        def dim(ax, p, q, offset, text=None, side=1.0, fs=7):
            """A linear dimension between p and q, drawn ``offset`` mm off the line pq (to its left for side=1)."""
            p, q = np.asarray(p, float), np.asarray(q, float)
            d = q - p
            length = float(np.linalg.norm(d))
            t = d / length
            nrm = np.array([-t[1], t[0]]) * side
            a, b = p + nrm * offset, q + nrm * offset
            for base, tip in ((p, a), (q, b)):
                ax.plot(*zip(base + nrm * 1.0, tip + nrm * 1.5, strict=True), color="tab:blue", lw=0.4)
            ax.annotate("", xy=a, xytext=b, arrowprops={"arrowstyle": "<->", "lw": 0.5, "color": "tab:blue",
                                                         "shrinkA": 0, "shrinkB": 0})  # fmt: skip
            mid = (a + b) / 2 + nrm * 1.8
            angle = math.degrees(math.atan2(t[1], t[0]))
            if angle > 90 or angle <= -90:
                angle -= 180 * np.sign(angle)
            ax.text(*mid, text or f"{length:g}", fontsize=fs, color="tab:blue", ha="center", va="center",
                    rotation=angle, rotation_mode="anchor", bbox={"fc": "white", "ec": "none", "pad": 0.3})  # fmt: skip

        def note(ax, xy, text, xytext, fs=6.5):
            ax.annotate(text, xy=xy, xytext=xytext, fontsize=fs, color="tab:red",
                        arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "tab:red"},
                        bbox={"fc": "white", "ec": "none", "pad": 0.3})  # fmt: skip

        for font in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Micro Hei", "Droid Sans Fallback"):
            if any(font == f.name for f in matplotlib.font_manager.fontManager.ttflist):
                plt.rcParams["font.family"] = [font, "DejaVu Sans"]
                break
        else:
            cjk = [p for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                               "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                               "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf") if Path(p).exists()]  # fmt: skip
            if cjk:
                matplotlib.font_manager.fontManager.addfont(cjk[0])
                plt.rcParams["font.family"] = [
                    matplotlib.font_manager.FontProperties(fname=cjk[0]).get_name(),
                    "DejaVu Sans",
                ]
        log(f"font {plt.rcParams['font.family']}")

        def sheet(title, subtitle):
            fig = plt.figure(figsize=(16.54, 11.69))  # A3 landscape
            fig.text(0.02, 0.975, title, fontsize=15, weight="bold", va="top")
            fig.text(0.02, 0.945, subtitle, fontsize=8.5, va="top", color="0.3")
            fig.text(0.98, 0.012, f"dual_fr3_compliant_handle_cad.py · {P.get('created', '')} · 单位 mm · 第三角投影",
                     fontsize=7, ha="right", color="0.4")  # fmt: skip
            return fig

        def panel(fig, rect, label):
            ax = fig.add_axes(rect)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(label, fontsize=9, loc="left")
            return ax

        def fit(ax, edges, pad=14):
            pts = np.concatenate(edges["visible"] + edges["hidden"])
            (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
            ax.set_xlim(x0 - pad, x1 + pad)
            ax.set_ylim(y0 - pad, y1 + pad)

        def text_block(fig, rect, lines, fs=7.5):
            ax = fig.add_axes(rect)
            ax.axis("off")
            ax.text(
                0,
                1,
                "\n".join(lines),
                fontsize=fs,
                va="top",
                family=plt.rcParams["font.family"],
                linespacing=1.45,
            )

        chosen = P.get("chosen", {})
        k_t, k_r = chosen.get("k_n_per_mm", {}), chosen.get("k_nm_per_rad", {})
        buffer_name = f"Ø{BD:g}×{BH:g} {B['thread']} 双头外牙橡胶减震柱 {B['shore_a']:g}A"

        pdf = PdfPages(OUT / "drawing.pdf")
        pngs = []

        def finish(fig, name):
            pdf.savefig(fig)
            fig.savefig(OUT / f"{name}.png", dpi=110)
            pngs.append(f"{name}.png")
            plt.close(fig)

        # ---- sheet 1: the handle assembly
        assembly = Part.makeCompound([bar, cradle, *buffers, *screws, *nuts,
                                      box(-T["length_mm"] / 2, T["length_mm"] / 2, -WC / 2 - 15, WC / 2 + 15, Z_TABLE, Z_PLATE_TOP)])  # fmt: skip
        fig = sheet(
            "D6 柔性把手 · 装配（一侧；另一侧相同、绕 z 转 180°）",
            f"夹爪夹住 {W:g} mm 宽的平面把手（与 pivot 插件 v2 同宽、同面）；把手坐在两只 {buffer_name} 上；"
            f"两端台阶螺钉穿过支座耳板 Ø{2 * HOLE_R:g} 孔，四周留 {TRAVEL:g} mm：柔性用完就刚性限位，减震柱撕裂时托盘挂在螺钉上。",
        )
        e_front = view_edges(assembly, *VIEWS["front"])
        ax = panel(fig, [0.03, 0.36, 0.5, 0.55], "主视图（从 −y 看，右手侧把手；x 沿把手）")
        draw(ax, e_front)
        fit(ax, e_front, 22)
        dim(ax, (-L / 2, HB / 2), (L / 2, HB / 2), 8, f"把手 {L:g}")
        dim(ax, (-PITCH / 2, Z_DECK_TOP), (PITCH / 2, Z_DECK_TOP), -6, f"减震柱间距 {PITCH:g}")
        dim(ax, (-X_OUT, Z_PLATE_TOP), (X_OUT, Z_PLATE_TOP), -12, f"支座 {2 * X_OUT:g}")
        dim(ax, (X_OUT + 6, Z_TABLE), (X_OUT + 6, 0), 8, f"把手中心离桌面 {GRIP:g}", side=-1)
        dim(ax, (-L / 2 - 26, -HB / 2), (-L / 2 - 26, HB / 2), 4, f"{HB:g}")
        dim(ax, (L / 2, 5), (X_IN, 5), 7, f"{TRAVEL:g}")
        dim(ax, (-X_OUT - 4, Z_DECK_TOP), (-X_OUT - 4, -HB / 2), 4, f"{BH:g}", side=-1)
        note(
            ax,
            (0, HB / 2 - RECESS),
            f"夹持区：顶面下沉 {RECESS:g} 的 {GRIP_ZONE:g} 段（给夹爪内指节让位）",
            (-20, HB / 2 + 26),
        )
        note(
            ax,
            (X_IN + CHEEK / 2, HOLE_R),
            f"耳板孔 Ø{2 * HOLE_R:g}，台阶 Ø{PIN:g}：四周 {TRAVEL:g} mm 限位",
            (X_IN - 30, 46),
        )
        note(ax, (PITCH / 2, Z_DECK_BOT - 2.5), "M6 螺母从侧面开口拧", (PITCH / 2 + 10, Z_DECK_BOT - 30))
        note(ax, (-PITCH / 2, Z_DECK_TOP + BH / 2), buffer_name, (-PITCH / 2 - 60, Z_DECK_TOP - 5))
        note(ax, (-X_OUT + 10, Z_PLATE_TOP - TP / 2), f"托盘 {TP:g} mm 铝板", (-X_OUT - 30, Z_TABLE - 14))

        e_side = view_edges(assembly, *VIEWS["side"])
        ax = panel(fig, [0.55, 0.36, 0.25, 0.55], "右视图（从 +x 看；y 是两手连线方向）")
        draw(ax, e_side)
        fit(ax, e_side, 22)
        dim(ax, (-W / 2, HB / 2), (W / 2, HB / 2), 8, f"夹持面间距 {W:g}")
        dim(ax, (-WC / 2, Z_PLATE_TOP), (WC / 2, Z_PLATE_TOP), -12, f"{WC:g}")
        note(ax, (W / 2, 0), "手指在这两面合拢", (W / 2 + 6, 30))

        e_top = view_edges(Part.makeCompound([bar, cradle, *buffers, *screws]), *VIEWS["top"])
        ax = panel(fig, [0.80, 0.55, 0.19, 0.36], "俯视图")
        draw(ax, e_top)
        fit(ax, e_top, 10)

        lines = [
            "刚度（公式值，货架件 ±30%；装上后按「台架检查」实测）",
            f"  沿把手 x {k_t.get('x', '?')} N/mm · 两手连线 y {k_t.get('y', '?')} N/mm · 竖直 z {k_t.get('z', '?')} N/mm",
            f"  转动 x {k_r.get('x', '?')} · y {k_r.get('y', '?')} · z {k_r.get('z', '?')} Nm/rad",
            f"  限位 {TRAVEL:g} mm / {chosen.get('turn_stop_deg', '?')}°；托盘 {total_g / 1e3:.2f} kg 自重压缩 {chosen.get('weight_sag_mm', '?')} mm",
            "",
            "为什么这样做（D6 预算，tools/fr3/dual_fr3_compliant_handle.py）",
            "  · 只降关节刚度不够：每只手的力矩是瓶颈，再软就下沉、关节摩擦死区超标",
            "  · 柔性放在物体侧：夹爪、指垫、F8 打滑实验都不用改；硬度可换",
            "  · B3 实测摆放下所有候选都满足；没做 B3 时只有 40–50A 的软胶勉强满足",
            "",
            "台架检查（装好后、上机前）",
            f"  1 支座压在桌上，把手中点挂 1 kg：竖直下沉应 ≈ {9.81 / k_t['z']:.2f} mm"
            if k_t.get("z")
            else "  1 支座压在桌上，把手中点挂 1 kg，读竖直下沉",
            f"  2 沿 y 推把手 5 N（弹簧秤），百分表应读 ≈ {5.0 / k_t['y']:.1f} mm；差 2 倍以上换硬度"
            if k_t.get("y")
            else "  2 沿 y 推把手 5 N，读位移",
            "  3 推到限位：台阶螺钉应先碰耳板孔，把手不碰支座",
            "",
            "装配顺序",
            "  1 把手、支座压入热熔螺母（M6×2 底面、M8×2 两端；M4×4 支座底）",
            "  2 减震柱上端旋入把手底面；下端穿支座 deck 孔，侧面开口上 M6 螺母",
            "  3 台阶螺钉从耳板外侧穿入，拧进把手两端（螺钉台阶顶住把手端面）",
            "  4 支座用 4×M4 沉头螺钉从托盘底面固定",
        ]
        text_block(fig, [0.03, 0.03, 0.45, 0.3], lines)
        bom = [
            "物料（每个托盘：两套把手）",
            f"  托盘板  {T['length_mm']:g}×{T['width_mm']:g}×{TP:g} 6061 铝板，圆角 R{T['corner_mm']:g}，8 个 M4 沉孔 ×1",
            "  把手    PETG 打印（4 壁、40% 填充、夹持面朝侧面）×2",
            "  支座    PETG 打印（底面朝下）×2",
            f"  减震柱  {buffer_name} ×4（另备 50A/60A 各 4 只，同孔可换）",
            f"  台阶螺钉 ISO 7379 Ø{PIN:g}×16 M8（等高螺丝）×4",
            "  热熔螺母 M6×4、M8×4、M4×8；M6 螺母 ×4；M4×8 沉头螺钉 ×8",
            "",
            f"估计质量  板 {masses['plate']:.0f} g + 把手 2×{per_handle:.0f} g ≈ {total_g:.0f} g",
            "（打印件按 60% 实心估，装好后称重填回 --mass）",
        ]
        text_block(fig, [0.52, 0.03, 0.46, 0.3], bom)
        finish(fig, "sheet1_assembly")

        # ---- sheet 2: the printed parts
        fig = sheet("D6 柔性把手 · 打印件", "PETG；热熔螺母孔按所购螺母规格微调；尖角可倒 0.5")
        e = view_edges(bar, *VIEWS["front"])
        ax = panel(fig, [0.03, 0.55, 0.42, 0.36], "把手 · 主视图")
        draw(ax, e)
        fit(ax, e, 20)
        dim(ax, (-L / 2, HB / 2), (L / 2, HB / 2), 8)
        dim(ax, (-PITCH / 2, -HB / 2), (PITCH / 2, -HB / 2), -8, f"{PITCH:g}")
        dim(ax, (L / 2 + 6, -HB / 2), (L / 2 + 6, HB / 2), 4, side=-1)
        dim(
            ax,
            (-GRIP_ZONE / 2, HB / 2 - RECESS),
            (GRIP_ZONE / 2, HB / 2 - RECESS),
            9,
            f"夹持区 {GRIP_ZONE:g}，下沉 {RECESS:g}",
        )
        note(
            ax,
            (PITCH / 2, -HB / 2 + 7),
            f"Ø{INSERT_M6[0]:g}×{INSERT_M6[1]:g} 深（M6 热熔螺母）×2",
            (PITCH / 2 - 10, -HB / 2 - 20),
        )
        note(
            ax,
            (-L / 2 + 8, 0),
            f"Ø{INSERT_M8[0]:g}×{INSERT_M8[1]:g} 深（M8 热熔螺母）两端",
            (-L / 2 + 10, HB / 2 + 18),
        )
        e = view_edges(bar, *VIEWS["side"])
        ax = panel(fig, [0.46, 0.55, 0.2, 0.36], "把手 · 右视图")
        draw(ax, e)
        fit(ax, e, 14)
        dim(ax, (-W / 2, HB / 2), (W / 2, HB / 2), 6, f"{W:g}（夹持面，平面度 0.2）")
        e = view_edges(bar, *VIEWS["top"])
        ax = panel(fig, [0.68, 0.55, 0.3, 0.36], "把手 · 俯视图")
        draw(ax, e)
        fit(ax, e, 14)

        e = view_edges(cradle, *VIEWS["front"])
        ax = panel(fig, [0.03, 0.06, 0.42, 0.45], "支座 · 主视图")
        draw(ax, e)
        fit(ax, e, 20)
        dim(ax, (-X_OUT, Z_PLATE_TOP), (X_OUT, Z_PLATE_TOP), -8, f"{2 * X_OUT:g}")
        dim(ax, (-X_IN, Z_DECK_TOP), (X_IN, Z_DECK_TOP), 6, f"{2 * X_IN:g}")
        dim(
            ax,
            (X_OUT + 5, Z_PLATE_TOP),
            (X_OUT + 5, Z_CHEEK_TOP),
            6,
            f"{Z_CHEEK_TOP - Z_PLATE_TOP:g}",
            side=-1,
        )
        dim(ax, (X_OUT + 5, Z_PLATE_TOP), (X_OUT + 5, 0), 18, f"孔心 {-Z_PLATE_TOP:g}", side=-1)
        dim(ax, (-X_OUT - 5, Z_PLATE_TOP), (-X_OUT - 5, Z_DECK_TOP), 6, f"{Z_DECK_TOP - Z_PLATE_TOP:g}")
        dim(ax, (-PITCH / 2, Z_DECK_BOT), (PITCH / 2, Z_DECK_BOT), -6, f"Ø6.5 ×2，间距 {PITCH:g}")
        dim(ax, (X_IN, Z_CHEEK_TOP), (X_OUT, Z_CHEEK_TOP), 5, f"{CHEEK:g}")
        note(
            ax,
            (X_IN + CHEEK / 2, HOLE_R),
            f"Ø{2 * HOLE_R:g} 通孔（与把手轴同心）",
            (X_IN - 40, Z_CHEEK_TOP + 14),
        )
        dim(ax, (-X_IN - CHEEK - 0, Z_BASE_TOP), (-X_IN - CHEEK, Z_PLATE_TOP), 14, f"{BASE:g}")
        e = view_edges(cradle, *VIEWS["side"])
        ax = panel(fig, [0.46, 0.06, 0.2, 0.45], "支座 · 右视图")
        draw(ax, e)
        fit(ax, e, 14)
        dim(ax, (-WC / 2, Z_PLATE_TOP), (WC / 2, Z_PLATE_TOP), -8, f"{WC:g}")
        e = view_edges(cradle.copy().rotate(V(0, 0, 0), V(1, 0, 0), 180), (1, 0, 0), (0, 1, 0))
        ax = panel(fig, [0.68, 0.06, 0.3, 0.45], "支座 · 仰视图（底面）")
        draw(ax, e)
        fit(ax, e, 14)
        dim(ax, (-45, 15), (45, 15), 10, "90")
        dim(ax, (45, -15), (45, 15), 10, "30", side=-1)
        note(ax, (45, 15), f"Ø{INSERT_M4[0]:g}×{INSERT_M4[1]:g} 深（M4 热熔螺母）×4", (10, 40))
        finish(fig, "sheet2_printed_parts")

        # ---- sheet 3: the plate and the whole tray
        fig = sheet(
            "D6 柔性把手 · 托盘板与整体",
            f"{T['length_mm']:g}×{T['width_mm']:g}×{TP:g} 6061 铝板，激光/水切；8×Ø4.5 过孔 + 90° 沉孔 Ø8.4（M4 沉头），底面平",
        )
        e = view_edges(plate, (0, 1, 0), (-1, 0, 0))
        ax = panel(fig, [0.03, 0.45, 0.6, 0.45], "托盘板 · 俯视图（横向 = y，两手连线方向）")
        draw(ax, e)
        fit(ax, e, 26)
        dim(ax, (-T["width_mm"] / 2, T["length_mm"] / 2), (T["width_mm"] / 2, T["length_mm"] / 2), 12)
        dim(
            ax, (T["width_mm"] / 2, -T["length_mm"] / 2), (T["width_mm"] / 2, T["length_mm"] / 2), 12, side=-1
        )
        dim(
            ax,
            (-T["handle_y_mm"], -T["length_mm"] / 2),
            (T["handle_y_mm"], -T["length_mm"] / 2),
            -12,
            f"两把手中心 {2 * T['handle_y_mm']:g}",
        )
        dim(ax, (T["handle_y_mm"] - 15, -45), (T["handle_y_mm"] + 15, -45), -8, "30")
        dim(ax, (T["handle_y_mm"] - 30, -45), (T["handle_y_mm"] - 30, 45), 4, "90")

        def picture(rect, name, title):
            """A render the repo side made from ``meshes/`` (MuJoCo), or a note that it is missing."""
            ax = fig.add_axes(rect)
            ax.axis("off")
            ax.set_title(title, fontsize=9, loc="left")
            path = OUT / f"{name}.png"
            if path.exists():
                ax.imshow(plt.imread(path))
            else:
                ax.text(0.5, 0.5, f"{name}.png 未生成", ha="center", va="center", color="0.5")

        picture(
            [0.62, 0.50, 0.37, 0.42],
            "render_tray",
            "整体（右手侧把手在 −y；黄 = 夹持把手，蓝 = 支座，黑 = 橡胶减震柱）",
        )
        picture([0.62, 0.04, 0.37, 0.42], "render_handle", "单个把手")
        text_block(
            fig,
            [0.03, 0.05, 0.55, 0.30],
            [
                "和仿真的对应（tools/fr3/dual_fr3_coop_carry.py 的 Tray）",
                f"  把手中心 y = ±{T['handle_y_mm']:g}，离桌面 {GRIP:g}：与 coop carry 计划的两只手同一位置",
                f"  夹爪合到 {W:g} mm；TCP 在把手中心下方 {H['tcp_below_bar_mm']:g} mm（pivot 插件 v2 的激光实测）",
                "",
                "使用前提（D6 结论）",
                "  · 先做 B3（两台 base 之间的关系实测到 ~3 mm），否则摆放误差会把把手推到限位",
                "  · 第一次上机：空托盘，持物段 --held-scale 1（柔性在把手里，不再降关节刚度），",
                "    runtime 的内力门禁保持 15 N",
                "  · 质量按实物称重后重跑 tools/fr3/dual_fr3_compliant_handle.py --mass <kg>",
            ],
        )
        finish(fig, "sheet3_plate_tray")

        # ---- sheet 4: in the arms
        if (OUT / "render_gripped.png").exists():
            fig = sheet(
                "D6 柔性把手 · 在双 FR3 上（coop carry 计划，持物段中间一帧）",
                f"夹爪按 MJCF 合到 {W:g} mm；右手在 −y 把手。仿真里托盘是跟随计划的 mocap 体，不做动力学。",
            )
            picture([0.02, 0.06, 0.55, 0.84], "render_gripped", "整体")
            picture([0.58, 0.06, 0.41, 0.84], "render_gripped_closeup", "右手与把手")
            finish(fig, "sheet4_in_the_arms")
        pdf.close()
        (OUT / "cad_drawings_done.json").write_text(
            json.dumps({"ok": True, "sheets": pngs}) + "\n", encoding="utf-8"
        )
        log("drawings done")
except Exception:
    traceback.print_exc(file=LOG)
finally:
    LOG.close()
