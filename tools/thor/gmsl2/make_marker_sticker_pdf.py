"""Print-at-actual-size ArUco sticker sheet for a marker rig (vector PDF, A4).

Every module is a vector rectangle in one filled path, so nothing is resampled between this file and the
printer: the black square is exactly ``--black-mm`` when the sheet is printed at 100 %. A 150 mm horizontal
and a 100 mm vertical ruler on the page check that (print shops scale "to fit" by default, and the two axes
can come out different). The printed size is still a measurement, not this number: caliper the black
square of each sticker and use that (the 09-07 carrier stickers came out 47.24 mm for a nominal 48).

    .venv/bin/python tools/thor/gmsl2/make_marker_sticker_pdf.py --ids 35 36 37 38 39 \
        --out outputs/marker_rig_v3_stickers/rig_v3_dict6x6_50_ids35-39_48mm_A4.pdf

12 stickers per A4 sheet (3 x 4, shared cut lines), the IDs cycled over the slots; each sticker carries
"ArUco <dictionary> ID <n>" in the bottom white band, which also marks the marker's bottom edge.

Needs cv2 with aruco (the lerobot .venv); the PDF itself is written by hand, no PDF library.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

PT_PER_MM = 72.0 / 25.4
PAGE_W_MM, PAGE_H_MM = 210.0, 297.0
COLS, ROWS = 3, 4


def marker_modules(dictionary: str, marker_id: int) -> np.ndarray:
    """Module grid incl. the 1-module black border, True = black, row 0 = the marker's top edge."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    n = d.markerSize + 2
    img = cv2.aruco.generateImageMarker(d, marker_id, n, borderBits=1)
    return img < 128


class Page:
    """Top-left origin, millimetres; emits a PDF content stream."""

    def __init__(self) -> None:
        self.ops: list[str] = []

    def _xy(self, x: float, y: float) -> tuple[float, float]:
        return x * PT_PER_MM, (PAGE_H_MM - y) * PT_PER_MM

    def rects(self, rects: list[tuple[float, float, float, float]], gray: float = 0.0) -> None:
        """Fill all rectangles as ONE path, so abutting modules have no anti-aliasing seams."""
        if not rects:
            return
        self.ops.append(f"{gray:.3f} g")
        for x, y, w, h in rects:
            px, py = self._xy(x, y + h)
            self.ops.append(f"{px:.4f} {py:.4f} {w * PT_PER_MM:.4f} {h * PT_PER_MM:.4f} re")
        self.ops.append("f")

    def outline(self, x: float, y: float, w: float, h: float, width_mm: float, gray: float) -> None:
        px, py = self._xy(x, y + h)
        self.ops.append(
            f"{gray:.3f} G {width_mm * PT_PER_MM:.4f} w {px:.4f} {py:.4f} "
            f"{w * PT_PER_MM:.4f} {h * PT_PER_MM:.4f} re S"
        )

    def line(self, x0: float, y0: float, x1: float, y1: float, width_mm: float = 0.15) -> None:
        a, b = self._xy(x0, y0)
        c, d = self._xy(x1, y1)
        self.ops.append(f"0 G {width_mm * PT_PER_MM:.4f} w {a:.4f} {b:.4f} m {c:.4f} {d:.4f} l S")

    def text(
        self, x: float, y: float, s: str, size_pt: float = 8.0, bold: bool = False, gray: float = 0.0, font: str = ""
    ) -> None:
        """``y`` is the baseline. ASCII only (standard Helvetica, nothing embedded)."""
        px, py = self._xy(x, y)
        esc = s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        font = font or ("/F2" if bold else "/F1")
        self.ops.append(f"{gray:.3f} g BT {font} {size_pt:.2f} Tf {px:.4f} {py:.4f} Td ({esc}) Tj ET")

    def stream(self) -> bytes:
        return ("\n".join(self.ops) + "\n").encode("latin-1")


def draw_marker(page: Page, modules: np.ndarray, x: float, y: float, black_mm: float) -> None:
    n = modules.shape[0]
    m = black_mm / n
    rects = []
    for r in range(n):
        c = 0
        while c < n:  # horizontal runs: fewer rectangles, same union
            if modules[r, c]:
                c0 = c
                while c < n and modules[r, c]:
                    c += 1
                rects.append((x + c0 * m, y + r * m, (c - c0) * m, m))
            else:
                c += 1
    page.rects(rects)


def draw_ruler(page: Page, x: float, y: float, length_mm: int, vertical: bool) -> None:
    if vertical:
        page.line(x, y, x, y + length_mm, 0.2)
    else:
        page.line(x, y, x + length_mm, y, 0.2)
    for i in range(length_mm + 1):
        tick = 5.0 if i % 10 == 0 else (3.0 if i % 5 == 0 else 1.6)
        if vertical:
            page.line(x, y + i, x + tick, y + i, 0.12)
            if i % 10 == 0:
                page.text(x + 6.0, y + i + 1.0, str(i), 6)
        else:
            page.line(x + i, y, x + i, y - tick, 0.12)
            if i % 10 == 0:
                page.text(x + i - (0.9 if i < 10 else 1.8 if i < 100 else 2.6), y - 6.2, str(i), 6)


def sticker_label(dictionary: str, marker_id: int) -> str:
    return f"ArUco {dictionary.removeprefix('DICT_')}  ID {marker_id}"


def build_page(args: argparse.Namespace, ids: list[int], title: str, page_no: int, pages: int) -> Page:
    """3 x 4 stickers sharing their cut lines (61.5 mm: 4 across or 5 down do not fit on A4)."""
    p = Page()
    s, b = args.sticker_mm, args.black_mm
    n_mod = marker_modules(args.dictionary, ids[0]).shape[0]
    quiet = (s - b) / 2
    x0, y0 = 8.0, 20.0
    p.text(
        x0, 12.0,
        f"{title + '  -  ' if title else ''}ArUco {args.dictionary}  IDs {', '.join(map(str, sorted(set(ids))))}"
        f"  -  black {b:.1f} mm "
        f"({n_mod}x{n_mod} x {b / n_mod:.2f} mm)  -  cut {s:.1f} mm  -  sheet {page_no}/{pages}",
        8.0, True,
    )
    p.text(
        x0, 16.5,
        "PRINT AT 100% / ACTUAL SIZE, not 'fit to page'. Cut on the grey lines. Matte self-adhesive, no glossy "
        "lamination. The label marks the BOTTOM edge.",
        6.5,
    )
    label_pt = args.label_pt
    for k, marker_id in enumerate(ids):
        r, c = divmod(k, COLS)
        x, y = x0 + c * s, y0 + r * s
        p.outline(x, y, s, s, 0.1, 0.6)
        draw_marker(p, marker_modules(args.dictionary, marker_id), x + quiet, y + quiet, b)
        label = sticker_label(args.dictionary, marker_id)
        width = len(label) * 0.6 * label_pt / PT_PER_MM  # Courier: 600/1000 em per character
        # baseline 1.8 mm above the cut: the text stays >= 3.5 mm (0.6 module) clear of the black square
        p.text(x + (s - width) / 2, y + s - 1.8, label, label_pt, gray=args.label_gray, font="/F3")

    grid_bottom = y0 + ROWS * s
    draw_ruler(p, x0 + COLS * s + 2.5, y0, 100, vertical=True)
    y_ruler = grid_bottom + 12.0
    draw_ruler(p, x0, y_ruler, 150, vertical=False)
    p.text(
        x0, y_ruler + 5.0,
        "Rulers: this one 150.0 mm, the one at the right edge 100.0 mm. Caliper every black square after "
        "printing and use the",
        6.5, True,
    )
    p.text(
        x0, y_ruler + 9.0,
        f"measured value, not {b:.1f} (09-07 carrier stickers came out 1.6 % small). Also caliper the "
        "sticker thickness.",
        6.5, True,
    )
    return p


def write_pdf(pages: list[Page], out: Path) -> None:
    objs: list[bytes] = []
    n = len(pages)
    page_ids = [6 + 2 * i for i in range(n)]
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{' '.join(f'{i} 0 R' for i in page_ids)}] /Count {n} >>".encode())
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier-Bold /Encoding /WinAnsiEncoding >>")
    w, h = PAGE_W_MM * PT_PER_MM, PAGE_H_MM * PT_PER_MM
    for i, page in enumerate(pages):
        content = page.stream()
        objs.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w:.4f} {h:.4f}] "
                f"/Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R >> >> /Contents {page_ids[i] + 1} 0 R >>"
            ).encode()
        )
        objs.append(f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"endstream")
    buf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for k, body in enumerate(objs, start=1):
        offsets.append(len(buf))
        buf += f"{k} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(buf)
    buf += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        buf += f"{off:010d} 00000 n \n".encode()
    buf += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(bytes(buf))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--ids", type=int, nargs="+", help="one ID set, cycled over --sheets sheets")
    group.add_argument(
        "--sheet",
        nargs="+",
        action="append",
        metavar=("TITLE", "ID"),
        help="one sheet of its own IDs, cycled over its 12 slots: --sheet 'Rig A' 100 101 102 103 (repeatable)",
    )
    ap.add_argument("--dictionary", default="DICT_6X6_50")
    ap.add_argument("--black-mm", type=float, default=48.0)
    ap.add_argument("--sticker-mm", type=float, default=61.5)
    ap.add_argument("--sheets", type=int, default=1)
    ap.add_argument("--label-pt", type=float, default=6.5, help="sticker label size (Courier-Bold)")
    ap.add_argument("--label-gray", type=float, default=0.35, help="0 black .. 1 white")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    per_sheet = COLS * ROWS
    if args.sheet:
        try:
            groups = [(g[0], [int(v) for v in g[1:]]) for g in args.sheet]
        except ValueError:
            ap.error("--sheet takes a title and then integer IDs")
        if any(not ids for _, ids in groups):
            ap.error("--sheet needs at least one ID after its title")
        sheets = [(t, [ids[k % len(ids)] for k in range(per_sheet)]) for t, ids in groups]
    else:
        slots = [args.ids[k % len(args.ids)] for k in range(per_sheet * args.sheets)]
        sheets = [("", slots[i * per_sheet : (i + 1) * per_sheet]) for i in range(args.sheets)]
    all_ids = [i for _, ids in sheets for i in ids]
    if len({i for _, ids in groups for i in ids} if args.sheet else set(args.ids)) != sum(
        len(set(ids)) for _, ids in (groups if args.sheet else [("", args.ids)])
    ):
        ap.error("an ID appears on two sheets: each rig needs its own IDs")
    n_mod = marker_modules(args.dictionary, all_ids[0]).shape[0]
    if (args.sticker_mm - args.black_mm) / 2 < args.black_mm / n_mod:
        ap.error("white quiet zone would be narrower than one module")
    if COLS * args.sticker_mm + 8.0 + 13.0 > PAGE_W_MM:
        ap.error("three stickers plus the right-edge ruler do not fit across A4")
    write_pdf([build_page(args, ids, t, i + 1, len(sheets)) for i, (t, ids) in enumerate(sheets)], args.out)
    counts = {i: all_ids.count(i) for i in dict.fromkeys(all_ids)}
    print(args.out, "stickers per id:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
