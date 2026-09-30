#!/usr/bin/env python3
"""One page to audit a grasp-loop run after the fact: every trial's verdicts next to its photos.

An unattended run grades itself -- the lift's width for the grasp, the servo's stop height for
the insertion -- and nothing on the rig sees a staged peg that fell over. The loop saves the side
and wrist cameras at three moments (staged, over the hole, after the descent); this lays them out
beside what the loop decided, so a person can check fifty trials in a few minutes and mark the
ones where the picture disagrees. Marks stay in the browser (localStorage) and copy out as text.

    python tools/fr3/grasp_loop_audit.py outputs/analysis/grasp_loop/grasp_YYYYMMDD_HHMMSS.jsonl

writes `<run>_audit.html` next to the row file; it reads the photos from `<run>_peg/`, so open it
where the run's outputs are.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from tools.fr3.grasp_loop import summarize_grasp_loop

MOMENTS = (("staged", "摆放后"), ("before", "插入前"), ("after", "插入后"))


def load_run(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(trial rows, every other record), in file order."""

    trials, events = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        (trials if row.get("kind") == "trial" else events).append(row)
    return trials, events


def photos(peg_dir: Path, trial: int) -> dict[str, list[str]]:
    """Moment -> the photo files for it, relative to the page (which sits beside `peg_dir`)."""

    found: dict[str, list[str]] = {}
    for moment, _label in MOMENTS:
        names = sorted(p.name for p in peg_dir.glob(f"trial_{trial:03d}_{moment}_*.png")) if peg_dir.is_dir() else []
        found[moment] = [f"{peg_dir.name}/{name}" for name in names]
    return found


def trial_facts(row: dict[str, Any]) -> list[tuple[str, str, str]]:
    """(label, value, tone) for a trial card; tone is '' or 'bad' or 'good'."""

    insert = row.get("insert") or {}
    verdict = str(row.get("verdict"))
    facts = [
        ("摆放", str(row.get("staging")), ""),
        ("抓取", verdict, "good" if verdict == "held" else "bad" if verdict in ("empty", "no_close", "collision") else ""),
    ]
    if "inserted" in row:
        inserted = row.get("inserted")
        facts.append(("插入", {True: "进", False: "没进", None: "-"}[inserted], "good" if inserted else "bad"))
    if insert:
        facts.append(("自动判定", str(insert.get("autoVerdict")), ""))
        if insert.get("grade") is not None:
            facts.append(("人工判定", str(insert.get("grade")), ""))
        tried = insert.get("searchTried") or int(insert.get("searchIndex") or 0) + 1
        facts.append(("落点", f"idx {insert.get('searchIndex')} / 第 {tried} 次", ""))
        facts.append(("停止高度", f"{insert.get('aboveTargetMm')} mm", ""))
        if insert.get("dfzPeakN") is not None:
            facts.append(("峰值力", f"{insert.get('dfzPeakN')} N" + (" 封顶" if insert.get("pressCapped") else ""), ""))
    facts.append(("用时", f"{row.get('trialS')} s", ""))
    return facts


def render(run_path: Path) -> str:
    trials, events = load_run(run_path)
    peg_dir = run_path.with_name(f"{run_path.stem}_peg")
    start = next((e for e in events if e.get("kind") == "run_start"), {})
    end = next((e for e in reversed(events) if e.get("kind") == "run_end"), {})
    request = start.get("request") or {}
    # A run still going, or cut off, has no run_end: summarise its rows the way the loop would.
    summary = end.get("summary") or summarize_grasp_loop(trials)
    e2e = summary.get("endToEnd") or {}

    head = [
        f"试验 {len(trials)} 次",
        f"抓住 {summary.get('held', '-')}/{summary.get('graded', '-')}",
        f"作废 {summary.get('voided', sum(1 for t in trials if t.get('verdict') == 'voided'))}",
    ]
    if e2e:
        head.append(f"插入 {e2e.get('inserted')}/{e2e.get('graded')}")
    head.append("有人值守" if request.get("attended") else "无人值守")
    if end:
        head.append(f"结束：{end.get('halted') or '跑完'}")

    event_items = "".join(
        f"<li><b>{html.escape(str(e.get('kind')))}</b> trial {html.escape(str(e.get('trial', '-')))} "
        f"{html.escape(str(e.get('reason') or e.get('error') or e.get('pickWidth') or ''))}</li>"
        for e in events
        if e.get("kind") not in ("run_start", "run_end")
    )

    cards = []
    for row in trials:
        trial = int(row.get("trial", 0))
        facts = "".join(
            f'<span class="fact {tone}"><i>{html.escape(label)}</i>{html.escape(value)}</span>'
            for label, value, tone in trial_facts(row)
        )
        shots = photos(peg_dir, trial)
        strips = []
        for moment, label in MOMENTS:
            if not shots[moment]:
                continue
            imgs = "".join(
                f'<a href="{html.escape(src)}" target="_blank"><img loading="lazy" src="{html.escape(src)}" '
                f'alt="{html.escape(Path(src).name)}"></a>'
                for src in shots[moment]
            )
            strips.append(f'<figure><figcaption>{label}</figcaption><div class="imgs">{imgs}</div></figure>')
        flags = "".join(
            f'<button class="flag" data-trial="{trial}" data-flag="{key}">{text}</button>'
            for key, text in (("fell", "销倒了"), ("wrong", "判定错"), ("other", "其他"))
        )
        cards.append(
            f'<section class="card" id="t{trial}"><header><h2>#{trial}</h2>{facts}</header>'
            f'<div class="strips">{"".join(strips) or "<p class=none>没有照片</p>"}</div>'
            f'<footer>{flags}</footer></section>'
        )

    title = f"审核 {run_path.stem}"
    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ --bg:#f6f6f4; --card:#fff; --ink:#1d1d1b; --mute:#6b6b66; --line:#deded8; --bad:#b3261e; --good:#1e6b34; --mark:#fbe7a1; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141413; --card:#1f1f1d; --ink:#ecece8; --mute:#a0a09a; --line:#34342f; --bad:#f2998f; --good:#8fd19e; --mark:#5c4b12; }} }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 system-ui, sans-serif; }}
main {{ max-width:1200px; margin:0 auto; padding:16px; }}
h1 {{ font-size:18px; margin:0 0 4px; }}
.head span {{ margin-right:14px; }}
.bar {{ position:sticky; top:0; background:var(--bg); padding:8px 0; border-bottom:1px solid var(--line); z-index:1; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; margin:12px 0; padding:10px 12px; }}
.card.marked {{ background:var(--mark); }}
header {{ display:flex; flex-wrap:wrap; gap:6px 12px; align-items:baseline; }}
h2 {{ font-size:16px; margin:0 6px 0 0; }}
.fact i {{ font-style:normal; color:var(--mute); margin-right:4px; }}
.fact.bad {{ color:var(--bad); font-weight:600; }} .fact.good {{ color:var(--good); }}
.strips {{ display:flex; flex-wrap:wrap; gap:12px; margin-top:8px; }}
figure {{ margin:0; }} figcaption {{ color:var(--mute); font-size:12px; }}
.imgs {{ display:flex; gap:4px; }}
img {{ width:180px; max-width:42vw; border-radius:4px; display:block; }}
button {{ font:inherit; padding:3px 10px; margin:6px 6px 0 0; border:1px solid var(--line); border-radius:14px; background:transparent; color:var(--ink); cursor:pointer; }}
button.on {{ background:var(--bad); color:#fff; border-color:var(--bad); }}
textarea {{ width:100%; min-height:60px; margin-top:6px; }}
</style></head>
<body><main>
<h1>{html.escape(title)}</h1>
<div class="head">{"".join(f"<span>{html.escape(h)}</span>" for h in head)}</div>
<ul>{event_items}</ul>
<div class="bar"><label><input type="checkbox" id="onlyFail"> 只看失败 / 已标记</label>
<button id="copy">复制已标记</button><span id="count"></span></div>
{"".join(cards)}
<textarea id="out" readonly placeholder="标记会出现在这里"></textarea>
</main>
<script>
const KEY = "grasp_audit:{html.escape(run_path.stem)}";
let marks = {{}};
try {{ marks = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ marks = {{}}; }}
function save() {{ try {{ localStorage.setItem(KEY, JSON.stringify(marks)); }} catch (e) {{}} }}
function paint() {{
  let lines = [];
  document.querySelectorAll(".card").forEach(card => {{
    const t = card.id.slice(1), set = marks[t] || [];
    card.classList.toggle("marked", set.length > 0);
    card.querySelectorAll(".flag").forEach(b => b.classList.toggle("on", set.includes(b.dataset.flag)));
    if (set.length) lines.push(`#${{t}}: ${{set.join(",")}}`);
    const fail = card.querySelector(".fact.bad") !== null;
    card.style.display = document.getElementById("onlyFail").checked && !fail && !set.length ? "none" : "";
  }});
  document.getElementById("out").value = lines.join("\\n");
  document.getElementById("count").textContent = ` 已标记 ${{lines.length}} 次`;
}}
document.querySelectorAll(".flag").forEach(b => b.addEventListener("click", () => {{
  const set = new Set(marks[b.dataset.trial] || []);
  set.has(b.dataset.flag) ? set.delete(b.dataset.flag) : set.add(b.dataset.flag);
  marks[b.dataset.trial] = [...set]; save(); paint();
}}));
document.getElementById("onlyFail").addEventListener("change", paint);
document.getElementById("copy").addEventListener("click", () => {{
  const out = document.getElementById("out"); out.select();
  try {{ navigator.clipboard.writeText(out.value); }} catch (e) {{ document.execCommand("copy"); }}
}});
paint();
</script></body></html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("run", type=Path, help="the run's row file, grasp_*.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="default: <run>_audit.html beside it")
    args = parser.parse_args(argv)
    out = args.out or args.run.with_name(f"{args.run.stem}_audit.html")
    out.write_text(render(args.run), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
