import { useState } from "react";
import type { CalibrationPromotionReview } from "../types";
import { intrinsicsPromotionNote, promotionView } from "./status";

/**
 * The review a calibration has to pass through before production loads it.
 *
 * Shaped by one incident and one measurement. The incident: a solve produced
 * extrinsics whose own summary said "cam_09 has moved", the panel showed that
 * run as live, and production went on loading the previous one for seven days
 * because the only way to make a solve take effect was a hand edit of a YAML
 * file. The measurement: the two candidate runs could not have been separated
 * automatically -- the one that self-scored better (0.244 px against 0.273) was
 * the one missing the moved camera -- so the decision has to stay with a person.
 *
 * Hence the split. The comparison below is not behind a button, because a review
 * you have to ask for is one that gets skipped. The promotion is one click,
 * because a step you have to do in an editor is one that gets skipped. And there
 * is no verdict anywhere on screen, because the numbers that could produce one
 * are known to produce the wrong one.
 */
export function PromotionPanel({
  review,
  disabled,
  onPromote,
}: {
  review: CalibrationPromotionReview | undefined;
  disabled: boolean;
  onPromote: (options: {
    kinds: ("intrinsics" | "extrinsics")[];
    acknowledge?: string[];
  }) => Promise<{ ok: boolean; error?: string }>;
}) {
  // Acknowledgement is per-blocker and starts empty on every render of a new
  // review: a single "I understand" checkbox would be ticked once and then
  // carried silently across later, different risks.
  const [acknowledged, setAcknowledged] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  const view = promotionView(review);
  if (!view.visible) return null;

  const outstanding = view.blockers.filter((blocker) => !acknowledged.includes(blocker.kind));
  const intrinsicsNote = intrinsicsPromotionNote(review);

  const promote = async () => {
    setPending(true);
    setError("");
    const result = await onPromote({ kinds: view.kinds, acknowledge: acknowledged });
    setPending(false);
    if (!result.ok) setError(result.error || "提升失败");
  };

  return (
    <div className="cali-promotion">
      <p className="cali-promotion-head">
        <b>{view.headline}</b>
      </p>

      {view.summary ? <p className="small">{view.summary}</p> : null}
      {view.world ? <p className="small mono">{view.world}</p> : null}
      {intrinsicsNote ? <p className="small">{intrinsicsNote}</p> : null}

      {view.rows.length ? (
        <div className="table-scroll">
          <table className="cali-promotion-table">
            <thead>
              <tr>
                <th>相机</th>
                <th>与其余相机的间距变化（中位）</th>
                <th>相对朝向变化（中位）</th>
              </tr>
            </thead>
            <tbody>
              {view.rows.map((row) => (
                <tr key={row.camera}>
                  <td className="mono">{row.camera}</td>
                  <td>{row.baselineMm} mm</td>
                  <td>{row.rotationDeg}°</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
      <p className="small">
        以上都是<b>规范无关</b>的量：相机间距与相对朝向不随整体坐标系变化，所以不需要先对齐两份标定，
        也没有「哪台没动」这个前提。取中位而不是最大值，是因为一台相机独自移动时，
        <b>它自己的每一条基线都会变，而其它相机各自只变一条</b>——中位能把移动的那台单独指出来。
      </p>
      {view.rmseNote ? <p className="small cali-warn">{view.rmseNote}</p> : null}

      {view.blockers.length ? (
        <div className="cali-promotion-blockers">
          {view.blockers.map((blocker) => (
            <label key={blocker.kind} className="cali-check">
              <input
                type="checkbox"
                checked={acknowledged.includes(blocker.kind)}
                disabled={disabled || pending}
                onChange={(event) =>
                  setAcknowledged((previous) =>
                    event.target.checked
                      ? [...previous, blocker.kind]
                      : previous.filter((kind) => kind !== blocker.kind),
                  )
                }
              />
              <span>{blocker.message}</span>
            </label>
          ))}
        </div>
      ) : null}

      <div className="control-row">
        <button
          className="cali-btn-primary"
          disabled={disabled || pending || outstanding.length > 0}
          onClick={() => void promote()}
        >
          提升为生产标定
        </button>
        <span className="small">
          {outstanding.length
            ? `还有 ${outstanding.length} 项风险未确认`
            : "写入追踪配置的两个指针，并记一条带证据的提升日志"}
        </span>
      </div>
      {error ? <p className="small cali-warn">{error}</p> : null}
    </div>
  );
}
