import { useCallback, useEffect, useState } from "react";

import { api } from "../apiClient";
import type { SideCamCheckStatus } from "../types";
import { describePose, describeReference, formatTime, verdictBadgeClass } from "./sideCamCheck";

/** Has the side camera been bumped? Read off where the yellow tape square sits in its still.
 *
 * The tape is fixed to the table, so it only moves in the image when the camera does. The
 * reference defaults to a still from the training data's era (10-27), so "未移动" means "where
 * the recorded data saw it from", not merely "where it was this morning". Replacing the
 * reference is a second click because it erases exactly the difference this panel exists to show.
 *
 * What a move costs is stated modestly on purpose: the 10-29 swap probe found R4 reads position
 * off the ee image only, so a bumped side camera mostly misplaces the maps' backdrop and makes
 * new recordings inconsistent with old ones; it is not a reason to stop a policy run.
 */
export function SideCamCheckPanel({ cameraKey }: { cameraKey: string }) {
  const [status, setStatus] = useState<SideCamCheckStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [confirmReference, setConfirmReference] = useState(false);
  const [showImage, setShowImage] = useState(false);
  const [nonce, setNonce] = useState(0);

  const load = useCallback(async () => {
    const next = await api.fetchSideCamCheck(cameraKey);
    if (next) setStatus(next);
  }, [cameraKey]);

  useEffect(() => {
    void load();
  }, [load]);

  const runCheck = async () => {
    setBusy(true);
    setError("");
    const result = await api.runSideCamCheck(cameraKey);
    setBusy(false);
    if (!result.ok) {
      setError(result.error || "检查失败");
      return;
    }
    setStatus(result);
    setNonce(Date.now());
    if (result.result && result.result.verdict !== "ok") setShowImage(true);
  };

  const replaceReference = async () => {
    if (!confirmReference) {
      setConfirmReference(true);
      return;
    }
    setConfirmReference(false);
    setBusy(true);
    setError("");
    const result = await api.setSideCamReference(cameraKey);
    setBusy(false);
    if (!result.ok) {
      setError(result.error || "设置基准失败");
      return;
    }
    setStatus(result);
    setNonce(Date.now());
  };

  const last = status?.last ?? null;
  const earlier = (status?.history ?? []).slice(0, -1).reverse().slice(0, 5);

  return (
    <section className="card">
      <div className="card-head">
        <h3>侧相机位置自检</h3>
        {last && <span className={verdictBadgeClass(last.verdict)}>{last.verdictText}</span>}
      </div>
      <p className="hint">
        根据 <code>{cameraKey}</code> 画面里黄色胶带方框的位置判断相机有没有被碰动；齿轮和遮挡会被自动排除。
        {describeReference(status)}。
      </p>
      <div className="row-actions">
        <button type="button" onClick={() => void runCheck()} disabled={busy || !status?.reference}>
          {busy ? "检查中…" : "检查侧相机"}
        </button>
        {last && (
          <button type="button" onClick={() => setShowImage((v) => !v)}>
            {showImage ? "隐藏对比图" : "看对比图"}
          </button>
        )}
        <button
          type="button"
          onClick={() => void replaceReference()}
          onBlur={() => setConfirmReference(false)}
          disabled={busy}
          title="只在确认相机位置正确、且有意改用新位置时使用：替换后，与训练数据时期的差异就看不出来了。"
        >
          {confirmReference ? "确认：用当前画面替换基准" : "用当前画面作为基准"}
        </button>
      </div>
      {error && <div className="banner banner-error">{error}</div>}
      {last && (
        <div>
          <p>
            <strong>{formatTime(last.checkedAt)}</strong> {last.reason}
          </p>
          {last.pose && <p className="hint">{describePose(last)}</p>}
          {last.verdict === "shifted" && (
            <p className="hint">
              不用停实验：10-29 换图测试显示 R4 判断位置只看末端相机，侧视图几乎不参与。侧相机主要影响落点地图背景、
              peg 位置的人工核对，以及新录数据的侧视图一致性；方便时再调回（让对比图里红线和绿线重合）。
            </p>
          )}
          {last.verdict === "moved" && (
            <p className="hint">
              策略本身受影响不大（R4 只看末端相机），但录新数据前最好把相机调回（让对比图里红线和绿线重合）再检查一次，
              否则新数据的侧视图和以往不一致，落点地图背景也会错位。确实要换位置再替换基准。
            </p>
          )}
        </div>
      )}
      {showImage && last && (
        <figure style={{ margin: "8px 0", maxWidth: 640 }}>
          <img
            src={api.sideCamImageUrl(cameraKey, "overlay", nonce)}
            alt="侧相机胶带对比"
            style={{ width: "100%", height: "auto", display: "block" }}
          />
          <figcaption className="hint">红 = 基准里的胶带，绿 = 现在的胶带，黄 = 两者重合。</figcaption>
        </figure>
      )}
      {earlier.length > 0 && (
        <details>
          <summary className="hint">之前的检查</summary>
          <ul className="hint">
            {earlier.map((row, i) => (
              <li key={`${row.checkedAt}-${i}`}>
                {formatTime(row.checkedAt)} {row.verdictText}
                {row.pose ? `（${row.pose.shiftPx.toFixed(1)} px，${row.pose.rotDeg.toFixed(1)}°）` : ""}
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}
