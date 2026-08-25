import { useState } from "react";
import type { DataCollectionGuiApi, GuiSnapshot } from "../api";
import type { MarkerTcpSample, MarkerTcpSession } from "../types";
import { Metric, StatusDot, stateLabel } from "../shared/ui";
import { deviceBoxIdentity } from "./adapters";
import { Modal } from "./ConfirmModal";

type BoxOption = { id: string; label: string };
type FormatHelp = "cad" | "static";

const CAD_EXAMPLE = `{
  "schema": "marker_rig_cad/v1",
  "source": "Onshape did=... wid=... eid=...",
  "units": "m",
  "cubes": {
    "left": {
      "device_id": "box1672693301",

      // 旋转来源（可选）：rig→TCP 的 4x4。单点 pivot 观测不到旋转，
      // 给了就用它，不给就从现有 production bundle 继承。
      "T_cube_tcp": [
        [ 0,  0,  1,  0.0000],
        [-1,  0,  0,  0.1019],
        [ 0, -1,  0,  0.0085],
        [ 0,  0,  0,  1.0000]
      ],

      // 刚体搬家（可选）：cube 本体系 → 生产 rig 系（CAD 系）的 4x4。
      // 不给 = rig 系就是 cube 系，解算只更新 translation。
      "T_rig_cube": [
        [1, 0, 0, 0.000],
        [0, 1, 0, 0.000],
        [0, 0, 1, 0.000],
        [0, 0, 0, 1.000]
      ]
    }
  }
}`;

const STATIC_EXAMPLE = `{
  "T_ee_cube": [
    [ 0,  0,  1,  0.0000],
    [-1,  0,  0,  0.1019],
    [ 0, -1,  0,  0.0085],
    [ 0,  0,  0,  1.0000]
  ],
  "num_samples_used": 412,
  "recording_dir": "outputs/datasets/tcp_marker_rig_calib_20260825_101530",
  "config": "third_party/opencv_kalibr/.../april_cube_tracking_in_robot_base_thor.yaml",
  "created_unix_s": 1787654321.0
}`;

function FormatHelpModal({ kind, onClose }: { kind: FormatHelp; onClose: () => void }) {
  const cad = kind === "cad";
  return (
    <Modal
      title={cad ? "CAD / 真值 JSON 格式" : "static_transform.json 格式"}
      onClose={onClose}
      footer={<button onClick={onClose}>关闭</button>}
    >
      {cad ? (
        <>
          <p className="cali-modal-lead">
            这个字段是<b>可选</b>的：留空时解算把 rig 系当成 cube 系，旋转从现有 production bundle 继承。
            填了它，两件事可以被 CAD 覆盖——rig→TCP 的<b>旋转</b>，和 cube 在生产 rig 系里的<b>刚体位姿</b>。
          </p>
          <ul>
            <li>路径可以是绝对路径，或相对于 gateway 仓库根目录。</li>
            <li>
              旋转键按顺序找：<code>cubes.&lt;cube&gt;.T_cube_tcp</code> → 根级
              <code>T_cube_tcp</code> / <code>T_marker_tcp</code> / <code>T_marker_to_tcp</code> /{" "}
              <code>T_rig_tcp</code>。
            </li>
            <li>
              刚体位姿键：<code>T_rig_cube</code>（或 <code>T_cube_rig</code>，会自动求逆）。非单位阵时解算会把
              layout 一起写进 EE 轨迹的 override 配置，否则生产 tracker 仍按解析 cube 角点跑，帧系对不上。
            </li>
            <li>
              全部是 <b>4×4 行主序</b>齐次矩阵，平移单位 <b>米</b>，最后一行必须是 <code>[0,0,0,1]</code>；旋转部分
              会校验正交且 det=+1。
            </li>
            <li>示例里的 <code>//</code> 注释只是说明，真实 JSON 不能带注释。</li>
          </ul>
          <pre className="marker-tcp-example">{CAD_EXAMPLE}</pre>
        </>
      ) : (
        <>
          <p className="cali-modal-lead">
            登记的是<b>已经解好</b>的 marker→TCP 常量，用来做重复性统计——不是原始录制。
            通常由 <code>evaluate/cube_tracking_teaching_error_eval.py</code> 写出，一次装夹一份。
          </p>
          <ul>
            <li>路径可以是绝对路径，或相对于 gateway 仓库根目录。</li>
            <li>
              必须至少包含 <code>T_ee_cube</code> / <code>T_marker_tcp</code> /{" "}
              <code>T_marker_to_tcp</code> / <code>T_cube_ee</code> 其中一个键，4×4 行主序，平移单位米。
            </li>
            <li>其余字段是溯源信息，缺了不影响登记，但建议保留。</li>
            <li>
              「条件」留空时用该文件所在目录名当条件名；重复性报告需要<b>至少 2 份</b>已登记结果。
            </li>
          </ul>
          <pre className="marker-tcp-example">{STATIC_EXAMPLE}</pre>
        </>
      )}
    </Modal>
  );
}

const EMPTY_SESSION: MarkerTcpSession = {
  active: false,
  sessionName: "",
  sessionRoot: "",
  stage: "idle",
  samples: [],
  pendingSampleId: "",
  message: "Marker→TCP repeatability session not started",
  reportPath: "",
  solvePath: "",
  solveSummaryPath: "",
  pivotReportPath: "",
  trackingRunPath: ""
};

const stageTone = (stage: string) => {
  if (stage === "failed") return "error";
  if (stage === "done") return "done";
  if (stage === "capture" || stage === "reporting") return "warning";
  return "idle";
};

/**
 * The boxes this session can attribute a marker→TCP sample to.
 *
 * Identity comes from `deviceBoxIdentity`, not `deviceBoxId`: on this rig the
 * single BOX is enumerated with an empty namespace by design, so keying the
 * dropdown on `box_id` left it permanently empty and every button disabled.
 * A row that carries no identity at all (static YAML rows, before Connect has
 * run discovery) is dropped, and the caller explains which case it is.
 */
function boxOptions(snapshot: GuiSnapshot): BoxOption[] {
  const seen = new Set<string>();
  const out: BoxOption[] = [];
  for (const device of snapshot.devices) {
    if (device.kind !== "box_collection") continue;
    const id = deviceBoxIdentity(device);
    if (!id) continue;
    if (seen.has(id)) continue;
    seen.add(id);
    out.push({ id, label: id });
  }
  return out;
}

/**
 * Whether the BOX rows on screen came from live discovery rather than the YAML.
 *
 * Before Connect the gateway publishes placeholder rows built from the static
 * config, which carry no serial and no device_id — so they cannot name a box,
 * and telling the operator "your BOX failed to report its serial" would be
 * wrong. Only once the recorder is up has `discover()` actually run.
 */
const CONNECTED_RECORDER_STATES = new Set([
  "connecting",
  "armed",
  "recording",
  "review",
  "saving",
  "discarding",
]);

function recorderConnected(snapshot: GuiSnapshot): boolean {
  return CONNECTED_RECORDER_STATES.has(snapshot.recording.state);
}

function sampleLabel(sample: MarkerTcpSample) {
  if (sample.staticTransformPath) return sample.staticTransformPath;
  if (sample.datasetRoot) {
    const episode = sample.episodeIndex >= 0 ? `episode ${sample.episodeIndex}` : "episode —";
    return `${sample.datasetRoot} · ${episode}`;
  }
  return "—";
}

function MarkerTcpCaptureGuide() {
  return (
    <div className="callout marker-tcp-guide">
      <b>采集指引</b>
      <ol>
        <li>
          使用专用 pivot 治具固定球窝/球座，夹爪夹持专用 TCP 夹具；球心尽量与目标 TCP 重合，若无法重合，记录球心到名义 TCP 的固定偏置。
        </li>
        <li>
          夹爪用本轮条件指定的 opening 和夹持力夹紧夹具，确认夹具不打滑、不翘起；相机与 marker rig 不许在本轮中被碰动。
        </li>
        <li>
          录制时让球心保持落座，只转动夹爪/腕部，不让球在座内滑移；围绕球心慢速扫 yaw、pitch、roll，覆盖大约正负 30 度以上的锥形姿态，并在极限姿态短暂停顿。
        </li>
        <li>
          运动全程保持 UMI marker 被至少 3 台相机看到；手、线缆和治具不要遮住 marker，画面里不要出现会反光或移动的无关物。
        </li>
        <li>
          重复性样本按条件分开命名：同一装夹至少 5 段，松开重夹至少 5 段；需要评估载荷时再单独采 light_push_x/y/z 和不同 opening，不要混在 same_mount 条件里。
        </li>
      </ol>
    </div>
  );
}

export function MarkerTcpPanel({
  snapshot,
  api,
  busy,
}: {
  snapshot: GuiSnapshot;
  api: DataCollectionGuiApi;
  busy: boolean;
}) {
  const [boxId, setBoxId] = useState("");
  const [condition, setCondition] = useState("same_mount_01");
  const [staticPath, setStaticPath] = useState("");
  const [cadPath, setCadPath] = useState("");
  const [socketBeyondTcpMm, setSocketBeyondTcpMm] = useState("0");
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  const [formatHelp, setFormatHelp] = useState<FormatHelp | null>(null);
  const session = snapshot.markerTcp ?? EMPTY_SESSION;
  const pendingSample = session.samples.find((sample) => sample.id === session.pendingSampleId);
  const options = boxOptions(snapshot);
  const selectedBoxId = options.length ? (options.some((option) => option.id === boxId) ? boxId : options[0].id) : "";
  const activeBoxId = pendingSample?.boxId || selectedBoxId;
  const hasBoxOptions = options.length > 0;
  const boxHint = hasBoxOptions
    ? ""
    : recorderConnected(snapshot)
      ? "BOX 已连接，但广播枚举没有给出 sn / device_id，无法给样本一个可追溯的身份。请断开后重新 Connect 让 discover() 再跑一次。"
      : "请先到「采集」页 Connect：现在列出的 BOX 行来自静态配置、没有设备身份，Connect 后才会换成广播枚举的结果。";
  const disabled = busy || pending;
  const savedForBox = session.samples.filter(
    (sample) => sample.status === "saved" && (sample.boxId || sample.side) === selectedBoxId
  );
  const canSolve = hasBoxOptions && savedForBox.length > 0 && !pendingSample;
  const canReport = session.samples.filter((sample) => sample.status === "registered" && sample.staticTransformPath).length >= 2;

  const call = async (fn: () => Promise<{ ok: boolean; error?: string }>) => {
    setPending(true);
    setError("");
    const result = await fn();
    setPending(false);
    if (!result.ok) setError(result.error || "操作失败");
  };

  return (
    <section className="panel marker-tcp-panel">
      <div className="panel-heading">
        <h2>UMI Marker→TCP 重复性</h2>
        <span className="state-pill">
          <StatusDot state={stageTone(session.stage)} />
          {stateLabel(session.stage)}
        </span>
      </div>

      {!session.active ? (
        <>
          <p className="panel-note">按 BOX ID 采集 UMI cube 的 marker→TCP 样本；不依赖 FR3，也不包含 head cube。</p>
          <MarkerTcpCaptureGuide />
          <div className="control-row">
            <button className="cali-btn-primary" disabled={disabled} onClick={() => call(() => api.startMarkerTcpSession())}>
              开始 UMI 采集会话
            </button>
          </div>
        </>
      ) : (
        <>
          <div className="summary-grid">
            <Metric label="会话" value={session.sessionName || "—"} />
            <Metric label="样本" value={session.samples.length} />
            <Metric label="录制器" value={snapshot.recording.state} />
            <Metric label="报告" value={session.reportPath || "—"} />
            <Metric label="生产 bundle" value={session.solvePath || "—"} />
          </div>

          <p className="panel-note">{session.message}</p>

          <MarkerTcpCaptureGuide />

          <div className="marker-tcp-controls">
            <label>
              BOX ID
              <select value={selectedBoxId} disabled={disabled || Boolean(pendingSample) || !hasBoxOptions} onChange={(event) => setBoxId(event.target.value)}>
                {hasBoxOptions ? (
                  options.map((option) => (
                    <option value={option.id} key={option.id}>{option.label}</option>
                  ))
                ) : (
                  <option value="">Connect 后显示 BOX ID</option>
                )}
              </select>
            </label>
            <label>
              条件
              <input
                value={condition}
                disabled={disabled || Boolean(pendingSample)}
                placeholder="same_mount_01"
                onChange={(event) => setCondition(event.target.value)}
              />
            </label>
          </div>

          {boxHint ? <p className="panel-note">{boxHint}</p> : null}

          <div className="control-row">
            {!pendingSample ? (
              <button
                className="cali-btn-primary"
                disabled={disabled || !condition.trim() || !hasBoxOptions}
                onClick={() => call(() => api.markerTcpRecordSample("start", selectedBoxId, condition))}
              >
                录制样本
              </button>
            ) : (
              <>
                <button className="cali-btn-primary" disabled={disabled} onClick={() => call(() => api.markerTcpRecordSample("save", activeBoxId, condition))}>
                  保存样本
                </button>
                <button disabled={disabled} onClick={() => call(() => api.markerTcpRecordSample("discard", activeBoxId, condition))}>
                  丢弃样本
                </button>
              </>
            )}
            <button disabled={disabled || Boolean(pendingSample)} onClick={() => call(() => api.cancelMarkerTcpSession())}>
              结束会话
            </button>
          </div>

          <div className="marker-tcp-controls marker-tcp-solve">
            <label>
              <span className="marker-tcp-label-row">
                CAD / 真值 JSON
                <button type="button" className="marker-tcp-help-btn" onClick={() => setFormatHelp("cad")}>
                  格式说明
                </button>
              </span>
              <input
                value={cadPath}
                disabled={disabled}
                placeholder="可选：含 T_cube_tcp / T_rig_cube 的 JSON，点「格式说明」看示例"
                onChange={(event) => setCadPath(event.target.value)}
              />
            </label>
            <label>
              球心→TCP 偏置 mm
              <input
                value={socketBeyondTcpMm}
                disabled={disabled}
                inputMode="decimal"
                placeholder="0"
                onChange={(event) => setSocketBeyondTcpMm(event.target.value)}
              />
            </label>
          </div>

          <div className="marker-tcp-solve-notes">
            <p className="panel-note">
              专用 pivot 治具的球心与 TCP_closed 重合，所以这里保持 <b>0</b>；只有换用球心不在 TCP 上的治具时才填非 0
              的固定偏置（沿手指轴，球心在 TCP 之外为正）。
            </p>
            <p className="panel-note">
              单点 pivot 只能观测 TCP 原点，观测不到旋转。留空 CAD/真值 JSON 时旋转从现有 production bundle
              继承，并写进 bundle 的 <code>rotation_source</code>；本次解算只更新 translation。
            </p>
          </div>

          <div className="control-row">
            <button
              className="cali-btn-primary"
              disabled={disabled || !canSolve}
              onClick={() => call(() => api.solveMarkerTcpTransform(selectedBoxId, cadPath, socketBeyondTcpMm))}
            >
              解算并写生产 bundle
            </button>
            <span className="panel-note">
              {canSolve
                ? `将用 ${selectedBoxId} 的 ${savedForBox.length} 段 saved 样本解算`
                : "需要至少 1 段已保存的 pivot 样本，且当前没有正在录制的段"}
            </span>
          </div>

          <div className="marker-tcp-controls marker-tcp-register">
            <label>
              <span className="marker-tcp-label-row">
                static_transform.json
                <button type="button" className="marker-tcp-help-btn" onClick={() => setFormatHelp("static")}>
                  格式说明
                </button>
              </span>
              <input
                value={staticPath}
                disabled={disabled}
                placeholder="outputs/.../static_transform.json，点「格式说明」看示例"
                onChange={(event) => setStaticPath(event.target.value)}
              />
            </label>
            <button disabled={disabled || !staticPath.trim() || !hasBoxOptions} onClick={() => call(() => api.registerMarkerTcpStaticTransform(staticPath, selectedBoxId, condition))}>
              登记结果
            </button>
            <button className="cali-btn-primary" disabled={disabled || !canReport} onClick={() => call(() => api.runMarkerTcpReport())}>
              生成报告
            </button>
          </div>

          {error ? <p className="panel-note error">{error}</p> : null}

          {session.samples.length ? (
            <div className="check-table calibration-table marker-tcp-table">
              {session.samples.map((sample) => (
                <div className="check-row" key={sample.id}>
                  <strong>
                    <StatusDot state={sample.status === "discarded" ? "error" : sample.status === "recording" ? "recording" : "running"} />
                    {sample.boxId || sample.side || "BOX"} · {sample.condition}
                  </strong>
                  <span>{sampleLabel(sample)}</span>
                  <em>{sample.status}</em>
                </div>
              ))}
            </div>
          ) : null}
        </>
      )}

      {formatHelp ? <FormatHelpModal kind={formatHelp} onClose={() => setFormatHelp(null)} /> : null}
    </section>
  );
}
