#!/usr/bin/env python3
"""Explicit one-shot execution or interactive stages; never recover or retry."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


def predecessor(root, episode, phase):
    from runtime import sha
    needed = {'start-only': 'start-check', 'replay': 'start-only'}.get(phase)
    if needed is None:
        return None
    attempts = []
    for path in (root / 'logs').glob('operator_*/receipt.json'):
        receipt = json.loads(path.read_text())
        if receipt.get('episode') == episode and receipt.get('bundle_manifest_sha256') == sha(root / 'manifest.json'):
            attempts.append((receipt['created_unix_s'], path, receipt))
    if not attempts:
        raise RuntimeError(f'先成功运行 {needed}；没有本版本的起步测试记录。')
    _, path, receipt = max(attempts, key=lambda x: x[0])
    result_path = path.parent / 'execute/execution_result.json'
    if receipt.get('phase') != needed or not result_path.exists() or not 0 <= time.time() - receipt['created_unix_s'] <= 7200:
        raise RuntimeError(f'上一阶段必须是两小时内成功完成的 {needed}；失败后不得自动跳过。')
    result = json.loads(result_path.read_text())
    report_path = path.parent / 'execute/report.json'
    if not result.get('completed') or result.get('fault_code') != 0 or not report_path.exists():
        raise RuntimeError('上一阶段未成功完成；请先分析错误，不得直接继续。')
    report = json.loads(report_path.read_text())
    if 'error' in report or not report.get('execution', {}).get('completed'):
        raise RuntimeError('上一阶段报告包含错误。')
    return dict(receipt_path=str(path), receipt_sha256=sha(path),
                result_path=str(result_path), result_sha256=sha(result_path),
                report_path=str(report_path), report_sha256=sha(report_path))


def validate_release(root, path, prepared, prepared_sha256):
    from runtime import sha
    receipt = json.loads(path.read_text())
    required = dict(approved=True, prepared_sha256=prepared_sha256,
                    bundle_manifest_sha256=sha(root / 'manifest.json'),
                    phase=prepared['phase'], episode=prepared['episode'],
                    observer_confirmed=True, gripper_ready_mode_required=1,
                    automatic_mode_switching_allowed=False)
    if any(receipt.get(k) != v for k, v in required.items()):
        raise RuntimeError('Operator receipt does not match this exact plan/phase/bundle')
    if receipt.get('authorization_source') not in ('explicit_cli_execute', 'interactive_plan_confirmation'):
        raise RuntimeError('Unknown motion authorization source')
    if not 0 <= time.time() - receipt['created_unix_s'] <= 600:
        raise RuntimeError('Operator confirmation expired')
    # Exclude this new attempt when validating the required previous stage.
    previous = receipt.get('predecessor')
    needed = {'start-only': 'start-check', 'replay': 'start-only'}.get(prepared['phase'])
    if needed:
        if not previous:
            raise RuntimeError('Missing successful prerequisite')
        for kind in ('receipt', 'result', 'report'):
            if sha(previous[kind + '_path']) != previous[kind + '_sha256']:
                raise RuntimeError('Prerequisite log changed')
        prior = json.loads(Path(previous['receipt_path']).read_text())
        result = json.loads(Path(previous['result_path']).read_text())
        report = json.loads(Path(previous['report_path']).read_text())
        if (prior.get('phase') != needed or prior.get('episode') != prepared['episode']
                or prior.get('bundle_manifest_sha256') != required['bundle_manifest_sha256']
                or not 0 <= time.time() - prior['created_unix_s'] <= 7200
                or not result.get('completed') or result.get('fault_code') != 0
                or 'error' in report or not report.get('execution', {}).get('completed')):
            raise RuntimeError('Successful recent prerequisite missing')
    return receipt


def main():
    from runtime import save, sha, verify_bundle
    from replay_timed_candidate import evaluate
    import numpy as np
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('episode', type=int, choices=[0, 1])
    p.add_argument('phase', choices=['start-check', 'start-only', 'replay', 'full-replay'])
    p.add_argument('--execute', action='store_true',
                   help='明确启动所选动作；各阶段不再询问 RUN，不跳过检查、不重试。')
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    verify_bundle(root)
    if not a.execute and not sys.stdin.isatty():
        raise RuntimeError('交互模式需要终端；一次启动请显式使用 --execute，不使用管道自动输入确认。')
    (root / 'logs').mkdir(exist_ok=True)
    if a.phase == 'full-replay':
        confirmation = '本次 --execute 启动命令授权整段流程，不再逐段询问。' if a.execute else '各阶段按提示确认。'
        print('整段入口：5 秒小幅起步检查 → 到达首帧 → 整段重播。' + confirmation +
              '任何检查或执行失败立即结束，不恢复、不重试；不会切换夹爪模式。', flush=True)
        with (root / 'logs/full_operator.lock').open('a') as full_lock:
            fcntl.flock(full_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for phase in ('start-check', 'start-only', 'replay'):
                command = [sys.executable, str(Path(__file__).resolve()), str(a.episode), phase]
                if a.execute:
                    command.append('--execute')
                rc = subprocess.run(command).returncode
                if rc:
                    return rc
        return 0
    # Separate orchestration lock: runtime retains its exclusive device lock.
    with (root / 'logs/operator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = predecessor(root, a.episode, a.phase)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        output = root / 'logs' / f'operator_{stamp}_{uuid.uuid4().hex[:8]}'
        output.mkdir()
        print(f'日志目录：{output}\n正在只读检查并验证路径，此时不会运动。', flush=True)
        prep = output / 'prepare'
        rc = subprocess.run([sys.executable, str(root / 'runtime.py'), 'prepare',
                             '--episode', str(a.episode), '--phase', a.phase, '--output', str(prep)]).returncode
        if rc:
            raise RuntimeError(f'准备检查失败，未发出运动命令。查看 {prep}')
        plan_path = prep / 'prepared_plan.json'
        plan = json.loads(plan_path.read_text())
        if not plan['geometry']['sampled_geometry_pass']:
            raise RuntimeError(f'碰撞/桌面间隙检查未通过，禁止执行。查看 {plan_path}')
        ep = plan['execution_plan']
        start, end = evaluate(ep, 0.), evaluate(ep, ep['duration_s'])
        max_change = float(np.max(abs(end[:7] - start[:7])))
        print(f'\nEpisode {a.episode} / {a.phase}：预计 {ep["duration_s"]:.1f} 秒；'
              f'首尾最大关节变化 {np.rad2deg(max_change):.2f}°；'
              f'夹爪首尾 {start[7]*1000:.2f} → {end[7]*1000:.2f} mm。\n'
              '这是真机运动测试，不是仿真。桌面检查仅覆盖已测量的空桌面；'
              '请移开面包、容器及其他物品，人员离开运动范围，急停在手边。\n'
              '夹爪必须已经处于控制模式，程序不会切换模式；只按本阶段目标开合，不会清故障或恢复重播。\n'
              '软件检查不等于实体安全认证。出现异常立即急停。', flush=True)
        digest = sha(plan_path)
        if a.execute:
            print('已通过 --execute 授权本阶段，检查通过后执行；没有额外的 RUN 输入。', flush=True)
        else:
            phrase = f'RUN {digest[:8]}'
            if input(f'如已在现场确认，输入 {phrase} 执行本阶段；其他输入取消：').strip() != phrase:
                print('已取消，没有执行运动。')
                return 2
        receipt = dict(approved=True, prepared_sha256=digest, bundle_manifest_sha256=sha(root / 'manifest.json'),
                       phase=a.phase, episode=a.episode, observer_confirmed=True,
                       gripper_ready_mode_required=1, automatic_mode_switching_allowed=False,
                       authorization_source='explicit_cli_execute' if a.execute else 'interactive_plan_confirmation',
                       created_unix_s=time.time(), predecessor=prior)
        receipt_path = output / 'receipt.json'
        save(receipt_path, receipt)
        env = dict(os.environ, FRANKA_PHYSICAL_WATCHER_CONFIRMED='YES')
        rc = subprocess.run([sys.executable, str(root / 'runtime.py'), 'execute',
                             '--prepared', str(plan_path), '--confirm-plan-sha256', digest,
                             '--operator-release', str(receipt_path), '--output', str(output / 'execute')], env=env).returncode
        print(f'本阶段{"完成" if rc == 0 else "失败，禁止直接继续"}。日志：{output}', flush=True)
        return rc


if __name__ == '__main__':
    raise SystemExit(main())
