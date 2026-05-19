#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def feishu_sign(timestamp: str, secret: str) -> str:
    key = f"{timestamp}\n{secret}".encode("utf-8")
    digest = hmac.new(key, b"", digestmod=hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def load_event(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    event_path = Path(path)
    if not event_path.is_file():
        return {}
    return json.loads(event_path.read_text(encoding="utf-8"))


def short_sha(value: str | None) -> str:
    if not value:
        return "unknown"
    return value[:12]


def markdown_escape(value: Any) -> str:
    text = str(value if value is not None else "")
    return text.replace("\\", "\\\\").replace("*", "\\*").replace("_", "\\_").replace("`", "\\`")


def github_url(repo: str) -> str:
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
    return f"{server}/{repo}"


def github_api_json(path: str) -> Any | None:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return None
    api_url = f"{os.environ.get('GITHUB_API_URL', 'https://api.github.com').rstrip('/')}{path}"
    request = urllib.request.Request(
        api_url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "lerobot-feishu-branch-notifier",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        print(f"warning: GitHub API lookup failed: {exc}", file=sys.stderr)
        return None


def push_has_associated_pr(repo: str, sha: str) -> bool:
    pulls = github_api_json(f"/repos/{repo}/commits/{sha}/pulls")
    return isinstance(pulls, list) and len(pulls) > 0


def event_branch(event_name: str, event: dict[str, Any]) -> str:
    if event_name in {"pull_request", "pull_request_target"}:
        pull = event.get("pull_request") or {}
        base = pull.get("base") if isinstance(pull, dict) else {}
        return str(base.get("ref") or "")
    if event_name == "push":
        ref = str(event.get("ref") or "")
        return ref.removeprefix("refs/heads/")
    return os.environ.get("NOTIFY_BRANCH", "box")


def build_message(event_name: str, event: dict[str, Any], branch: str) -> dict[str, Any] | None:
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    repo_link = github_url(repo) if repo else ""

    if event_name in {"pull_request", "pull_request_target"}:
        action = str(event.get("action") or "")
        pull = event.get("pull_request") or {}
        if action not in {"opened", "reopened", "closed"}:
            return None
        merged = bool(pull.get("merged"))
        if action == "closed" and not merged:
            return None
        number = pull.get("number")
        title = str(pull.get("title") or "")
        html_url = str(pull.get("html_url") or repo_link)
        user = (pull.get("user") or {}).get("login") or "unknown"
        head = (pull.get("head") or {}).get("ref") or "unknown"
        merge_commit = pull.get("merge_commit_sha")
        verb = "merged into" if merged else "opened for"
        return {
            "title": f"PR {verb} origin/{branch}",
            "template": "green" if merged else "blue",
            "url": html_url,
            "lines": [
                f"**Repository**: [{markdown_escape(repo)}]({repo_link})",
                f"**PR**: [#{number} {markdown_escape(title)}]({html_url})",
                f"**Author**: {markdown_escape(user)}",
                f"**Branch**: `{markdown_escape(head)}` -> `origin/{markdown_escape(branch)}`",
                f"**Merge commit**: `{short_sha(str(merge_commit or ''))}`" if merged else "",
            ],
        }

    if event_name == "push":
        after = str(event.get("after") or os.environ.get("GITHUB_SHA") or "")
        if os.environ.get("SKIP_PUSH_WITH_ASSOCIATED_PR", "true").lower() in {"1", "true", "yes"}:
            if repo and after and push_has_associated_pr(repo, after):
                print(f"push {short_sha(after)} is associated with a PR; pull_request notification will cover it")
                return None
        commits = event.get("commits") if isinstance(event.get("commits"), list) else []
        head_commit = event.get("head_commit") if isinstance(event.get("head_commit"), dict) else {}
        compare = str(event.get("compare") or repo_link)
        pusher = (event.get("pusher") or {}).get("name") or "unknown"
        message = str(head_commit.get("message") or "")
        first_line = message.splitlines()[0] if message else "(no commit message)"
        return {
            "title": f"origin/{branch} updated",
            "template": "blue",
            "url": compare,
            "lines": [
                f"**Repository**: [{markdown_escape(repo)}]({repo_link})",
                f"**Branch**: `origin/{markdown_escape(branch)}`",
                f"**Pusher**: {markdown_escape(pusher)}",
                f"**Commits**: {len(commits)}",
                f"**Head**: `{short_sha(after)}`",
                f"**Latest**: {markdown_escape(first_line)}",
                f"**Compare**: [open diff]({compare})",
            ],
        }

    if event_name == "workflow_dispatch":
        run_url = f"{repo_link}/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}" if repo_link else ""
        return {
            "title": f"Feishu notifier test for origin/{branch}",
            "template": "blue",
            "url": run_url or repo_link,
            "lines": [
                f"**Repository**: [{markdown_escape(repo)}]({repo_link})" if repo_link else "**Repository**: local",
                f"**Branch**: `origin/{markdown_escape(branch)}`",
                "**Status**: manual test message",
            ],
        }

    return None


def build_feishu_payload(message: dict[str, Any], secret: str | None = None) -> dict[str, Any]:
    lines = [line for line in message["lines"] if line]
    payload: dict[str, Any] = {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "template": message.get("template", "blue"),
                "title": {"tag": "plain_text", "content": message["title"]},
            },
            "elements": [
                {"tag": "div", "text": {"tag": "lark_md", "content": "\n".join(lines)}},
            ],
        },
    }
    url = message.get("url")
    if url:
        payload["card"]["elements"].append(
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "Open in GitHub"},
                        "url": url,
                        "type": "primary",
                    }
                ],
            }
        )
    if secret:
        timestamp = str(int(time.time()))
        payload["timestamp"] = timestamp
        payload["sign"] = feishu_sign(timestamp, secret)
    return payload


def post_feishu(webhook_url: str, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        body = response.read().decode("utf-8")
        if response.status >= 300:
            raise RuntimeError(f"Feishu webhook returned HTTP {response.status}: {body}")
        try:
            result = json.loads(body)
        except json.JSONDecodeError:
            return
        code = result.get("code")
        status_code = result.get("StatusCode")
        if code not in (None, 0) or status_code not in (None, 0):
            raise RuntimeError(f"Feishu webhook returned error: {body}")


def main() -> int:
    event_name = os.environ.get("GITHUB_EVENT_NAME", "workflow_dispatch")
    event = load_event(os.environ.get("GITHUB_EVENT_PATH"))
    tracked_branch = os.environ.get("NOTIFY_BRANCH", "box")
    current_branch = event_branch(event_name, event)
    if current_branch and current_branch != tracked_branch:
        print(f"skip: event targets {current_branch}, not {tracked_branch}")
        return 0

    message = build_message(event_name, event, tracked_branch)
    if message is None:
        print(f"skip: no notification needed for event {event_name}")
        return 0

    secret = os.environ.get("FEISHU_WEBHOOK_SECRET") or None
    payload = build_feishu_payload(message, secret=secret)
    dry_run = os.environ.get("FEISHU_DRY_RUN", "").lower() in {"1", "true", "yes"}
    webhook_url = os.environ.get("FEISHU_WEBHOOK_URL")
    if dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if not webhook_url:
        print("FEISHU_WEBHOOK_URL is required unless FEISHU_DRY_RUN=true", file=sys.stderr)
        return 2
    post_feishu(webhook_url, payload)
    print(f"sent Feishu notification: {message['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
