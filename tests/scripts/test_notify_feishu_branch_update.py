from __future__ import annotations

import base64
import hashlib
import hmac

from scripts import notify_feishu_branch_update as notifier


def test_feishu_sign_uses_timestamp_newline_secret_key():
    timestamp = "1700000000"
    secret = "test-secret"
    expected = base64.b64encode(
        hmac.new(f"{timestamp}\n{secret}".encode("utf-8"), b"", digestmod=hashlib.sha256).digest()
    ).decode("utf-8")

    assert notifier.feishu_sign(timestamp, secret) == expected


def test_build_pull_request_merged_message(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "linkage-x/lerobot")
    event = {
        "action": "closed",
        "pull_request": {
            "number": 42,
            "title": "Add box policy",
            "html_url": "https://github.com/linkage-x/lerobot/pull/42",
            "merged": True,
            "merge_commit_sha": "1234567890abcdef",
            "user": {"login": "alice"},
            "base": {"ref": "box"},
            "head": {"ref": "feature/box-policy"},
        },
    }

    message = notifier.build_message("pull_request_target", event, "box")

    assert message is not None
    assert message["title"] == "PR merged into origin/box"
    assert any("#42 Add box policy" in line for line in message["lines"])
    assert any("1234567890ab" in line for line in message["lines"])


def test_build_push_message(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "linkage-x/lerobot")
    monkeypatch.setenv("SKIP_PUSH_WITH_ASSOCIATED_PR", "false")
    event = {
        "ref": "refs/heads/box",
        "after": "abcdef1234567890",
        "compare": "https://github.com/linkage-x/lerobot/compare/a...b",
        "pusher": {"name": "bob"},
        "commits": [{"id": "abcdef1234567890"}],
        "head_commit": {"message": "Merge branch feature"},
    }

    message = notifier.build_message("push", event, "box")

    assert message is not None
    assert message["title"] == "origin/box updated"
    assert any("abcdef123456" in line for line in message["lines"])
    assert any("Merge branch feature" in line for line in message["lines"])


def test_build_feishu_payload_adds_signature(monkeypatch):
    monkeypatch.setattr(notifier.time, "time", lambda: 1700000000)
    message = {
        "title": "origin/box updated",
        "template": "blue",
        "url": "https://github.com/linkage-x/lerobot",
        "lines": ["**Branch**: `origin/box`"],
    }

    payload = notifier.build_feishu_payload(message, secret="test-secret")

    assert payload["msg_type"] == "interactive"
    assert payload["timestamp"] == "1700000000"
    assert payload["sign"] == notifier.feishu_sign("1700000000", "test-secret")
