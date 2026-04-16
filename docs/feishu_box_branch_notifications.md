# Feishu Notifications For origin/box

This repository can notify a Feishu group whenever a PR is opened for `box`, a PR is merged into `box`, or `origin/box` receives a direct push.

## Files

- `.github/workflows/notify_box_branch_feishu.yml`: GitHub Actions workflow.
- `scripts/notify_feishu_branch_update.py`: zero-dependency Feishu webhook sender.
- `tests/scripts/test_notify_feishu_branch_update.py`: payload and signature tests.

## Create The Feishu Group Bot

Use a Feishu custom bot webhook. Feishu's official custom bot guide is:

```text
https://open.feishu.cn/document/client-docs/bot-v3/add-custom-bot
```

Recommended setup:

1. Open the target Feishu group in the desktop client.
2. Open group settings, then `群机器人`.
3. Choose `添加机器人`.
4. Choose `自定义机器人`.
5. Name it something explicit, for example `lerobot-box-branch`.
6. In security settings, enable `签名校验`.
7. Copy the `Webhook 地址`.
8. Copy the signing secret shown after enabling signature verification.
9. Save the bot configuration.

Avoid keyword-only security for this workflow. Signature verification is less brittle because the GitHub notification content can change while the shared secret stays stable.

## Add GitHub Secrets

In GitHub:

1. Open the repository page.
2. Go to `Settings` -> `Secrets and variables` -> `Actions`.
3. Add repository secret `FEISHU_WEBHOOK_URL`.
4. Paste the full Feishu webhook URL, for example:

```text
https://open.feishu.cn/open-apis/bot/v2/hook/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

5. Add repository secret `FEISHU_WEBHOOK_SECRET`.
6. Paste the signing secret copied from the Feishu bot security settings.

Do not commit either value to the repository.

## Enable The Workflow

GitHub only runs a workflow for branch events after the workflow file exists on the branch whose event is being evaluated. For tracking `origin/box`, make sure `.github/workflows/notify_box_branch_feishu.yml` is present on `box`.

After this change is merged or cherry-picked into `box`, the workflow sends notifications for:

- PR opened or reopened with base branch `box`.
- PR closed and merged into base branch `box`.
- Direct push to branch `box`.

PR events use `pull_request_target` so repository secrets are available even when the PR comes from a fork. The workflow checks out and runs the notifier from the base branch, not from the contributor's PR branch.

The push path skips commits that GitHub associates with a PR, so a normal PR merge should be reported by the PR event rather than duplicated by the push event.

## Test Manually

Use the workflow dispatch button in GitHub Actions:

1. Open `Actions`.
2. Select `Notify box branch updates to Feishu`.
3. Run workflow.
4. Keep `branch` as `box`.
5. Set `dry_run=false` to send a real message.

For a local dry run:

```bash
cd /home/hanyu/Codes/lerobot
FEISHU_DRY_RUN=true \
GITHUB_EVENT_NAME=workflow_dispatch \
GITHUB_REPOSITORY=linkage-x/lerobot \
GITHUB_RUN_ID=local \
python scripts/notify_feishu_branch_update.py
```

To test a real webhook locally:

```bash
cd /home/hanyu/Codes/lerobot
FEISHU_WEBHOOK_URL='https://open.feishu.cn/open-apis/bot/v2/hook/...' \
FEISHU_WEBHOOK_SECRET='...' \
GITHUB_EVENT_NAME=workflow_dispatch \
GITHUB_REPOSITORY=linkage-x/lerobot \
GITHUB_RUN_ID=local \
python scripts/notify_feishu_branch_update.py
```

## Operational Notes

- The webhook URL and signing secret are read only from environment variables or GitHub Secrets.
- The sender uses Feishu interactive card messages.
- If `FEISHU_WEBHOOK_URL` is missing and this is not a dry run, the script exits non-zero so workflow misconfiguration is visible.
- The workflow tracks branch name `box`, which corresponds to `origin/box` in local Git terminology.
