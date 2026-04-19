#!/usr/bin/env python3
"""
AI-powered first-pass PR reviewer for byjlw/video-analyzer.

Supports two backends:
  github_models  – FREE, uses GitHub Models API (gpt-4o-mini by default).
                   No extra secrets needed; GITHUB_TOKEN is enough.
  anthropic      – Paid, uses Claude via the Anthropic API.
                   Requires ANTHROPIC_API_KEY secret.

Cost-control features:
  - Concurrency lock in the workflow (only 1 review per PR at a time).
  - Bot author filter in the workflow (skip Dependabot etc.).
  - MAX_DIFF_LINES env var: diffs larger than this get a "please split" comment
    instead of an AI call, capping per-review token spend.
  - max_tokens capped at 1024 (output only; input is bounded by MAX_DIFF_LINES).

Additional hardening tip: set a monthly spend limit in your Anthropic console
at https://console.anthropic.com → Settings → Billing → Usage limits.
"""

from __future__ import annotations

import os
import sys
from typing import Optional
from github import Github

# ── Tuneable limits ─────────────────────────────────────────────────────────
DEFAULT_MAX_DIFF_LINES = 800   # ~25-40 KB of patch text; well within context
MAX_OUTPUT_TOKENS = 1024

# Models used per backend
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"  # Cheapest Claude; swap to claude-sonnet-4-6 for quality
GITHUB_MODELS_MODEL = "openai/gpt-4o-mini"     # Free tier on GitHub Models

# ── Project context ──────────────────────────────────────────────────────────
PROJECT_PHILOSOPHY = """
## Project: video-analyzer
A tool that analyzes videos using vision LLMs (Ollama / OpenAI-compatible APIs)
and OpenAI Whisper for audio. It can run fully locally or via cloud APIs.

### Architecture (3 stages)
1. Frame Extraction & Audio Processing – OpenCV frames + Whisper transcription
2. Frame Analysis – each frame sent to a vision LLM with rolling context
3. Video Reconstruction – frame analyses + transcript merged into a final description

### Contributing philosophy (from docs/CONTRIBUTING.md)
- Changes should be proposed in GitHub Discussions *before* opening a PR.
- PRs must reference the relevant discussion thread.
- Follow PEP 8; use type hints; keep functions focused.
- Add/update unit tests AND integration tests for any new behaviour.
- Update relevant .md docs and docstrings.
- Add an entry to CHANGELOG.md.
- Use conventional commit messages: feat/fix/docs/test/chore.
- Squash commits before merge if requested.
- Dependencies live in requirements.txt (read by setup.py);
  conditional markers (e.g. `; python_version >= "3.13"`) are fine for shims.
- Python version target: 3.11+ (setup.py says >=3.8 but README says 3.11+).

### Code style
- PEP 8, meaningful names, type hints on signatures, document complex logic.
- Tests for new features + edge cases.
"""

REVIEW_SYSTEM_PROMPT = """
You are an expert open-source code reviewer for the video-analyzer project.
Perform a helpful, constructive first-pass review of a pull request.

Your review must:
1. Open with a one-paragraph summary of what the PR does.
2. Evaluate alignment with the project philosophy and contributing guidelines.
3. Flag issues with: correctness, code quality, tests, documentation,
   dependency hygiene, commit style, missing CHANGELOG entry.
4. Call out any breaking changes.
5. Close with a RECOMMENDATION:
   ✅ APPROVE – looks good, merge when ready
   📝 REQUEST CHANGES – specific issues must be addressed first
   💬 DISCUSS – needs a design conversation before proceeding

Be concise, specific, and kind. A human maintainer will follow up.
Do NOT approve security-sensitive changes on your own.
Write clean Markdown. Keep total length under ~600 words.
"""


# ── Diff helpers ─────────────────────────────────────────────────────────────

def get_pr_diff(repo, pr_number: int) -> tuple[str, int]:
    """Return (diff_text, total_lines_changed)."""
    pr = repo.get_pull(pr_number)
    parts: list[str] = []
    total_lines = 0
    for f in pr.get_files():
        additions = f.additions or 0
        deletions = f.deletions or 0
        total_lines += additions + deletions
        parts.append(f"### {f.filename} ({f.status}, +{additions}/-{deletions})")
        if f.patch:
            parts.append(f"```diff\n{f.patch}\n```")
        else:
            parts.append("_(binary or no patch available)_")
    return "\n\n".join(parts), total_lines


def build_user_message(pr_title: str, pr_author: str, pr_body: str, diff: str) -> str:
    return (
        f"## Pull Request to review\n\n"
        f"**Title:** {pr_title}\n"
        f"**Author:** @{pr_author}\n\n"
        f"**Description:**\n{pr_body or '_No description provided._'}\n\n"
        f"---\n\n## Changed Files (unified diff)\n\n{diff}\n"
    )


# ── AI backends ──────────────────────────────────────────────────────────────

def call_anthropic(user_message: str) -> str:
    import anthropic  # only imported when needed
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=REVIEW_SYSTEM_PROMPT + "\n\n" + PROJECT_PHILOSOPHY,
        messages=[{"role": "user", "content": user_message}],
    )
    return resp.content[0].text


def call_github_models(user_message: str) -> str:
    """Use the GitHub Models inference endpoint – completely free."""
    from openai import OpenAI  # openai SDK is compatible with GitHub Models
    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=os.environ["GITHUB_MODELS_TOKEN"],
    )
    resp = client.chat.completions.create(
        model=GITHUB_MODELS_MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        messages=[
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT + "\n\n" + PROJECT_PHILOSOPHY},
            {"role": "user", "content": user_message},
        ],
    )
    return resp.choices[0].message.content


def call_ai(user_message: str, backend: str) -> str:
    if backend == "anthropic":
        return call_anthropic(user_message)
    return call_github_models(user_message)  # default: free


# ── GitHub comment helpers ────────────────────────────────────────────────────

BOT_MARKER = "<!-- ai-pr-review -->"


def post_comment(repo, pr_number: int, body: str, backend: str) -> None:
    pr = repo.get_pull(pr_number)
    model_label = ANTHROPIC_MODEL if backend == "anthropic" else GITHUB_MODELS_MODEL
    header = (
        f"{BOT_MARKER}\n"
        f"## 🤖 Automated First-Pass Review\n"
        f"_Generated by `{model_label}`. A human maintainer will follow up._\n\n"
        f"---\n\n"
    )
    for comment in pr.get_issue_comments():
        if BOT_MARKER in comment.body:
            comment.delete()
            break
    pr.create_issue_comment(header + body)


def post_too_large_comment(repo, pr_number: int, total_lines: int, limit: int) -> None:
    pr = repo.get_pull(pr_number)
    body = (
        f"{BOT_MARKER}\n"
        f"## 🤖 Automated First-Pass Review\n\n"
        f"⚠️ **This PR is too large to review automatically** "
        f"({total_lines} changed lines; limit is {limit}).\n\n"
        f"Please consider splitting it into smaller, focused PRs. "
        f"See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidance."
    )
    for comment in pr.get_issue_comments():
        if BOT_MARKER in comment.body:
            comment.delete()
            break
    pr.create_issue_comment(body)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    token      = os.environ.get("GITHUB_TOKEN")
    repo_name  = os.environ.get("REPO_FULL_NAME")
    pr_number  = int(os.environ.get("PR_NUMBER", "0"))
    pr_title   = os.environ.get("PR_TITLE", "")
    pr_body    = os.environ.get("PR_BODY", "")
    pr_author  = os.environ.get("PR_AUTHOR", "")
    backend    = os.environ.get("REVIEW_BACKEND", "github_models").strip().lower()
    max_lines  = int(os.environ.get("MAX_DIFF_LINES", str(DEFAULT_MAX_DIFF_LINES)))

    if not all([token, repo_name, pr_number]):
        print("Missing required environment variables.", file=sys.stderr)
        sys.exit(1)

    if backend == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is required when REVIEW_BACKEND=anthropic.", file=sys.stderr)
        sys.exit(1)

    gh   = Github(token)
    repo = gh.get_repo(repo_name)

    print(f"Backend: {backend} | Max diff lines: {max_lines}")
    print(f"Fetching diff for PR #{pr_number}…")
    diff, total_lines = get_pr_diff(repo, pr_number)
    print(f"Diff size: {total_lines} lines changed")

    if total_lines > max_lines:
        print(f"Diff too large ({total_lines} > {max_lines}). Posting size warning.")
        post_too_large_comment(repo, pr_number, total_lines, max_lines)
        return

    print(f"Calling AI ({backend})…")
    review = call_ai(build_user_message(pr_title, pr_author, pr_body, diff), backend)

    print("Posting comment…")
    post_comment(repo, pr_number, review, backend)
    print("Done.")


if __name__ == "__main__":
    main()
