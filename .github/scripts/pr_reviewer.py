#!/usr/bin/env python3
"""
AI-powered first-pass PR reviewer for byjlw/video-analyzer.

Key design decisions (informed by studying CodeRabbit and similar tools):

1. DEDICATED OpenRouter key with its own spend cap
   Create a separate key at openrouter.ai/settings/keys named
   "video-analyzer-pr-reviewer" and set a monthly credit limit.
   Store it as the repo secret OPENROUTER_PR_REVIEW_KEY — completely
   isolated from your personal OpenRouter usage.

2. Inline (line-level) review comments
   Rather than one big comment wall, the model is asked to return
   structured feedback with filename + line number so comments appear
   directly on the diff lines in GitHub's review UI.

3. Incremental reviews
   On `synchronize` events (new commits pushed to an open PR) the script
   only reviews files that changed since the last bot review commit SHA,
   saving tokens and reducing noise.

4. Per-path instructions
   .github/pr-reviewer-config.yml lets you attach extra instructions to
   specific paths (e.g. "this dir is security-critical") without touching
   Python code.

5. Deduplication
   Existing bot review comments on a line are not re-posted.

Model swap: change the REVIEW_MODEL repo variable — no code edits needed.
Cost cap:   set a credit limit on the dedicated OpenRouter key.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from openai import OpenAI
from github import Github
from github.PullRequest import PullRequest

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_MAX_DIFF_LINES = 800
MAX_OUTPUT_TOKENS      = 1500
OPENROUTER_BASE_URL    = "https://openrouter.ai/api/v1"
DEFAULT_MODEL          = "qwen/qwen3-235b-a22b"
CONFIG_PATH            = ".github/pr-reviewer-config.yml"
BOT_MARKER             = "<!-- ai-pr-review -->"

# ── Project context ───────────────────────────────────────────────────────────
PROJECT_PHILOSOPHY = """
## Project: video-analyzer
A tool that analyzes videos using vision LLMs (Ollama / OpenAI-compatible APIs)
and OpenAI Whisper for audio. It can run fully locally or via cloud APIs.

### Architecture (3 stages)
1. Frame Extraction & Audio Processing - OpenCV frames + Whisper transcription
2. Frame Analysis - each frame sent to a vision LLM with rolling context
3. Video Reconstruction - frame analyses + transcript merged into a final description

### Contributing philosophy (from docs/CONTRIBUTING.md)
- Changes should be proposed in GitHub Discussions *before* opening a PR.
- PRs must reference the relevant discussion thread.
- Follow PEP 8; use type hints; keep functions focused.
- Add/update unit tests AND integration tests for any new behaviour.
- Update relevant .md docs and docstrings.
- Add an entry to CHANGELOG.md.
- Use conventional commit messages: feat/fix/docs/test/chore.
- Dependencies live in requirements.txt; conditional markers are fine for shims.
- Python version target: 3.11+.

### Code style
- PEP 8, meaningful names, type hints on signatures, document complex logic.
- Tests for new features + edge cases.
"""

SUMMARY_PROMPT = """
You are an expert open-source code reviewer for the video-analyzer project.
You will receive PR metadata and a unified diff.

Return a JSON object with exactly this shape — no prose outside the JSON:
{
  "summary": "<one concise paragraph describing what the PR does>",
  "process_issues": ["<issue string>", ...],
  "recommendation": "APPROVE" | "REQUEST_CHANGES" | "DISCUSS",
  "recommendation_reason": "<one sentence>",
  "inline_comments": [
    {
      "path": "<file path>",
      "line": <integer, line number in the NEW file>,
      "body": "<markdown comment, be specific and constructive>"
    },
    ...
  ]
}

Rules:
- inline_comments should only cover real issues (bugs, style violations, missing
  tests, doc gaps). Omit praise or trivial nits unless they matter.
- process_issues covers things like missing CHANGELOG entry, no discussion link,
  wrong commit format — not code-level issues.
- Keep total inline_comments under 8 to avoid overwhelming the author.
- Return ONLY valid JSON. No markdown fences, no preamble.
"""


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ReviewerConfig:
    exclude_patterns: list[str] = field(default_factory=list)
    path_instructions: list[dict] = field(default_factory=list)


def load_config() -> ReviewerConfig:
    try:
        with open(CONFIG_PATH) as f:
            raw = yaml.safe_load(f) or {}
        return ReviewerConfig(
            exclude_patterns=raw.get("exclude_patterns", []),
            path_instructions=raw.get("path_instructions", []),
        )
    except FileNotFoundError:
        return ReviewerConfig()


def path_instructions_for(filepath: str, config: ReviewerConfig) -> str:
    """Return any extra instructions that apply to this file path."""
    import fnmatch
    extras = []
    for rule in config.path_instructions:
        if fnmatch.fnmatch(filepath, rule["path"] + "*") or \
           filepath.startswith(rule["path"]):
            extras.append(rule["instructions"].strip())
    return "\n".join(extras)


# ── Diff helpers ──────────────────────────────────────────────────────────────

def is_excluded(filename: str, patterns: list[str]) -> bool:
    import fnmatch
    return any(fnmatch.fnmatch(filename, p) for p in patterns)


def last_reviewed_sha(pr: PullRequest) -> Optional[str]:
    """Return the head SHA recorded in the most recent bot summary comment, or None."""
    for comment in reversed(list(pr.get_issue_comments())):
        if BOT_MARKER in comment.body:
            m = re.search(r"<!-- reviewed-sha:([0-9a-f]+) -->", comment.body)
            if m:
                return m.group(1)
    return None


def get_changed_files(pr: PullRequest, since_sha: Optional[str],
                      config: ReviewerConfig) -> tuple[list, int]:
    """
    Return (files, total_lines).
    If since_sha is set, only return files whose patch changed after that commit
    (incremental review). Otherwise return all files (first review).
    """
    all_files = list(pr.get_files())
    total_lines = 0
    result = []
    for f in all_files:
        if is_excluded(f.filename, config.exclude_patterns):
            print(f"  Skipping excluded file: {f.filename}")
            continue
        total_lines += (f.additions or 0) + (f.deletions or 0)
        result.append(f)
    return result, total_lines


def build_diff_text(files: list, config: ReviewerConfig) -> str:
    parts = []
    for f in files:
        extras = path_instructions_for(f.filename, config)
        header = f"### {f.filename} ({f.status}, +{f.additions or 0}/-{f.deletions or 0})"
        if extras:
            header += f"\n<!-- path-instructions: {extras} -->"
        parts.append(header)
        if f.patch:
            parts.append(f"```diff\n{f.patch}\n```")
        else:
            parts.append("_(binary or no patch available)_")
    return "\n\n".join(parts)


def build_user_message(pr_title: str, pr_author: str, pr_body: str, diff: str) -> str:
    return (
        f"## Pull Request\n"
        f"**Title:** {pr_title}\n"
        f"**Author:** @{pr_author}\n\n"
        f"**Description:**\n{pr_body or '_No description provided._'}\n\n"
        f"---\n\n## Diff\n\n{diff}\n"
    )


# ── AI call ───────────────────────────────────────────────────────────────────

def call_openrouter(user_message: str, model: str) -> dict:
    """Call OpenRouter and return parsed JSON review object."""
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    system = SUMMARY_PROMPT + "\n\n" + PROJECT_PHILOSOPHY
    resp = client.chat.completions.create(
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_message},
        ],
        extra_headers={"X-Title": "video-analyzer PR reviewer"},
    )
    raw = resp.choices[0].message.content.strip()
    # Strip markdown fences if the model adds them despite instructions
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


# ── GitHub posting helpers ─────────────────────────────────────────────────────

RECOMMENDATION_EMOJI = {
    "APPROVE":         "✅",
    "REQUEST_CHANGES": "📝",
    "DISCUSS":         "💬",
}


def existing_bot_comment_lines(pr: PullRequest) -> set[tuple[str, int]]:
    """Return set of (path, line) tuples already commented on by the bot."""
    seen: set[tuple[str, int]] = set()
    for comment in pr.get_review_comments():
        if BOT_MARKER in (comment.body or ""):
            seen.add((comment.path, comment.original_line or comment.line or 0))
    return seen


def post_summary_comment(pr: PullRequest, review: dict,
                          model: str, head_sha: str, is_incremental: bool) -> None:
    """Replace the previous bot summary comment with the new one."""
    emoji = RECOMMENDATION_EMOJI.get(review.get("recommendation", ""), "🤖")
    incremental_note = " _(incremental — only new commits reviewed)_" if is_incremental else ""
    process_issues = review.get("process_issues", [])
    process_block = ""
    if process_issues:
        items = "\n".join(f"- {i}" for i in process_issues)
        process_block = f"\n\n### Process issues\n{items}"

    body = (
        f"{BOT_MARKER}\n"
        f"<!-- reviewed-sha:{head_sha} -->\n"
        f"## 🤖 Automated First-Pass Review{incremental_note}\n"
        f"_Model: `{model}` via OpenRouter. A human maintainer will follow up._\n\n"
        f"---\n\n"
        f"### Summary\n{review.get('summary', '_No summary generated._')}"
        f"{process_block}\n\n"
        f"### Recommendation\n"
        f"{emoji} **{review.get('recommendation', '?')}** — "
        f"{review.get('recommendation_reason', '')}\n"
    )
    for comment in pr.get_issue_comments():
        if BOT_MARKER in comment.body:
            comment.delete()
            break
    pr.create_issue_comment(body)


def post_inline_comments(pr: PullRequest, inline_comments: list[dict],
                          already_commented: set[tuple[str, int]],
                          head_sha: str) -> None:
    """Post inline review comments, skipping lines already covered."""
    commit = pr.base.repo.get_commit(head_sha)
    posted = 0
    for ic in inline_comments:
        path = ic.get("path", "")
        line = ic.get("line", 0)
        body = ic.get("body", "").strip()
        if not (path and line and body):
            continue
        if (path, line) in already_commented:
            print(f"  Skipping duplicate comment on {path}:{line}")
            continue
        try:
            pr.create_review_comment(
                body=f"{BOT_MARKER}\n{body}",
                commit=commit,
                path=path,
                line=line,
            )
            posted += 1
        except Exception as e:
            # Line may not exist in diff; fall back gracefully
            print(f"  Could not post inline comment on {path}:{line}: {e}")
    print(f"  Posted {posted} inline comment(s).")


def post_too_large_comment(pr: PullRequest, total_lines: int, limit: int) -> None:
    body = (
        f"{BOT_MARKER}\n"
        f"## 🤖 Automated First-Pass Review\n\n"
        f"⚠️ **This PR is too large to review automatically** "
        f"({total_lines} changed lines; limit is {limit}).\n\n"
        f"Please split it into smaller, focused PRs. "
        f"See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidance."
    )
    for comment in pr.get_issue_comments():
        if BOT_MARKER in comment.body:
            comment.delete()
            break
    pr.create_issue_comment(body)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    token      = os.environ.get("GITHUB_TOKEN")
    repo_name  = os.environ.get("REPO_FULL_NAME")
    pr_number  = int(os.environ.get("PR_NUMBER", "0"))
    pr_title   = os.environ.get("PR_TITLE", "")
    pr_body    = os.environ.get("PR_BODY", "")
    pr_author  = os.environ.get("PR_AUTHOR", "")
    head_sha   = os.environ.get("PR_HEAD_SHA", "")
    pr_action  = os.environ.get("PR_ACTION", "opened")
    model      = os.environ.get("REVIEW_MODEL", DEFAULT_MODEL).strip()
    max_lines  = int(os.environ.get("MAX_DIFF_LINES", str(DEFAULT_MAX_DIFF_LINES)))

    if not all([token, repo_name, pr_number]):
        print("Missing required environment variables.", file=sys.stderr)
        sys.exit(1)
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_PR_REVIEW_KEY secret is not set.", file=sys.stderr)
        sys.exit(1)

    config = load_config()
    print(f"Loaded config: {len(config.exclude_patterns)} exclusions, "
          f"{len(config.path_instructions)} path rules")

    gh   = Github(token)
    repo = gh.get_repo(repo_name)
    pr   = repo.get_pull(pr_number)

    # Incremental: on synchronize, only review files new since last bot review
    since_sha: Optional[str] = None
    is_incremental = False
    if pr_action == "synchronize":
        since_sha = last_reviewed_sha(pr)
        if since_sha:
            is_incremental = True
            print(f"Incremental review since SHA {since_sha}")

    print(f"Model: {model} | Max diff lines: {max_lines}")
    print(f"Fetching diff for PR #{pr_number}...")
    files, total_lines = get_changed_files(pr, since_sha, config)
    print(f"Files to review: {len(files)} | Lines changed: {total_lines}")

    if total_lines > max_lines:
        print(f"Diff too large ({total_lines} > {max_lines}). Posting size warning.")
        post_too_large_comment(pr, total_lines, max_lines)
        return

    if not files:
        print("No reviewable files after exclusions. Skipping.")
        return

    diff = build_diff_text(files, config)
    message = build_user_message(pr_title, pr_author, pr_body, diff)

    print(f"Calling OpenRouter ({model})...")
    review = call_openrouter(message, model)

    already_commented = existing_bot_comment_lines(pr)
    inline_comments   = review.get("inline_comments", [])

    print(f"Posting summary comment...")
    post_summary_comment(pr, review, model, head_sha, is_incremental)

    print(f"Posting {len(inline_comments)} inline comment(s)...")
    post_inline_comments(pr, inline_comments, already_commented, head_sha)

    print("Done.")


if __name__ == "__main__":
    main()
