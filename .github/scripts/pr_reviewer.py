#!/usr/bin/env python3
"""
AI-powered first-pass PR reviewer for byjlw/video-analyzer.

No git checkout needed — the script fetches everything (config file, PR
metadata, diff) via the GitHub API. This means it works on fork PRs and
requires zero JS actions in the workflow.

Model swap: change the REVIEW_MODEL repo variable — no code edits needed.
Cost cap:   set a credit limit on the dedicated OPENROUTER_PR_REVIEW_KEY.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
from dataclasses import dataclass, field
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
CONFIG_API_PATH        = ".github/pr-reviewer-config.yml"
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

### Contributing philosophy
- Follow PEP 8; use type hints; keep functions focused.
- Add/update unit tests AND integration tests for any new behaviour.
- Update relevant .md docs and docstrings.
- Use conventional commit messages: feat/fix/docs/test/chore.
- Dependencies live in requirements.txt; conditional markers are fine for shims.
- Python version target: 3.11+.

### Code quality rules — flag any violation
1. SCOPE CREEP: The PR must only change what is necessary to achieve its stated
   goal. Flag any refactoring, style fixes, whitespace changes, or edits to code
   that is unrelated to the PR's purpose. Every changed line should be justified
   by the PR description.

2. TEST PLAN: The PR description must include a concrete test plan explaining how
   the author verified the change works and did not break existing functionality.
   Specific steps, commands, or scenarios are required — not just "I tested it".
   Flag if absent or vague.

3. NEW DEPENDENCIES: New packages in requirements.txt (or setup.py
   install_requires) must be genuinely necessary. Flag any dependency that
   duplicates existing functionality, could be replaced by the stdlib, or is
   added without clear justification in the PR description.

4. BACKWARDS COMPATIBILITY: The user-facing interface must not break.
   This includes:
   - CLI commands and flags (anything in cli.py or documented in docs/USAGES.md)
   - Config file keys and structure (config/config.json schema)
   - Output format and JSON schema (anything consumers of the JSON output rely on)
   Flag any change that renames, removes, or alters the behaviour of an existing
   CLI argument, config key, or output field — even if a new alternative is added.
   Deprecations are acceptable only if the old interface still works unchanged.
"""

SUMMARY_PROMPT = """
You are an expert open-source code reviewer for the video-analyzer project.
You will receive PR metadata and a unified diff.

Return a JSON object with exactly this shape — no prose outside the JSON:
{
  "summary": "<one concise paragraph describing what the PR does>",
  "recommendation": "APPROVE" | "REQUEST_CHANGES" | "DISCUSS",
  "recommendation_reason": "<one sentence>",
  "inline_comments": [
    {
      "path": "<file path>",
      "line": <integer, line number in the NEW file>,
      "body": "<markdown comment, be specific and constructive>"
    }
  ]
}

Rules for inline_comments:
- Flag real issues only: bugs, scope creep (unnecessary changes), missing or
  vague test plan, unjustified new dependencies, type hint omissions, doc gaps,
  and backwards compatibility breaks.
- DO NOT comment on style or formatting that is already consistent with the
  surrounding code, and DO NOT suggest changes to code not touched by this PR.
- If the PR description lacks a concrete test plan (specific steps/commands to
  verify the change works and didn't break anything), raise it as an inline
  comment on the first changed file.
- If a new dependency is added, verify it is justified. Flag it if the same
  thing could be done with the stdlib or an already-present package.
- If any changed lines appear to be unrelated style fixes or refactoring not
  mentioned in the PR description, flag them as scope creep.
- If any CLI argument, config key, or output field is renamed, removed, or
  changes its behaviour, flag it as a backwards compatibility break. The old
  interface must continue to work unchanged.
- Keep total inline_comments under 12.
- Return ONLY valid JSON. No markdown fences, no preamble.
"""


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ReviewerConfig:
    exclude_patterns: list[str] = field(default_factory=list)
    path_instructions: list[dict] = field(default_factory=list)


def load_config(repo) -> ReviewerConfig:
    """Fetch .github/pr-reviewer-config.yml from the default branch via API."""
    try:
        file = repo.get_contents(CONFIG_API_PATH)
        raw = yaml.safe_load(base64.b64decode(file.content).decode()) or {}
        return ReviewerConfig(
            exclude_patterns=raw.get("exclude_patterns", []),
            path_instructions=raw.get("path_instructions", []),
        )
    except Exception as e:
        print(f"Config not found or unreadable ({e}), using defaults.")
        return ReviewerConfig()


def path_instructions_for(filepath: str, config: ReviewerConfig) -> str:
    import fnmatch
    return "\n".join(
        rule["instructions"].strip()
        for rule in config.path_instructions
        if fnmatch.fnmatch(filepath, rule["path"] + "*")
        or filepath.startswith(rule["path"])
    )


# ── Diff helpers ──────────────────────────────────────────────────────────────

def is_excluded(filename: str, patterns: list[str]) -> bool:
    import fnmatch
    return any(fnmatch.fnmatch(filename, p) for p in patterns)


def last_reviewed_sha(pr: PullRequest) -> Optional[str]:
    for comment in reversed(list(pr.get_issue_comments())):
        if BOT_MARKER in comment.body:
            m = re.search(r"<!-- reviewed-sha:([0-9a-f]+) -->", comment.body)
            if m:
                return m.group(1)
    return None


def get_changed_files(pr: PullRequest, config: ReviewerConfig) -> tuple[list, int]:
    total_lines = 0
    result = []
    for f in pr.get_files():
        if is_excluded(f.filename, config.exclude_patterns):
            print(f"  Skipping excluded: {f.filename}")
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
        parts.append(f"```diff\n{f.patch}\n```" if f.patch
                     else "_(binary or no patch available)_")
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
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    resp = client.chat.completions.create(
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        messages=[
            {"role": "system", "content": SUMMARY_PROMPT + "\n\n" + PROJECT_PHILOSOPHY},
            {"role": "user",   "content": user_message},
        ],
        extra_headers={"X-Title": "video-analyzer PR reviewer"},
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


# ── GitHub posting ────────────────────────────────────────────────────────────

RECOMMENDATION_EMOJI = {
    "APPROVE":         "\u2705",
    "REQUEST_CHANGES": "\U0001f4dd",
    "DISCUSS":         "\U0001f4ac",
}


def existing_bot_comment_lines(pr: PullRequest) -> set[tuple[str, int]]:
    seen: set[tuple[str, int]] = set()
    for comment in pr.get_review_comments():
        if BOT_MARKER in (comment.body or ""):
            seen.add((comment.path, comment.original_line or comment.line or 0))
    return seen


def post_summary_comment(pr: PullRequest, review: dict,
                         model: str, head_sha: str, is_incremental: bool) -> None:
    emoji = RECOMMENDATION_EMOJI.get(review.get("recommendation", ""), "\U0001f916")
    note  = " _(incremental \u2014 only new commits reviewed)_" if is_incremental else ""
    body  = (
        f"{BOT_MARKER}\n"
        f"<!-- reviewed-sha:{head_sha} -->\n"
        f"## \U0001f916 Automated First-Pass Review{note}\n"
        f"_Model: `{model}` via OpenRouter. A human maintainer will follow up._\n\n"
        f"---\n\n"
        f"### Summary\n{review.get('summary', '_No summary generated._')}\n\n"
        f"### Recommendation\n"
        f"{emoji} **{review.get('recommendation', '?')}** \u2014 "
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
    commit = pr.base.repo.get_commit(head_sha)
    posted = 0
    for ic in inline_comments:
        path = ic.get("path", "")
        line = ic.get("line", 0)
        body = ic.get("body", "").strip()
        if not (path and line and body):
            continue
        if (path, line) in already_commented:
            print(f"  Skipping duplicate on {path}:{line}")
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
            print(f"  Could not post inline comment on {path}:{line}: {e}")
    print(f"  Posted {posted} inline comment(s).")


def post_too_large_comment(pr: PullRequest, total_lines: int, limit: int) -> None:
    body = (
        f"{BOT_MARKER}\n"
        f"## \U0001f916 Automated First-Pass Review\n\n"
        f"\u26a0\ufe0f **This PR is too large to review automatically** "
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

    gh   = Github(token)
    repo = gh.get_repo(repo_name)
    pr   = repo.get_pull(pr_number)

    # For workflow_dispatch the env vars are empty; fetch from API instead
    if not pr_title:
        pr_title  = pr.title
        pr_body   = pr.body or ""
        pr_author = pr.user.login
        head_sha  = pr.head.sha

    config = load_config(repo)
    print(f"Config: {len(config.exclude_patterns)} exclusions, "
          f"{len(config.path_instructions)} path rules")

    is_incremental = False
    if pr_action == "synchronize":
        since_sha = last_reviewed_sha(pr)
        if since_sha:
            is_incremental = True
            print(f"Incremental review since SHA {since_sha}")

    print(f"Model: {model} | Max diff lines: {max_lines}")
    print(f"Fetching diff for PR #{pr_number}...")
    files, total_lines = get_changed_files(pr, config)
    print(f"Files: {len(files)} | Lines changed: {total_lines}")

    if total_lines > max_lines:
        print(f"Too large ({total_lines} > {max_lines}), posting warning.")
        post_too_large_comment(pr, total_lines, max_lines)
        return

    if not files:
        print("No reviewable files after exclusions. Skipping.")
        return

    diff    = build_diff_text(files, config)
    message = build_user_message(pr_title, pr_author, pr_body, diff)

    print(f"Calling OpenRouter ({model})...")
    review = call_openrouter(message, model)

    already_commented = existing_bot_comment_lines(pr)
    inline_comments   = review.get("inline_comments", [])

    print("Posting summary comment...")
    post_summary_comment(pr, review, model, head_sha, is_incremental)

    print(f"Posting {len(inline_comments)} inline comment(s)...")
    post_inline_comments(pr, inline_comments, already_commented, head_sha)

    print("Done.")


if __name__ == "__main__":
    main()
