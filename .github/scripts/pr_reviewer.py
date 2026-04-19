#!/usr/bin/env python3
"""
AI-powered first-pass PR reviewer for byjlw/video-analyzer.

Reads the PR diff + metadata, then asks Claude to review it against
the project's contributing guidelines and design philosophy.
Posts the result as a PR comment.
"""

import os
import sys
import anthropic
from github import Github

# ── Project context baked in so the reviewer always has it ──────────────────
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
- Dependencies: the project uses requirements.txt (read by setup.py);
  conditional markers (e.g. `; python_version >= "3.13"`) are acceptable
  for compatibility shims.
- Python version target: 3.11+ (setup.py declares >=3.8 but README says 3.11+).

### Code style
- PEP 8
- Meaningful variable names
- Type hints on function signatures
- Document complex logic
- Tests for new features + edge cases
"""

REVIEW_SYSTEM_PROMPT = """
You are an expert open-source code reviewer for the video-analyzer project.
Your job is to perform a helpful, constructive first-pass review of a pull request.

You will be given:
- PR metadata (title, description, author)
- The unified diff of all changed files
- The project's philosophy and contributing guidelines

Your review should:
1. Start with a short summary of what the PR does.
2. Evaluate whether it aligns with the project philosophy and contributing guidelines.
3. Highlight any issues with: correctness, code quality, tests, documentation,
   dependency hygiene, commit style, or missing CHANGELOG entry.
4. Call out anything that looks like a breaking change.
5. End with a clear RECOMMENDATION — one of:
   ✅ APPROVE – looks good, merge when ready
   📝 REQUEST CHANGES – specific issues must be addressed first
   💬 DISCUSS – needs a design conversation before proceeding

Be concise, specific, and kind. This is a first-pass automated review;
a human maintainer will follow up. Do NOT approve security-sensitive changes.
Format your review in clean Markdown. Keep total length under ~600 words.
"""


def get_pr_diff(repo, pr_number: int) -> str:
    """Fetch the unified diff for a PR via PyGithub."""
    pr = repo.get_pull(pr_number)
    files = pr.get_files()
    diff_parts = []
    for f in files:
        diff_parts.append(f"### {f.filename} ({f.status}, +{f.additions}/-{f.deletions})")
        if f.patch:
            diff_parts.append(f"```diff\n{f.patch}\n```")
        else:
            diff_parts.append("_(binary or no patch available)_")
    return "\n\n".join(diff_parts)


def build_user_message(pr_title: str, pr_author: str, pr_body: str, diff: str) -> str:
    return f"""## Pull Request to review

**Title:** {pr_title}
**Author:** @{pr_author}

**Description:**
{pr_body or '_No description provided._'}

---

## Changed Files (unified diff)

{diff}
"""


def call_claude(user_message: str) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=REVIEW_SYSTEM_PROMPT + "\n\n" + PROJECT_PHILOSOPHY,
        messages=[{"role": "user", "content": user_message}],
    )
    return message.content[0].text


def post_comment(repo, pr_number: int, body: str) -> None:
    pr = repo.get_pull(pr_number)
    header = (
        "<!-- ai-pr-review -->\n"
        "## 🤖 Automated First-Pass Review\n"
        "_This review was generated automatically by Claude. "
        "A human maintainer will follow up._\n\n"
        "---\n\n"
    )
    # Delete any previous bot comment so we don't stack duplicates on re-runs
    for comment in pr.get_issue_comments():
        if "<!-- ai-pr-review -->" in comment.body:
            comment.delete()
            break
    pr.create_issue_comment(header + body)


def main():
    token = os.environ.get("GITHUB_TOKEN")
    repo_name = os.environ.get("REPO_FULL_NAME")
    pr_number = int(os.environ.get("PR_NUMBER", "0"))
    pr_title = os.environ.get("PR_TITLE", "")
    pr_body = os.environ.get("PR_BODY", "")
    pr_author = os.environ.get("PR_AUTHOR", "")

    if not all([token, repo_name, pr_number]):
        print("Missing required environment variables.", file=sys.stderr)
        sys.exit(1)

    gh = Github(token)
    repo = gh.get_repo(repo_name)

    print(f"Fetching diff for PR #{pr_number}…")
    diff = get_pr_diff(repo, pr_number)

    print("Calling Claude for review…")
    review = call_claude(build_user_message(pr_title, pr_author, pr_body, diff))

    print("Posting comment…")
    post_comment(repo, pr_number, review)
    print("Done.")


if __name__ == "__main__":
    main()
