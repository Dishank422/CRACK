#!/usr/bin/env python3
"""
Evaluate CRACK against real code reviews on vllm PRs.

Usage:
    python eval/run_eval.py --repo-path /path/to/vllm --token ghp_xxx

The script:
1. Fetches PR metadata and review comments from GitHub
2. Identifies review rounds (commits with human/bot reviewer comments)
3. Replays history: checks out each commit, computes diffs, runs CRACK
4. Saves CRACK's output alongside actual human/bot comments for comparison
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import requests
from git import Repo

# Add CRACK to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CRACK.agent.reviewer import run_review, build_tool_context, build_tool_registry
from CRACK.agent.config import AgentConfig
from CRACK.agent.pr_context import PRContext
from CRACK.agent.models import ReviewResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# PRs to evaluate
PR_NUMBERS = [41043, 41024, 40950, 40338, 40412, 40273, 40133, 39986, 40982, 40538, 40531, 40845]

REPO_OWNER = "vllm-project"
REPO_NAME = "vllm"
REPO_FULL = f"{REPO_OWNER}/{REPO_NAME}"

# Known bot accounts (not human reviewers)
BOT_USERS = {"gemini-code-assist[bot]", "github-actions[bot]", "mergify[bot]"}


def github_get(url: str, token: str, params: dict | None = None) -> dict | list | None:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code == 200:
        return resp.json()
    logging.warning(f"GitHub API {resp.status_code} for {url}")
    return None


def github_get_all(url: str, token: str, params: dict | None = None) -> list:
    """Paginate through all results."""
    params = dict(params or {})
    params.setdefault("per_page", 100)
    all_items = []
    page = 1
    while True:
        params["page"] = page
        data = github_get(url, token, params)
        if not data:
            break
        all_items.extend(data)
        if len(data) < params["per_page"]:
            break
        page += 1
    return all_items


@dataclass
class ReviewComment:
    """A single inline review comment from GitHub."""
    user: str
    path: str
    line: int | None
    side: str
    body: str
    commit_id: str
    created_at: str
    is_bot: bool

    def to_dict(self):
        return asdict(self)


@dataclass
class ReviewRound:
    """A review round = a commit that has human/bot reviewer comments."""
    commit_sha: str
    commit_date: str
    comments: list[ReviewComment] = field(default_factory=list)

    def human_comments(self):
        return [c for c in self.comments if not c.is_bot]

    def bot_comments(self):
        return [c for c in self.comments if c.is_bot]


@dataclass
class PRData:
    """All data for one PR."""
    number: int
    title: str
    body: str
    author: str
    base_sha: str
    head_sha: str
    commits: list[dict]  # [{sha, message, date}]
    review_rounds: list[ReviewRound] = field(default_factory=list)


def fetch_pr_data(pr_number: int, token: str) -> PRData:
    """Fetch all data needed for evaluating one PR."""
    base = f"https://api.github.com/repos/{REPO_FULL}"

    # PR metadata
    pr = github_get(f"{base}/pulls/{pr_number}", token)
    if not pr:
        raise RuntimeError(f"Could not fetch PR #{pr_number}")

    # Commits
    raw_commits = github_get_all(f"{base}/pulls/{pr_number}/commits", token)
    commits = []
    for c in raw_commits:
        commits.append({
            "sha": c["sha"],
            "message": c["commit"]["message"].split("\n")[0],
            "date": c["commit"]["committer"]["date"],
        })

    # Inline review comments
    raw_comments = github_get_all(f"{base}/pulls/{pr_number}/comments", token)

    # Group comments by commit_id
    comments_by_commit: dict[str, list[ReviewComment]] = {}
    for rc in raw_comments:
        user = rc["user"]["login"]
        # Skip PR author's own replies (they're responses, not reviews)
        if user == pr["user"]["login"]:
            continue

        comment = ReviewComment(
            user=user,
            path=rc.get("path", ""),
            line=rc.get("line"),
            side=rc.get("side", "RIGHT"),
            body=rc.get("body", ""),
            commit_id=rc["commit_id"],
            created_at=rc["created_at"],
            is_bot=user in BOT_USERS,
        )
        comments_by_commit.setdefault(rc["commit_id"], []).append(comment)

    # Build review rounds: only commits that have reviewer comments
    review_rounds = []
    for commit_sha, comments in comments_by_commit.items():
        # Find commit date
        commit_date = ""
        for c in commits:
            if c["sha"] == commit_sha:
                commit_date = c["date"]
                break
        review_rounds.append(ReviewRound(
            commit_sha=commit_sha,
            commit_date=commit_date,
            comments=comments,
        ))

    # Sort rounds chronologically
    review_rounds.sort(key=lambda r: r.commit_date or r.comments[0].created_at)

    return PRData(
        number=pr_number,
        title=pr.get("title", ""),
        body=pr.get("body", "") or "",
        author=pr["user"]["login"],
        base_sha=pr["base"]["sha"],
        head_sha=pr["head"]["sha"],
        commits=commits,
        review_rounds=review_rounds,
    )


def compute_diff(repo: Repo, base_sha: str, head_sha: str) -> tuple[str, list[dict]]:
    """Compute diff and changed file list between two commits.

    Uses merge-base to get only the PR's changes, excluding commits
    that were merged into the base branch after the PR was created.
    """
    merge_base = repo.git.merge_base(base_sha, head_sha)
    diff_text = repo.git.diff(merge_base, head_sha)

    # Get changed files
    name_status = repo.git.diff("--name-status", merge_base, head_sha)
    changed_files = []
    for line in name_status.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        status_code = parts[0][0]  # First char: A, M, D, R
        path = parts[-1]  # Last part is the path (handles renames)
        status_map = {"A": "added", "M": "modified", "D": "deleted", "R": "renamed"}
        changed_files.append({
            "path": path,
            "status": status_map.get(status_code, "modified"),
        })

    return diff_text, changed_files


def build_pr_context_for_round(
    pr_data: PRData,
    round_idx: int,
    available_commits: set[str] | None = None,
) -> PRContext:
    """
    Build PRContext for a given review round.

    For round 0: just PR metadata and timeline up to this point.
    For round N: include previous rounds' comments as "previous review",
    set last_reviewed_commit for incremental review.
    """
    current_round = pr_data.review_rounds[round_idx]

    # Build timeline from commits and previous round comments
    events = []
    for c in pr_data.commits:
        # Only include commits up to this round's commit
        if c["date"] <= (current_round.commit_date or "9999"):
            events.append((c["date"], f"[commit {c['sha'][:8]}] \"{c['message']}\" by @{pr_data.author}"))

    # Include comments from previous rounds
    for prev_idx in range(round_idx):
        prev_round = pr_data.review_rounds[prev_idx]
        for comment in prev_round.comments:
            label = " (review bot)" if comment.is_bot else ""
            body_preview = comment.body[:150].replace("\n", " ")
            events.append((
                comment.created_at,
                f"[inline comment by @{comment.user}{label} on {comment.path}] {body_preview}"
            ))

    events.sort(key=lambda x: x[0])
    timeline = "\n".join(e[1] for e in events)

    ctx = PRContext(
        pr_title=pr_data.title,
        pr_body=pr_data.body,
        pr_author=pr_data.author,
        timeline=timeline,
    )

    # For rounds after the first, set up incremental review
    # Only reference previous rounds whose commits are available (not orphaned)
    if round_idx > 0:
        for prev_idx in range(round_idx - 1, -1, -1):
            prev_sha = pr_data.review_rounds[prev_idx].commit_sha
            if available_commits is None or prev_sha in available_commits:
                ctx.last_reviewed_commit = prev_sha
                break

    return ctx


def update_embedding_cache(repo: Repo, prev_sha: str | None, current_sha: str, cache_dir: str):
    """
    Compute which files changed between checkouts and set env var
    so the embedding tool knows what to re-embed.
    """
    if prev_sha is None:
        return  # Cold start, embedding tool handles this

    # Get files changed between the two checkouts
    try:
        changed = repo.git.diff("--name-only", prev_sha, current_sha).strip().split("\n")
        changed = [f for f in changed if f]
        if changed:
            logging.info(f"Embedding cache: {len(changed)} files changed since last checkout")
    except Exception as e:
        logging.warning(f"Could not compute changed files for embedding cache: {e}")


async def run_crack_review(
    repo: Repo,
    diff_text: str,
    changed_files: list[dict],
    pr_context: PRContext,
    pr_number: int,
    github_token: str,
    config: AgentConfig,
) -> ReviewResult:
    """Run CRACK's agent review."""
    return await run_review(
        repo_path=repo.working_tree_dir,
        diff_text=diff_text,
        changed_files=changed_files,
        github_token=github_token,
        github_repo=REPO_FULL,
        pr_number=pr_number,
        config=config,
        pr_context=pr_context,
    )


def save_round_result(
    output_dir: Path,
    pr_data: PRData,
    round_idx: int,
    review_round: ReviewRound,
    crack_result: ReviewResult | None,
    error: str | None = None,
):
    """Save results for one review round."""
    pr_dir = output_dir / f"pr_{pr_data.number}"
    pr_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "pr_number": pr_data.number,
        "pr_title": pr_data.title,
        "round": round_idx + 1,
        "commit": review_round.commit_sha,
        "commit_date": review_round.commit_date,
        "is_incremental": round_idx > 0,
        "human_comments": [c.to_dict() for c in review_round.human_comments()],
        "bot_comments": [c.to_dict() for c in review_round.bot_comments()],
    }

    if crack_result:
        result["crack_summary"] = crack_result.summary
        result["crack_event"] = crack_result.event
        result["crack_comments"] = [
            {
                "path": c.path,
                "start_line": c.start_line,
                "line": c.line,
                "side": c.side,
                "start_side": c.start_side,
                "body": c.body,
            }
            for c in crack_result.comments
        ]
    if error:
        result["error"] = error

    out_path = pr_dir / f"round_{round_idx + 1}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logging.info(f"Saved results to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CRACK against vllm PR reviews")
    parser.add_argument("--repo-path", required=True, help="Path to local vllm clone")
    parser.add_argument("--token", default=os.getenv("GITHUB_TOKEN"), help="GitHub token")
    parser.add_argument("--output-dir", default="eval/results", help="Output directory")
    parser.add_argument("--prs", nargs="*", type=int, default=PR_NUMBERS, help="PR numbers to evaluate")
    parser.add_argument("--skip-embeddings", action="store_true", help="Disable embedding tool")
    parser.add_argument("--model", default=None, help="Override LLM model (e.g., google-gla:gemini-3-flash-preview)")
    args = parser.parse_args()

    if not args.token:
        logging.error("GitHub token required. Pass --token or set GITHUB_TOKEN.")
        sys.exit(1)

    # Set env vars needed by CRACK's bootstrap (runs before our agent code)
    os.environ.setdefault("LLM_API_TYPE", "google")
    os.environ.setdefault("LLM_API_KEY", os.environ.get("GEMINI_API_KEY", "dummy"))
    os.environ.setdefault("MODEL", "gemini-3-flash-preview")
    os.environ.setdefault("CODE_CHECKS", "meta,optimality,modularity,exception_handling,testing,style")
    os.environ.setdefault("CRACK_AGENT_MAX_TOOL_CALLS", "100")
    os.environ.setdefault("CRACK_AGENT_MAX_REQUESTS", "120")

    repo = Repo(args.repo_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save original state to restore later
    original_ref = repo.head.commit.hexsha

    # Configure CRACK agent
    config = AgentConfig.from_env()
    if args.model:
        config.model = args.model
    if args.skip_embeddings:
        os.environ["CRACK_SKIP_EMBEDDINGS"] = "1"

    # Set embedding cache dir inside output
    os.environ["CRACK_EMBEDDINGS_DIR"] = str(output_dir / "embedding_cache")

    # Step 1: Fetch all PR data
    logging.info(f"Fetching data for {len(args.prs)} PRs...")
    all_pr_data: list[PRData] = []
    for pr_num in args.prs:
        logging.info(f"Fetching PR #{pr_num}...")
        pr_data = fetch_pr_data(pr_num, args.token)
        logging.info(
            f"  PR #{pr_num}: {pr_data.title} "
            f"({len(pr_data.review_rounds)} review rounds, "
            f"{sum(len(r.comments) for r in pr_data.review_rounds)} comments)"
        )
        all_pr_data.append(pr_data)

    # Step 2: Build chronological order of all (pr, round) pairs
    # Sort by commit date so embedding cache diffs are minimal
    eval_items: list[tuple[PRData, int]] = []
    for pr_data in all_pr_data:
        for round_idx in range(len(pr_data.review_rounds)):
            eval_items.append((pr_data, round_idx))

    eval_items.sort(
        key=lambda x: x[0].review_rounds[x[1]].commit_date
        or x[0].review_rounds[x[1]].comments[0].created_at
    )

    total_rounds = len(eval_items)
    logging.info(f"Total review rounds to evaluate: {total_rounds}")

    # Step 3: Cold-start embeddings at the oldest base SHA
    oldest_base = min(all_pr_data, key=lambda p: p.base_sha)
    # Actually sort by base commit date - we already have this from earlier
    # For simplicity, just use the first eval item's base
    if eval_items:
        first_pr = eval_items[0][0]
        logging.info(f"Checking out oldest base SHA {first_pr.base_sha[:8]} for embedding cold start...")
        repo.git.checkout(first_pr.base_sha, force=True)

    prev_checkout_sha = first_pr.base_sha if eval_items else None

    # Step 4: Process each review round
    for idx, (pr_data, round_idx) in enumerate(eval_items):
        review_round = pr_data.review_rounds[round_idx]
        commit_sha = review_round.commit_sha

        logging.info(
            f"\n{'='*60}\n"
            f"[{idx+1}/{total_rounds}] PR #{pr_data.number} round {round_idx+1} "
            f"(commit {commit_sha[:8]})\n"
            f"  {len(review_round.human_comments())} human comments, "
            f"{len(review_round.bot_comments())} bot comments\n"
            f"{'='*60}"
        )

        # Check if results already exist (resume support)
        result_path = output_dir / f"pr_{pr_data.number}" / f"round_{round_idx + 1}.json"
        if result_path.exists():
            logging.info(f"Results already exist at {result_path}, skipping.")
            prev_checkout_sha = commit_sha
            continue

        # Check if commit exists locally (force-pushed commits may be orphaned)
        try:
            repo.git.cat_file("-t", commit_sha)
        except Exception:
            logging.warning(f"Commit {commit_sha[:8]} not found locally (likely force-pushed), skipping.")
            save_round_result(output_dir, pr_data, round_idx, review_round, None, "orphaned commit")
            continue

        # Checkout the commit
        try:
            repo.git.checkout(commit_sha, force=True)
        except Exception as e:
            logging.error(f"Failed to checkout {commit_sha}: {e}")
            save_round_result(output_dir, pr_data, round_idx, review_round, None, str(e))
            continue

        # Compute diff against PR base
        try:
            diff_text, changed_files = compute_diff(repo, pr_data.base_sha, commit_sha)
        except Exception as e:
            logging.error(f"Failed to compute diff: {e}")
            save_round_result(output_dir, pr_data, round_idx, review_round, None, str(e))
            prev_checkout_sha = commit_sha
            continue

        if not diff_text.strip():
            logging.warning("Empty diff, skipping.")
            prev_checkout_sha = commit_sha
            continue

        logging.info(f"Diff: {len(changed_files)} changed files, {len(diff_text)} chars")

        # Build PR context
        available_commits = {c["sha"] for c in pr_data.commits}
        pr_context = build_pr_context_for_round(pr_data, round_idx, available_commits)

        # Compute incremental diff if this is a follow-up round
        if pr_context.last_reviewed_commit:
            try:
                pr_context.incremental_diff = repo.git.diff(
                    pr_context.last_reviewed_commit, commit_sha
                )
                logging.info(
                    f"Incremental diff from {pr_context.last_reviewed_commit[:8]}: "
                    f"{len(pr_context.incremental_diff)} chars"
                )
            except Exception as e:
                logging.warning(f"Could not compute incremental diff: {e}")

        # Run CRACK
        try:
            crack_result = asyncio.run(
                run_crack_review(
                    repo=repo,
                    diff_text=diff_text,
                    changed_files=changed_files,
                    pr_context=pr_context,
                    pr_number=pr_data.number,
                    github_token=args.token,
                    config=config,
                )
            )
            logging.info(
                f"CRACK result: {crack_result.event}, "
                f"{len(crack_result.comments)} inline comments"
            )
        except Exception as e:
            logging.error(f"CRACK review failed: {e}")
            crack_result = None
            save_round_result(output_dir, pr_data, round_idx, review_round, None, str(e))
            prev_checkout_sha = commit_sha
            continue

        # Save results
        save_round_result(output_dir, pr_data, round_idx, review_round, crack_result)
        prev_checkout_sha = commit_sha

    # Restore original state
    logging.info(f"\nRestoring repo to {original_ref[:8]}...")
    repo.git.checkout(original_ref, force=True)

    # Print summary
    logging.info("\n" + "=" * 60)
    logging.info("EVALUATION COMPLETE")
    logging.info("=" * 60)
    total_human = sum(
        len(r.human_comments())
        for pr_data in all_pr_data
        for r in pr_data.review_rounds
    )
    total_bot = sum(
        len(r.bot_comments())
        for pr_data in all_pr_data
        for r in pr_data.review_rounds
    )
    logging.info(f"PRs evaluated: {len(all_pr_data)}")
    logging.info(f"Review rounds: {total_rounds}")
    logging.info(f"Total human comments: {total_human}")
    logging.info(f"Total bot comments: {total_bot}")
    logging.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
