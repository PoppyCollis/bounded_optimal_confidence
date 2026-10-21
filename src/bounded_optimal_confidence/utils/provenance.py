"""Which code produced a run: the short commit hash of this repository, '-dirty' when it has uncommitted changes."""
import subprocess

from bounded_optimal_confidence.paths import REPO_ROOT


def git_commit():
    """Short commit hash (+ '-dirty'), or 'unknown' if git is unavailable: the log must never kill a finished fit."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return git("rev-parse", "--short", "HEAD") + ("-dirty" if git("status", "--porcelain") else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
