"""Git hygiene checks that prevent committing data and generated artifacts."""

import subprocess
import sys
from collections.abc import Sequence

FORBIDDEN_PREFIXES = (
    "data/raw/",
    "h-and-m-personalized-fashion-recommendations/",
    "artifacts/",
    "models/",
    "outputs/",
    "submissions/",
)

FORBIDDEN_ROOT_FILES = {
    "articles.csv",
    "customers.csv",
    "sample_submission.csv",
    "transactions_train.csv",
}

FORBIDDEN_SUFFIXES = (
    ".arrow",
    ".ckpt",
    ".faiss",
    ".feather",
    ".jpeg",
    ".jpg",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".png",
    ".pt",
    ".pth",
)


def tracked_files() -> list[str]:
    """Return files tracked by git.

    Returns:
        Repository-relative tracked paths.

    Raises:
        subprocess.CalledProcessError: If ``git ls-files`` fails.
    """

    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def is_forbidden_path(path: str) -> bool:
    """Return whether a tracked path violates artifact/data policy.

    Args:
        path: Repository-relative path reported by git.

    Returns:
        ``True`` when the path is a raw Kaggle file, generated artifact,
        model/index/checkpoint, image, or submission output.
    """

    return (
        path in FORBIDDEN_ROOT_FILES
        or path.startswith(FORBIDDEN_PREFIXES)
        or path.endswith(FORBIDDEN_SUFFIXES)
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run repository hygiene checks from the command line.

    Args:
        argv: Ignored command-line arguments reserved for future options.

    Returns:
        Process exit code: ``0`` when clean, ``1`` when forbidden tracked files
        are found.
    """

    del argv
    violations = [path for path in tracked_files() if is_forbidden_path(path)]

    if violations:
        print("Forbidden tracked data/artifact paths:")
        for path in violations:
            print(f"- {path}")
        return 1

    print("No forbidden raw data or generated artifact paths are tracked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
