from pathlib import Path
from typing import Optional, Iterable
from huggingface_hub import snapshot_download


def download_hf_repo(
    repo_id: str,
    target_dir: str | Path,
    *,
    revision: Optional[str] = None,
    repo_type: str = "model",          # "model", "dataset", "space"
    allow_patterns: Optional[Iterable[str]] = None,
    ignore_patterns: Optional[Iterable[str]] = None,
    resume: bool = True,
) -> Path:
    """
    Download a Hugging Face repository into:

        <target_dir>/<namespace>/<repo_name>/

    Example:
        repo_id    = "black-forest-labs/FLUX.1-dev"
        target_dir = "/models"

        Result:
        /models/black-forest-labs/FLUX.1-dev/

    Parameters
    ----------
    repo_id : str
        Hugging Face repository id, in the form "namespace/name".

    target_dir : str | Path
        Parent directory under which the repository folder will be created.

    revision : str, optional
        Branch, tag, or commit hash for reproducibility.

    repo_type : str
        One of: "model", "dataset", "space".

    allow_patterns / ignore_patterns :
        Optional glob filters.

    resume : bool
        Resume interrupted downloads.

    Returns
    -------
    Path
        Path to the downloaded repository directory.
    """
    target_dir = Path(target_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    # Validate and split repo_id
    try:
        namespace, repo_name = repo_id.split("/", 1)
    except ValueError:
        raise ValueError(
            f"Invalid repo_id '{repo_id}'. Expected format 'namespace/name'."
        )

    repo_dir = target_dir / namespace / repo_name
    repo_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=repo_dir,
        local_dir_use_symlinks=False,   # critical for portability
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        resume_download=resume,
    )

    return repo_dir



if __name__ == "__main__":
    # repo_id = "black-forest-labs/FLUX.1-schnell"
    repo_id = "black-forest-labs/FLUX.1-dev"
    target_dir = "/home/bo/workspace/models"
    # Example usage
    download_hf_repo(
        repo_id=repo_id,
        target_dir=target_dir,
    )
 