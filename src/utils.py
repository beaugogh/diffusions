from pathlib import Path
from datetime import datetime
from typing import Optional


def save_image(
    image,
    output_dir: str | Path,
    *,
    filename: Optional[str] = None,
    ext: str = "png",
    overwrite: bool = False,
) -> Path:
    """
    Save a PIL image to disk.

    Parameters
    ----------
    image
        PIL.Image.Image returned by Diffusers.

    output_dir : str | Path
        Directory to save the image into.

    filename : str, optional
        Filename without extension.
        If None, a timestamp-based name is generated.

    ext : str
        Image extension (e.g. "png", "jpg", "webp").

    overwrite : bool
        Whether to overwrite an existing file.

    Returns
    -------
    Path
        Path to the saved image.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    if not ext.startswith("."):
        ext = f".{ext}"

    path = output_dir / f"{filename}{ext}"

    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {path}")

    image.save(path)
    return path


def is_notebook() -> bool:
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is None:
            return False
        return ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False
