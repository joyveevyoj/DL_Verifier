import torch
import yaml
from pathlib import Path
from types import SimpleNamespace


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    return obj


def _resolve_path(path):
    p = Path(path)
    if p.is_absolute():
        return p
    # Try a few common locations so this works regardless of CWD:
    # 1) As given (relative to current working directory)
    # 2) Repo root
    # 3) Scripts/ (where configure.yaml currently lives)
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        Path.cwd() / p,
        repo_root / p,
        repo_root / "Scripts" / p,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Default (for helpful error messages)
    return candidates[-1]


def load_config(path):
    resolved = _resolve_path(path)
    with open(resolved, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    namespace = _to_namespace(data)
    if getattr(namespace, "DEVICE", "").lower() == "auto":
        namespace.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return namespace

