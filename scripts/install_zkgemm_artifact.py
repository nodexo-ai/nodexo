#!/usr/bin/env python3
"""Install the prebuilt zkgemm_cuda artifact for this Python/Torch ABI."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUDA_DIR = ROOT / "zkgemm" / "cuda"
USER_AGENT = "nodexo-artifact-installer/1.0"


def _read_json(path_or_url: str) -> object:
    if path_or_url.startswith(("http://", "https://")):
        req = urllib.request.Request(path_or_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    return json.loads(Path(path_or_url).read_text())


def _download(url: str, target: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        target.write_bytes(resp.read())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_sha256(path: Path, expected: str | None) -> None:
    if not expected:
        return
    actual = _sha256(path)
    if actual.lower() != expected.lower():
        raise SystemExit(
            f"SHA-256 mismatch for {path.name}\nexpected: {expected}\nactual:   {actual}"
        )


def _artifact_kind(artifact: dict) -> str:
    kind = str(artifact.get("kind") or artifact.get("type") or artifact.get("format") or "").lower()
    if kind in {"wheel", "whl"}:
        return "wheel"
    if kind in {"so", "shared_object", "shared-library", "shared_library"}:
        return "so"
    filename = str(artifact.get("filename") or artifact.get("name") or artifact.get("url") or "").lower()
    return "wheel" if filename.endswith(".whl") else "so"


def _host_tags() -> dict[str, str]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - install-time guard
        raise SystemExit(f"torch import failed while selecting zkgemm artifact: {exc}") from exc

    machine = platform.machine().lower()
    platform_tag = "linux_x86_64" if machine in {"amd64", "x86_64"} else f"{sys.platform}_{machine}"
    torch_version = str(torch.__version__).split("+", 1)[0]
    torch_major_minor = ".".join(torch_version.split(".")[:2])
    torch_cuda = torch.version.cuda or ""
    return {
        "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "platform": platform_tag,
        "torch": torch_version,
        "torch_major_minor": torch_major_minor,
        "cuda": torch_cuda,
        "cuda_tag": "cu" + torch_cuda.replace(".", "") if torch_cuda else "",
    }


def _norm(value: object) -> str:
    return str(value or "").strip()


def _matches(artifact: dict, tags: dict[str, str]) -> bool:
    platform_value = _norm(artifact.get("platform") or artifact.get("platform_tag") or tags["platform"])
    if platform_value and platform_value != tags["platform"]:
        return False

    python_value = _norm(artifact.get("python_abi") or artifact.get("python") or artifact.get("abi"))
    if python_value and python_value != tags["python_abi"]:
        return False

    torch_value = _norm(artifact.get("torch") or artifact.get("torch_version"))
    if torch_value:
        torch_value = torch_value.removeprefix("torch")
        if not (
            tags["torch"] == torch_value
            or tags["torch"].startswith(torch_value + ".")
            or tags["torch_major_minor"] == torch_value
        ):
            return False

    cuda_value = _norm(artifact.get("cuda") or artifact.get("cuda_tag") or artifact.get("torch_cuda"))
    if cuda_value:
        cuda_value = cuda_value.replace(".", "")
        wanted = {tags["cuda_tag"], tags["cuda"].replace(".", "")}
        if cuda_value not in wanted:
            return False

    return True


def _select_from_manifest(manifest_path_or_url: str) -> dict:
    manifest = _read_json(manifest_path_or_url)
    artifacts = manifest.get("artifacts", manifest if isinstance(manifest, list) else [])
    if not isinstance(artifacts, list):
        raise SystemExit("zkgemm manifest must contain an artifacts list")

    tags = _host_tags()
    matches = [a for a in artifacts if isinstance(a, dict) and _matches(a, tags)]
    matches.sort(key=lambda item: 0 if _artifact_kind(item) == "wheel" else 1)
    if matches:
        return matches[0]

    available = [
        {
            "platform": a.get("platform") or a.get("platform_tag"),
            "python": a.get("python_abi") or a.get("python") or a.get("abi"),
            "torch": a.get("torch") or a.get("torch_version"),
            "cuda": a.get("cuda") or a.get("cuda_tag") or a.get("torch_cuda"),
        }
        for a in artifacts
        if isinstance(a, dict)
    ]
    raise SystemExit(
        "no zkgemm artifact matches "
        f"platform={tags['platform']} python={tags['python_abi']} "
        f"torch={tags['torch_major_minor']} cuda={tags['cuda_tag']}; "
        f"available={available}"
    )


def _install_wheel(path: Path, sha256: str | None) -> None:
    _verify_sha256(path, sha256)
    for old in CUDA_DIR.glob("zkgemm_cuda*.so"):
        old.unlink()
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps", str(path)])


def _install_so(path: Path, sha256: str | None) -> Path:
    _verify_sha256(path, sha256)
    CUDA_DIR.mkdir(parents=True, exist_ok=True)
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    target = CUDA_DIR / f"zkgemm_cuda{suffix}"
    for old in CUDA_DIR.glob("zkgemm_cuda*.so"):
        old.unlink()
    shutil.copy2(path, target)
    target.chmod(0o755)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-url", default="")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--wheel-url", default="")
    parser.add_argument("--wheel-path", default="")
    parser.add_argument("--so-url", default="")
    parser.add_argument("--so-path", default="")
    parser.add_argument("--sha256", default="")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="nodexo_zkgemm_") as td:
        tmpdir = Path(td)
        if args.wheel_path:
            _install_wheel(Path(args.wheel_path), args.sha256)
            print(f"installed zkgemm_cuda wheel: {args.wheel_path}")
            return 0
        if args.wheel_url:
            path = tmpdir / Path(args.wheel_url.split("?", 1)[0]).name
            _download(args.wheel_url, path)
            _install_wheel(path, args.sha256)
            print(f"installed zkgemm_cuda wheel: {args.wheel_url}")
            return 0
        if args.so_path:
            target = _install_so(Path(args.so_path), args.sha256)
            print(f"installed zkgemm_cuda shared object: {target}")
            return 0
        if args.so_url:
            path = tmpdir / Path(args.so_url.split("?", 1)[0]).name
            _download(args.so_url, path)
            target = _install_so(path, args.sha256)
            print(f"installed zkgemm_cuda shared object: {target}")
            return 0

        manifest = args.manifest_path or args.manifest_url
        if not manifest:
            raise SystemExit(
                "provide --manifest-url/path, --wheel-url/path, or --so-url/path"
            )
        artifact = _select_from_manifest(manifest)
        url = _norm(artifact.get("url"))
        sha = _norm(artifact.get("sha256"))
        if not url:
            raise SystemExit("matching zkgemm artifact has no url")
        path = tmpdir / Path(url.split("?", 1)[0]).name
        _download(url, path)
        if _artifact_kind(artifact) == "wheel":
            _install_wheel(path, sha)
            print(f"installed zkgemm_cuda wheel: {url}")
        else:
            target = _install_so(path, sha)
            print(f"installed zkgemm_cuda shared object: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
