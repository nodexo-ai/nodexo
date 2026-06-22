"""Curated container images for renter-facing presets.

This module is intentionally stdlib-only so setup scripts can import it before
the Python virtualenv is installed. Operators can override the catalog with:

  NODEXO_IMAGE_CATALOG=image1,image2,...
  NODEXO_REQUIRED_RENTAL_IMAGES=image1,image2,...
  NODEXO_IMAGE_PULL_RESERVE_GB=image=gb,image2=gb2,...
  CANARY_IMAGES=image1,image2,...

The catalog is what miners report in heartbeat cache status. Required images
are pulled/verified before a production miner advertises rentable capacity.
"""
from __future__ import annotations

import os

PYTORCH_RUNTIME = "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime"
PYTORCH_DEVEL = "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel"
CUDA_RUNTIME = "nvidia/cuda:12.8.2-runtime-ubuntu22.04"
CUDA_DEVEL = "nvidia/cuda:12.8.2-devel-ubuntu22.04"
VLLM_OPENAI = "vllm/vllm-openai:v0.21.0-cu129-ubuntu2404"
UBUNTU_2204 = "ubuntu:22.04"

DEFAULT_WARM_RENTAL_IMAGES = (
    PYTORCH_RUNTIME,
    CUDA_RUNTIME,
    UBUNTU_2204,
)

DEFAULT_CANARY_IMAGES = (
    PYTORCH_RUNTIME,
)

DEFAULT_RENTAL_IMAGE_CATALOG = (
    *DEFAULT_WARM_RENTAL_IMAGES,
    PYTORCH_DEVEL,
    CUDA_DEVEL,
    VLLM_OPENAI,
)

DEFAULT_IMAGE_PULL_RESERVE_GB = {
    UBUNTU_2204: 1,
    CUDA_RUNTIME: 5,
    CUDA_DEVEL: 12,
    PYTORCH_RUNTIME: 10,
    PYTORCH_DEVEL: 22,
    VLLM_OPENAI: 40,
}

DEFAULT_UNKNOWN_IMAGE_PULL_RESERVE_GB = 20

# Registry digests for the curated public presets. These protect the default
# catalog from mutable tag drift and local retags. Operators who override the
# catalog can add matching entries with NODEXO_IMAGE_DIGESTS.
DEFAULT_IMAGE_DIGESTS = {
    PYTORCH_RUNTIME: "sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca",
    PYTORCH_DEVEL: "sha256:b574d4ccf6d8856a5d87dcadc667aa4f95dc18d337ef3a28d02b7b01897d7081",
    CUDA_RUNTIME: "sha256:74b1c7a184bf45d80c2991b3868d4a9188ab3fc5e118cff0f5065e20781d9c6b",
    CUDA_DEVEL: "sha256:4c3d61ac8217a4933cc3855ddd0a75e2a058b6757a170993040272f4310ef113",
    VLLM_OPENAI: "sha256:ee4aa7089fa71510297b404e13e026c7adfffea997be05a34accfd0cdad9fd9b",
    UBUNTU_2204: "sha256:4f838adc7181d9039ac795a7d0aba05a9bd9ecd480d294483169c5def983b64d",
}


def _csv(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _dedupe(items: list[str] | tuple[str, ...], *, limit: int = 64) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def rental_image_catalog() -> list[str]:
    """Images whose local cache status miners report to validators."""
    return _dedupe(_csv("NODEXO_IMAGE_CATALOG") or DEFAULT_RENTAL_IMAGE_CATALOG)


def required_rental_images() -> list[str]:
    """Images a production miner should have cached before registering."""
    return _dedupe(_csv("NODEXO_REQUIRED_RENTAL_IMAGES") or DEFAULT_WARM_RENTAL_IMAGES)


def canary_image_catalog() -> list[str]:
    """Images the validator may use for canaries.

    The canary script requires Python, torch, and numpy inside the container.
    By default we use required PyTorch images only; operators can widen the
    pool with CANARY_IMAGES when those images are also canary-compatible.
    """
    configured = _csv("CANARY_IMAGES")
    if configured:
        return _dedupe(configured)
    required = [
        image for image in required_rental_images()
        if "pytorch" in image.lower()
    ]
    return _dedupe(required or DEFAULT_CANARY_IMAGES)


def image_digest_from_reference(image: str) -> str | None:
    """Return sha256 digest embedded in an image reference, if present."""
    ref = (image or "").strip()
    if "@sha256:" not in ref:
        return None
    digest = ref.rsplit("@", 1)[1].strip().lower()
    return digest if digest.startswith("sha256:") else None


def image_catalog_reference(image: str) -> str | None:
    """Return the catalog tag matching a tag or digest-pinned reference."""
    ref = (image or "").strip()
    catalog = rental_image_catalog()
    if ref in catalog:
        return ref
    digest = image_digest_from_reference(ref)
    if not digest:
        return None
    for candidate in catalog:
        expected = image_expected_digest(candidate)
        if expected and expected.lower() == digest:
            return candidate
    return None


def image_in_catalog(image: str) -> bool:
    """True when `image` is in the curated renter image catalog."""
    return image_catalog_reference(image) is not None


def image_canary_compatible(image: str) -> bool:
    """True when the image should contain Python, torch, and numpy."""
    ref = image_catalog_reference(image) or (image or "").strip()
    return "pytorch" in ref.lower()


def _digest_overrides() -> dict[str, str | None]:
    """Parse optional digest overrides.

    Format:

      NODEXO_IMAGE_DIGESTS=image=sha256:abc,image2=sha256:def

    Set a value to "none" or "off" to explicitly disable pinning for an
    overridden catalog image.
    """
    raw = os.environ.get("NODEXO_IMAGE_DIGESTS", "").strip()
    out: dict[str, str | None] = {}
    if not raw:
        return out
    for item in raw.split(","):
        if "=" not in item:
            continue
        key, value = item.rsplit("=", 1)
        key = key.strip()
        value = value.strip().lower()
        if not key:
            continue
        if value in {"", "none", "off", "0", "false"}:
            out[key] = None
        elif value.startswith("sha256:"):
            out[key] = value
    return out


def image_expected_digest(image: str) -> str | None:
    """Return the approved digest for a catalog image, if one is configured."""
    ref = (image or "").strip()
    embedded = image_digest_from_reference(ref)
    if embedded:
        return embedded
    overrides = _digest_overrides()
    if ref in overrides:
        return overrides[ref]
    return DEFAULT_IMAGE_DIGESTS.get(ref)


def image_runtime_reference(image: str) -> str:
    """Return a digest-pinned reference for Docker run/pull when available."""
    ref = (image or "").strip()
    digest = image_expected_digest(ref)
    if not digest or "@" in ref:
        return ref
    repo = ref.rsplit(":", 1)[0] if ":" in ref else ref
    return f"{repo}@{digest}"


def image_digest_matches(image: str, repo_digests: list[str] | tuple[str, ...]) -> bool:
    """True when Docker inspect repo digests contain the approved digest.

    Docker reports values as repo@sha256:..., while the catalog stores only
    sha256:... so the check is repository-name agnostic.
    """
    expected = image_expected_digest(image)
    if not expected:
        return True
    wanted = expected.lower()
    return any(str(item).lower().endswith(wanted) for item in repo_digests or [])


def image_pull_reserve_gb(image: str) -> int:
    """Estimated free Docker storage needed to safely cold-pull an image.

    This is a conservative placement reserve, not a storage cap. Operators can
    override per image with:

      NODEXO_IMAGE_PULL_RESERVE_GB=ubuntu:22.04=1,vllm/...=40

    A plain integer override applies to unknown images when custom cold images
    are explicitly enabled.
    """
    ref = image_catalog_reference(image) or (image or "").strip()
    override = os.environ.get("NODEXO_IMAGE_PULL_RESERVE_GB", "").strip()
    unknown_default = DEFAULT_UNKNOWN_IMAGE_PULL_RESERVE_GB
    if override:
        if "=" not in override:
            try:
                unknown_default = max(0, int(override))
            except ValueError:
                pass
        else:
            for item in override.split(","):
                if "=" not in item:
                    continue
                key, value = item.rsplit("=", 1)
                if key.strip() != ref:
                    continue
                try:
                    return max(0, int(value.strip()))
                except ValueError:
                    break
    return int(DEFAULT_IMAGE_PULL_RESERVE_GB.get(ref, unknown_default))
