#!/usr/bin/env python3
"""Hot backup for the nodexo validator database.

SQLite deployments use sqlite3's online backup API so the validator can
keep running while the backup is taken. PostgreSQL deployments use pg_dump
in custom format. Both paths write a sibling sha256 checksum and prune old
local backups.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    load_env(REPO_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Back up the validator DB")
    parser.add_argument(
        "--db-url",
        default=(
            os.environ.get("DB_URL")
            or os.environ.get("NODEXO_VALIDATOR_DB_URL")
            or default_db_url()
        ),
    )
    parser.add_argument(
        "--backup-dir",
        default=os.environ.get("NODEXO_VALIDATOR_BACKUP_DIR")
        or str(data_dir() / "backups"),
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=int(os.environ.get("NODEXO_VALIDATOR_BACKUP_RETENTION_DAYS", "14")),
    )
    args = parser.parse_args()

    backup_dir = Path(args.backup_dir).expanduser().resolve()
    backup_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        backup_dir.chmod(0o700)
    except OSError:
        pass

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    if is_sqlite_url(args.db_url):
        output = backup_sqlite(args.db_url, backup_dir, stamp)
    elif args.db_url.startswith(("postgresql://", "postgresql+")):
        output = backup_postgres(args.db_url, backup_dir, stamp)
    else:
        raise SystemExit(f"unsupported DB_URL scheme: {args.db_url.split(':', 1)[0]}")

    digest = sha256_file(output)
    checksum = output.with_suffix(output.suffix + ".sha256")
    checksum.write_text(f"{digest}  {output.name}\n")
    checksum.chmod(0o600)

    upload_if_configured(output, checksum)
    prune_backups(backup_dir, args.retention_days)

    size_mib = output.stat().st_size / 1024 / 1024
    print(f"created {output} ({size_mib:.2f} MiB, sha256 {digest})")
    return 0


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def data_dir() -> Path:
    return Path(os.environ.get("NODEXO_DATA_DIR", "~/.nodexo")).expanduser().resolve()


def default_db_url() -> str:
    return "sqlite:///" + str(data_dir() / "validator.db")


def is_sqlite_url(db_url: str) -> bool:
    return db_url.startswith("sqlite:///")


def sqlite_path(db_url: str) -> Path:
    raw = db_url.removeprefix("sqlite:///")
    if raw == ":memory:":
        raise SystemExit("cannot back up in-memory SQLite DB")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def backup_sqlite(db_url: str, backup_dir: Path, stamp: str) -> Path:
    source_path = sqlite_path(db_url)
    if not source_path.exists():
        raise SystemExit(f"SQLite DB not found: {source_path}")

    tmp_sqlite = backup_dir / f"validator-{stamp}.sqlite.tmp"
    output = backup_dir / f"validator-{stamp}.sqlite.gz"

    source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
    try:
        dest = sqlite3.connect(tmp_sqlite)
        try:
            source.backup(dest)
        finally:
            dest.close()
    finally:
        source.close()

    try:
        tmp_sqlite.chmod(0o600)
        with tmp_sqlite.open("rb") as src, gzip.open(output, "wb", compresslevel=9) as dst:
            shutil.copyfileobj(src, dst)
        output.chmod(0o600)
    finally:
        tmp_sqlite.unlink(missing_ok=True)
    return output


def backup_postgres(db_url: str, backup_dir: Path, stamp: str) -> Path:
    pg_dump = shutil.which("pg_dump")
    if not pg_dump:
        raise SystemExit("pg_dump not found in PATH")
    pg_dump_url = db_url
    if pg_dump_url.startswith("postgresql+"):
        pg_dump_url = "postgresql://" + pg_dump_url.split("://", 1)[1]

    tmp_output = backup_dir / f"validator-{stamp}.dump.tmp"
    output = backup_dir / f"validator-{stamp}.dump"
    result = subprocess.run(
        [pg_dump, "--format=custom", "--file", str(tmp_output), pg_dump_url],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=600,
    )
    if result.returncode != 0:
        tmp_output.unlink(missing_ok=True)
        raise SystemExit(f"pg_dump failed: {result.stderr.strip()[:500]}")
    tmp_output.chmod(0o600)
    tmp_output.replace(output)
    output.chmod(0o600)
    return output


def upload_if_configured(output: Path, checksum: Path) -> None:
    bucket = os.environ.get("NODEXO_VALIDATOR_BACKUP_R2_BUCKET", "").strip()
    if not bucket:
        return
    upload_r2(output)
    upload_r2(checksum)


def upload_r2(path: Path) -> None:
    account_id = require_env("NODEXO_VALIDATOR_BACKUP_R2_ACCOUNT_ID")
    access_key_id = require_env("NODEXO_VALIDATOR_BACKUP_R2_ACCESS_KEY_ID")
    secret_access_key = require_env("NODEXO_VALIDATOR_BACKUP_R2_SECRET_ACCESS_KEY")
    bucket = require_env("NODEXO_VALIDATOR_BACKUP_R2_BUCKET")
    prefix = os.environ.get("NODEXO_VALIDATOR_BACKUP_R2_PREFIX", "").strip().strip("/")
    key = "/".join(part for part in (prefix, path.name) if part)

    client = boto3.client(
        "s3",
        region_name="auto",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    )
    with path.open("rb") as body:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType=content_type_for(path),
        )
    print(f"uploaded r2://{bucket}/{key}")


def require_env(key: str) -> str:
    value = os.environ.get(key, "").strip()
    if not value:
        raise SystemExit(f"{key} is required when NODEXO_VALIDATOR_BACKUP_R2_BUCKET is set")
    return value


def content_type_for(path: Path) -> str:
    name = path.name
    if name.endswith(".sha256"):
        return "text/plain"
    if name.endswith(".gz"):
        return "application/gzip"
    return "application/octet-stream"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prune_backups(backup_dir: Path, retention_days: int) -> None:
    if retention_days <= 0:
        return
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    for path in backup_dir.iterdir():
        if not path.name.startswith("validator-"):
            continue
        if not (
            path.name.endswith(".sqlite.gz")
            or path.name.endswith(".sqlite.gz.sha256")
            or path.name.endswith(".dump")
            or path.name.endswith(".dump.sha256")
        ):
            continue
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if mtime < cutoff:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
