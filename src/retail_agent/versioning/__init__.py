"""VersionRegistry：模型版本 + Prompt 版本管理，支持一键回滚"""
from __future__ import annotations
import json
import shutil
from datetime import datetime
from pathlib import Path

_REGISTRY_PATH = Path(__file__).parents[5] / "data" / "version_registry.json"
_SNAPSHOT_DIR  = Path(__file__).parents[5] / "data" / "snapshots"


def _load() -> dict:
    if _REGISTRY_PATH.exists():
        return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    return {"versions": [], "current": None}


def _save(registry: dict) -> None:
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REGISTRY_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


def register(
    version: str,
    model_tag: str,
    prompt_hash: str,
    notes: str = "",
) -> None:
    """注册一个新版本快照"""
    registry = _load()
    entry = {
        "version":     version,
        "model_tag":   model_tag,
        "prompt_hash": prompt_hash,
        "notes":       notes,
        "created_at":  datetime.utcnow().isoformat(),
    }
    registry["versions"].append(entry)
    registry["current"] = version

    # 快照当前 experts 目录
    snapshot_dir = _SNAPSHOT_DIR / version
    src = Path(__file__).parents[3] / "layer2_orchestration" / "experts"
    if src.exists():
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        shutil.copytree(src, snapshot_dir)

    _save(registry)
    print(f"[VersionRegistry] 注册版本 {version}（model={model_tag}, prompt={prompt_hash[:8]}）")


def rollback(version: str) -> bool:
    """回滚到指定版本，恢复 experts 目录"""
    registry = _load()
    versions = {v["version"]: v for v in registry["versions"]}

    if version not in versions:
        print(f"[VersionRegistry] 版本 {version} 不存在，可用版本: {list(versions.keys())}")
        return False

    snapshot_dir = _SNAPSHOT_DIR / version
    if not snapshot_dir.exists():
        print(f"[VersionRegistry] 版本 {version} 的快照文件不存在")
        return False

    dest = Path(__file__).parents[3] / "layer2_orchestration" / "experts"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(snapshot_dir, dest)

    registry["current"] = version
    _save(registry)
    print(f"[VersionRegistry] 已回滚到版本 {version}")
    return True


def current_version() -> str | None:
    return _load().get("current")


def list_versions() -> list[dict]:
    return _load().get("versions", [])
