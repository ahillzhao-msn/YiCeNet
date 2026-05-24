#!/usr/bin/env python3
"""
Checkpoint management script for YiCeNet.
- Maintains registry.json with active + ready + fallback + history
- Prunes old checkpoints to control disk usage
- Hot-swap support via registry update

Registry stores paths RELATIVE to checkpoints/ directory.
All path resolution happens at read time.
"""
import json, os, time
from pathlib import Path

# Resolve project root relative to this script's location
PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
REGISTRY_PATH = CHECKPOINT_DIR / "registry.json"
MAX_KEEP = 3


def _resolve(p):
    """Resolve a registry path (relative or absolute) to absolute."""
    p = Path(p)
    if p.is_absolute():
        return p
    return (CHECKPOINT_DIR / p).resolve()


def _store(p):
    """Store a path RELATIVE to CHECKPOINT_DIR in registry."""
    p = Path(p).resolve()
    try:
        return os.path.relpath(str(p), start=str(CHECKPOINT_DIR))
    except ValueError:
        # Different drive on Windows — fall back to absolute
        return str(p)


def score_checkpoint(path: Path) -> float:
    """Score a checkpoint by recency + reward (if saved metadata)."""
    try:
        import torch
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        reward = ckpt.get("avg_reward", 0.0)
        version = ckpt.get("version", "")
        ver_num = 0
        if version and version.startswith("v"):
            try:
                ver_num = int(version[1:])
            except ValueError:
                pass
        return float(reward) * 100.0 + float(ver_num)
    except Exception:
        return 0.0


def clean_registry():
    """Clean registry.json, keeping only valid checkpoints."""
    if not REGISTRY_PATH.exists():
        return create_fresh_registry()
    
    with open(REGISTRY_PATH) as f:
        reg = json.load(f)
    
    def path_exists(p):
        if not p:
            return False
        return _resolve(p).exists()
    
    reg["history"] = [h for h in reg.get("history", []) if path_exists(h.get("path"))]
    
    for key in ["active", "ready", "fallback"]:
        entry = reg.get(key)
        if entry and isinstance(entry, dict):
            if not path_exists(entry.get("path")):
                if key == "active":
                    print(f"  Active checkpoint missing, searching for best available...")
                    best = find_best_checkpoint()
                    if best:
                        reg["active"] = best
                else:
                    reg[key] = None
    
    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)
    
    return reg


def find_best_checkpoint() -> dict:
    """Find the best checkpoint in the directory."""
    best_score = -1
    best = None
    
    for pt in CHECKPOINT_DIR.glob("yicenet_v*.pt"):
        if "test" in pt.name:
            continue
        score = score_checkpoint(pt)
        if score > best_score:
            best_score = score
            best = {
                "version": pt.stem.replace("yicenet_", ""),
                "path": _store(pt),
                "score": round(score, 2),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "notes": "Auto-detected best checkpoint"
            }
    
    return best


def create_fresh_registry():
    """Create a fresh registry from existing checkpoints."""
    print("Creating fresh registry.json...")
    
    checkpoints = []
    for pt in sorted(CHECKPOINT_DIR.glob("yicenet_v*.pt")):
        if "test" in pt.name:
            continue
        try:
            import torch
            ckpt = torch.load(str(pt), map_location="cpu", weights_only=False)
            ver = ckpt.get("version", pt.stem.replace("yicenet_", ""))
            reward = ckpt.get("avg_reward", 0.0)
            checkpoints.append({
                "version": ver,
                "path": _store(pt),
                "avg_reward": round(reward, 4),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "notes": f"v{ver} auto-detected"
            })
        except Exception as e:
            print(f"  Skipping {pt.name}: {e}")
    
    checkpoints.sort(key=lambda x: x.get("avg_reward", 0), reverse=True)
    active = checkpoints[0] if checkpoints else None
    
    reg = {
        "active": active,
        "ready": checkpoints[1] if len(checkpoints) > 1 else None,
        "fallback": checkpoints[2] if len(checkpoints) > 2 else None,
        "history": checkpoints,
    }
    
    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)
    
    print(f"  Active: {active['version'] if active else 'none'}")
    print(f"  History: {len(checkpoints)} entries")
    return reg


def prune_checkpoints(dry_run=False):
    """Remove low-scoring checkpoints, keep only MAX_KEEP best."""
    if not REGISTRY_PATH.exists():
        return
    
    with open(REGISTRY_PATH) as f:
        reg = json.load(f)
    
    protected = set()
    for key in ["active", "ready", "fallback"]:
        entry = reg.get(key)
        if entry and isinstance(entry, dict) and entry.get("path"):
            protected.add(str(_resolve(entry["path"])))
    
    scored = []
    for pt in CHECKPOINT_DIR.glob("yicenet_v*.pt"):
        if str(pt) in protected:
            continue
        if "test" in pt.name:
            continue
        score = score_checkpoint(pt)
        scored.append((score, pt))
    
    scored.sort(key=lambda x: -x[0])
    
    keep = set()
    for _, pt in scored[:MAX_KEEP]:
        keep.add(str(pt))
    
    removed = []
    for score, pt in scored:
        if str(pt) not in keep:
            if dry_run:
                print(f"  Would remove: {pt.name} (score={score:.2f})")
            else:
                pt.unlink()
                removed.append(pt.name)
    
    if removed:
        print(f"  Pruned {len(removed)} low-score checkpoints")
    elif not dry_run:
        print(f"  No checkpoints to prune ({len(scored)} scored, keeping top {MAX_KEEP})")


def register_new_checkpoint(version: str, path: str, metrics: dict):
    """Register a newly trained checkpoint. Makes it active."""
    path = str(Path(path).expanduser().resolve())
    
    reg = clean_registry()
    
    new_entry = {
        "version": version,
        "path": _store(path),
        "avg_reward": round(metrics.get("avg_reward", 0.0), 4),
        "best_reward": round(metrics.get("best_avg_reward", metrics.get("avg_reward", 0.0)), 4),
        "samples": metrics.get("samples", 0),
        "endogenous": metrics.get("endogenous", False),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "notes": metrics.get("notes", f"v{version} trained"),
    }
    
    if reg.get("active"):
        reg["fallback"] = reg.get("ready")
        reg["ready"] = reg["active"]
    
    reg["active"] = new_entry
    
    if "history" not in reg:
        reg["history"] = []
    reg["history"].append(new_entry)
    
    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)
    
    print(f"  Registered v{version} as active")
    return reg


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "prune":
        prune_checkpoints(dry_run="--dry" in sys.argv)
    elif len(sys.argv) > 1 and sys.argv[1] == "clean":
        clean_registry()
    elif len(sys.argv) > 1 and sys.argv[1] == "register":
        ver = sys.argv[2]
        pth = sys.argv[3]
        reward = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
        register_new_checkpoint(ver, pth, {"avg_reward": reward})
    elif len(sys.argv) > 1 and sys.argv[1] == "fresh":
        create_fresh_registry()
    else:
        print("Usage: checkpoint_manager.py [prune|clean|register|fresh] [args]")
        print("  prune [--dry]  — remove low-score checkpoints")
        print("  clean          — validate and clean registry.json")
        print("  register <ver> <path> [reward] — add new checkpoint as active")
        print("  fresh          — rebuild registry from existing .pt files")
