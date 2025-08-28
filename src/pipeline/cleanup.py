from __future__ import annotations

import argparse
import json
import os
import shutil
# import sys  # ← not used; safe to remove
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional  # Iterable not used, remove it

from src.utils.logging_utils import configure_logging, get_logger
from src.utils.parser_utils import (
    add_common_logging_args, 
    add_common_cleanup_args,
    add_common_config_args,
)
from src.utils.paths import (
    VALIDATION_REPORTS_DIR,
    CLEANUP_REPORTS_DIR,
    QUARANTINE_ROOT,
)
from src.core.cleanup_policy import (
    STRICT_ERROR_CODES,
    NEVER_AUTO_MOVE,
)

log = get_logger(__name__)


@dataclass(frozen=True)
class Finding:
    path: str
    label: Optional[str]
    subset: Optional[str]
    kind: str              # "error" | "warning"
    code: str
    sha1: Optional[str] = None
    duplicate_of: Optional[str] = None


# --- Utilities ----------------------------------------------------------------

def _now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _find_latest_report(tag: str | None = None) -> Path | None:
    if not VALIDATION_REPORTS_DIR.exists():
        return None
    pattern = f"*_{tag}.json" if tag else "*.json"
    candidates = sorted(
        (p for p in VALIDATION_REPORTS_DIR.glob(pattern) if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None



def _load_report(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_move(src: Path, dst: Path) -> Path:
    """
    Move src -> dst, avoiding filename collisions by appending __{i} before suffix.
    Returns the final destination path used.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    candidate = dst
    if candidate.exists():
        stem, suffix = candidate.stem, candidate.suffix
        i = 1
        while True:
            alt = candidate.with_name(f"{stem}__{i}{suffix}")
            if not alt.exists():
                candidate = alt
                break
            i += 1
    shutil.move(str(src), str(candidate))
    return candidate


def _path_index_from_findings(findings: List[Finding]) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """
    Build a lookup: path -> (subset, label) using finding records.
    """
    idx: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for f in findings:
        idx[f.path] = (f.subset, f.label)
    return idx


def _derive_subset_label_from_path(p: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Best-effort parsing of subset/label from path like data/<subset>/<label>/<file>.
    We keep it defensive: return (None, None) if structure is unexpected.
    """
    parts = p.parts
    try:
        # Find "data" in parts, subset next, label next
        if "data" in parts:
            i = parts.index("data")
            subset = parts[i + 1] if len(parts) > i + 1 else None
            label = parts[i + 2] if len(parts) > i + 2 else None
            return subset, label
    except Exception:
        pass
    return None, None


def _should_act_on(kind: str, code: str, policy: str, act_on: str) -> bool:
    if code in NEVER_AUTO_MOVE:
        return False
    if act_on == "errors" and kind != "error":
        return False
    if act_on == "warnings" and kind != "warning":
        return False
    # report_only never moves
    if policy == "report_only":
        return False
    if policy == "strict":
        return code in STRICT_ERROR_CODES or (kind == "error")
    if policy == "within_class":
        # within_class uses stricter logic inside duplicate handling; for non-duplicate errors, act like strict
        if code == "DUPLICATE":
            return True
        return (code in STRICT_ERROR_CODES) or (kind == "error")
    return False


def _plan_moves(
    findings: List[Finding],
    policy: str,
    act_on: str,
    run_id: str,  # ← accept stable run id
) -> Tuple[List[Tuple[Path, Path, Finding]], Dict[str, int]]:
    """
    Plan moves based on policy. Returns a list of (src, dst, finding) and counts by code.
    For duplicates, apply special rules depending on policy.
    """
    planned: List[Tuple[Path, Path, Finding]] = []
    counts_by_code: Dict[str, int] = defaultdict(int)

    # Build quick path->(subset,label) index
    idx = _path_index_from_findings(findings)

    # We need deterministic order: sort by path
    sorted_findings = sorted(findings, key=lambda f: (f.code, f.path))

    # Track first-kept SHA1 per (subset,label) for within_class policy
    first_by_sig_within: Dict[Tuple[str, str, str], str] = {}  # (sha1, subset, label) -> first_path

    for f in sorted_findings:
        src = Path(f.path)
        # Compute destination even if we may skip, for dry-run visibility
        subset_dir = f.subset or _derive_subset_label_from_path(src)[0] or "unknown_subset"
        label_dir = f.label or _derive_subset_label_from_path(src)[1] or "unknown_label"

        dst = QUARANTINE_ROOT / run_id / subset_dir / label_dir / src.name

        # Decide if this finding should be acted on
        if not _should_act_on(f.kind, f.code, policy, act_on):
            continue

        # Special handling for duplicates based on policy
        if f.code == "DUPLICATE" and f.sha1:
            # Determine classes/subsets for current and first occurrence
            cur_subset, cur_label = f.subset, f.label
            first_path = Path(f.duplicate_of) if f.duplicate_of else None
            first_subset, first_label = (None, None)
            if first_path:
                first_subset, first_label = idx.get(str(first_path), (None, None))
                if not first_subset or not first_label:
                    # Fallback best-effort parse from path
                    first_subset, first_label = _derive_subset_label_from_path(first_path)

            if policy == "within_class":
                # Only quarantine duplicates within the same subset+label; keep-first logic
                if first_subset == cur_subset and first_label == cur_label:
                    key = (f.sha1, cur_subset or "", cur_label or "")
                    if key not in first_by_sig_within:
                        # Mark the first; do not move this current item because 'duplicate' entries are never firsts
                        first_by_sig_within[key] = str(first_path) if first_path else str(src)
                        # The current item is a duplicate (by definition), so we quarantine it
                        planned.append((src, dst, f))
                        counts_by_code[f.code] += 1
                    else:
                        planned.append((src, dst, f))
                        counts_by_code[f.code] += 1
                else:
                    # Cross-class or cross-subset duplicates -> do nothing in within_class mode
                    continue
            else:
                # strict: quarantine all duplicates (later occurrences)
                planned.append((src, dst, f))
                counts_by_code[f.code] += 1
            continue

        # Non-duplicate errors/warnings (per policy)
        planned.append((src, dst, f))
        counts_by_code[f.code] += 1

    return planned, counts_by_code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Quarantine bad files based on a validate.py report (read-only consumer).")
    
    add_common_config_args(parser, include_dry_run=True)  # --config, --override, --dry-run
    add_common_logging_args(parser)  # --log-level, --log-file
    add_common_cleanup_args(parser)  # --eval-in, --eval-out, --trained-model
    args = parser.parse_args(argv)


    cfg = None
    try:
        from src.core.config import build_cleanup_config, to_dict
        cfg = build_cleanup_config(args.config, overrides=args.override)
    except Exception:
        cfg = None  # fallback: no structured config available

    # Run-aware logging
    run_id = os.getenv("RUN_ID") or _now_run_id()
    log_level = (getattr(cfg.log, "level", None) or args.log_level or "INFO")
    log_file  = (getattr(cfg.log, "file",  None) or args.log_file  or None)

    configure_logging(
        log_level=log_level,
        file_mode="fixed" if log_file else "auto",
        log_file=log_file,
        run_id=run_id,
        stage="cleanup",
    )
    log.info("config.resolved", extra={"config": to_dict(cfg)})

    # Merge config → args (config wins when present)
    report  = getattr(cfg, "report", None)  or args.report
    policy  = getattr(cfg, "policy", None)  or args.policy
    why     = getattr(cfg, "why", None)     or args.why
    dry_run     = bool(getattr(cfg, "dry_run", False) or getattr(args, "dry_run", False))
    report_tag = getattr(cfg, "report_tag", None) or getattr(args, "report_tag", None)
    
    # Resolve report path
    if report == "latest":
        report_path = _find_latest_report(report_tag)
        if not report_path:
            log.error("cleanup.no_reports_found", extra={"reports_dir": str(VALIDATION_REPORTS_DIR), "tag": report_tag})
            print("❌ No validation reports found. Run validate.py first.")
            return 2
    else:
        report_path = Path(report)
        if not report_path.exists():
            log.error("cleanup.report_missing", extra={"report": str(report_path)})
            print(f"❌ Report not found: {report_path}")
            return 2


    # Load report
    try:
        report = _load_report(report_path)
    except Exception as e:
        log.error("cleanup.report_load_failed", extra={"report": str(report_path), "error": str(e)})
        print(f"❌ Failed to read report: {report_path} ({e})")
        return 2

    raw_findings = report.get("findings", [])
    findings: List[Finding] = [
        Finding(
            path=f.get("path"),
            label=f.get("label"),
            subset=f.get("subset"),
            kind=f.get("kind"),
            code=f.get("code"),
            sha1=f.get("sha1"),
            duplicate_of=f.get("duplicate_of"),
        )
        for f in raw_findings
        if "path" in f and "code" in f and "kind" in f
    ]

    # Filter by severity (errors|warnings|both)
    if why == "errors":
        findings = [f for f in findings if f.kind == "error"]
    elif why == "warnings":
        findings = [f for f in findings if f.kind == "warning"]
    # else both -> no filter

    # Plan moves per policy
    planned, counts_by_code = _plan_moves(findings, policy=policy, act_on=why, run_id=run_id)

    total_to_move = len(planned)
    if policy == "report_only" or dry_run:
        mode = "REPORT-ONLY" if policy == "report_only" else "DRY-RUN"
        print(f"[{mode}] Planned moves: {total_to_move}")
        if counts_by_code:
            print("  By code:", dict(counts_by_code))
        for src, dst, f in planned[:50]:  # don't flood the console
            print(f"    {f.code:12s} :: {src}  ->  {dst}")
        if total_to_move > 50:
            print(f"    ... and {total_to_move - 50} more")
        # Write a plan file for audit
        CLEANUP_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        plan_out = CLEANUP_REPORTS_DIR / f"cleanup_plan_{run_id}.json"
        with open(plan_out, "w", encoding="utf-8") as pf:
            json.dump({
                "run_id": run_id,
                "source_report": str(report_path),
                "policy": policy,
                "acted_on": why,
                "planned_count": total_to_move,
                "by_code": dict(counts_by_code),
                "items": [
                    {"from": str(src), "to": str(dst), "code": f.code}
                    for src, dst, f in planned
                ],
            }, pf, indent=2)
        log.info("cleanup.plan_written", extra={"path": str(plan_out), "count": total_to_move})
        return 0

    if total_to_move == 0:
        print("Nothing to quarantine. Dataset already clean per selected policy/severity.")
        return 0

    # Execute moves
    moved = []
    skipped = []
    for src, dst, f in planned:
        try:
            if not src.exists():
                skipped.append({"from": str(src), "reason": "missing"})
                log.warning("cleanup.skip_missing", extra={"src": str(src)})
                continue
            final_dst = _safe_move(src, dst)
            moved.append({"from": str(src), "to": str(final_dst), "code": f.code,
                          "sha1": f.sha1, "duplicate_of": f.duplicate_of})
            log.debug("cleanup.moved", extra={"from": str(src), "to": str(final_dst), "code": f.code})
        except Exception as e:
            skipped.append({"from": str(src), "reason": f"move_failed: {e}"})
            log.error("cleanup.move_failed", extra={"from": str(src), "to": str(dst), "error": str(e)})

    # Write manifest
    CLEANUP_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = CLEANUP_REPORTS_DIR / f"quarantine_{run_id}_{_now_run_id()}.json"
    manifest = {
        "run_id": run_id,
        "source_report": str(report_path),
        "policy": policy,
        "acted_on": why,
        "moved": moved,
        "skipped": skipped,
        "totals": {
            "planned": total_to_move,
            "moved": len(moved),
            "skipped": len(skipped),
            "by_code": dict(counts_by_code),
        },
        "quarantine_root": str(QUARANTINE_ROOT / run_id),
    }
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    print(f"Cleanup summary: moved={len(moved)} | skipped={len(skipped)}")
    print("  By code:", dict(counts_by_code))
    print(f"Manifest: {manifest_path}")
    log.info("cleanup.done", extra={"moved": len(moved), "skipped": len(skipped), "manifest": str(manifest_path)})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
