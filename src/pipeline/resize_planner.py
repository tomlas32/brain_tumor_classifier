from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, List, Dict, Mapping, Optional

@dataclass(frozen=True)
class ClassResizePlan:
    name: str
    found: int        # images found to read
    will_write: int   # images that would be written (1:1 here)

@dataclass(frozen=True)
class SubsetPlan:
    subset: str                 # "training" | "testing"
    root_in: Path
    root_out: Path
    classes: Tuple[ClassResizePlan, ...]
    total_found: int
    total_will_write: int

@dataclass(frozen=True)
class ResizePlan:
    size: int
    exts: Tuple[str, ...]       # resolved extensions (empty -> <any>)
    training: SubsetPlan | None
    testing: SubsetPlan | None

def _scan_subset(root_in: Path, root_out: Path, exts: set[str], subset_name: str, size: int) -> SubsetPlan:
    classes: List[ClassResizePlan] = []
    total_found = total_will_write = 0

    if root_in.exists():
        for class_dir in sorted(p for p in root_in.iterdir() if p.is_dir()):
            cnt = 0
            for p in class_dir.rglob("*"):
                if p.is_file() and (not exts or p.suffix.lower() in exts):
                    cnt += 1
            if cnt:
                classes.append(ClassResizePlan(class_dir.name, cnt, cnt))
                total_found += cnt
                total_will_write += cnt

    return SubsetPlan(
        subset=subset_name,
        root_in=root_in,
        root_out=root_out,
        classes=tuple(classes),
        total_found=total_found,
        total_will_write=total_will_write,
    )

def plan_resize(train_in: Path, train_out: Path,
                test_in: Optional[Path], test_out: Optional[Path],
                exts: set[str], size: int) -> ResizePlan:
    training = _scan_subset(train_in, train_out, exts, "training", size) if train_in.exists() else None
    testing  = _scan_subset(test_in, test_out, exts, "testing", size) if (test_in and test_in.exists()) else None

    return ResizePlan(
        size=size,
        exts=tuple(sorted(exts)),
        training=training,
        testing=testing,
    )

def make_log_extra(plan: ResizePlan) -> dict:
    def subset_dict(s: SubsetPlan | None):
        if s is None:
            return None
        return {
            "subset": s.subset,
            "root_in": str(s.root_in),
            "root_out": str(s.root_out),
            "totals": {"found": s.total_found, "will_write": s.total_will_write},
            "classes": [{"class": c.name, "found": c.found, "will_write": c.will_write} for c in s.classes],
        }
    return {
        "size": plan.size,
        "exts": list(plan.exts) if plan.exts else ["<any>"],
        "training": subset_dict(plan.training),
        "testing": subset_dict(plan.testing),
    }

def render_human(plan: ResizePlan, ctx: dict | None = None) -> str:
    lines = []
    lines.append("\n[DRY-RUN] Resize plan")
    lines.append(f"  size:          {plan.size}")
    exts_show = ", ".join(plan.exts) if plan.exts else "<any>"
    lines.append(f"  exts:          {exts_show}")

    def render_subset(s: SubsetPlan | None):
        if s is None:
            lines.append("  training/testing inputs: <missing>")
            return
        lines.append(f"\n  {s.subset.capitalize()} subset:")
        lines.append(f"    in : {s.root_in}")
        lines.append(f"    out: {s.root_out}")
        lines.append(f"    totals -> found: {s.total_found} | will_write: {s.total_will_write}")
        if s.classes:
            lines.append("    Per-class:")
            for c in s.classes:
                lines.append(f"      {c.name:15s} -> found: {c.found:5d} | will_write: {c.will_write:5d}")

    render_subset(plan.training)
    render_subset(plan.testing)

    lines.append("\n[DRY-RUN] No directories will be created and no files will be written.")
    return "\n".join(lines)

def build_empty_plan_context(*,
    train_in: Path, train_out: Path,
    test_in: Optional[Path], test_out: Optional[Path],
    exts: list[str], size: int
) -> tuple[ResizePlan, dict]:
    plan = ResizePlan(size=size, exts=tuple(exts), training=None, testing=None)
    extra = make_log_extra(plan)
    return plan, extra