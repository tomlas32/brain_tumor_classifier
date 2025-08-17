from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import random
from pathlib import Path

@dataclass(frozen=True)
class ClassPlan:
    name: str
    source: int
    n_train: int
    n_test: int

@dataclass(frozen=True)
class SplitPlan:
    classes: Tuple[ClassPlan, ...]
    total_source: int
    total_train: int
    total_test: int

def plan_split(
    combined: Mapping[str, Iterable[Path]],
    test_frac: float,
    seed: int,
) -> SplitPlan:
    """Compute per-class split counts deterministically (pure, no I/O)."""
    random.seed(seed)
    classes: List[ClassPlan] = []
    total_src = total_train = total_test = 0

    for cls in sorted(combined.keys()):
        uniq = sorted({str(p) for p in combined[cls]})
        n = len(uniq)
        if n == 0:
            continue
        n_test = max(1, int(n * test_frac))
        n_train = n - n_test

        classes.append(ClassPlan(cls, n, n_train, n_test))
        total_src += n
        total_train += n_train
        total_test += n_test

    return SplitPlan(tuple(classes), total_src, total_train, total_test)

def make_log_extra(
    plan: SplitPlan,
    *,
    dataset_slug: str,
    pointer: Path,
    src_training: Path,
    src_testing: Path,
    exts: Sequence[str] | None,
    test_frac: float,
    seed: int,
    clear_dest: bool,
    out_training: Path,
    out_testing: Path,
    mapping_use_dataset_subdir: bool,
    mapping_write_split_copy: bool,
    save_remap_to_project_root: bool,
) -> dict:
    """Build a structured dict for logger extra=…"""
    return {
        "dataset": dataset_slug,
        "pointer": str(pointer),
        "src_training": str(src_training),
        "src_testing": str(src_testing),
        "exts": sorted(exts) if exts else ["<any>"],
        "test_frac": test_frac,
        "seed": seed,
        "clear_dest": clear_dest,
        "outputs": {"training": str(out_training), "testing": str(out_testing)},
        "mapping": {
            "use_dataset_subdir": bool(mapping_use_dataset_subdir),
            "write_split_copy": bool(mapping_write_split_copy),
            "save_remap_to_project_root": bool(save_remap_to_project_root),
        },
        "totals": {
            "source": plan.total_source,
            "train": plan.total_train,
            "test": plan.total_test,
        },
        "classes": [
            {"class": c.name, "source": c.source, "train": c.n_train, "test": c.n_test}
            for c in plan.classes
        ],
    }

def render_human(plan: SplitPlan, context: dict) -> str:
    """Pretty, console-friendly summary string."""
    lines = []
    lines.append("\n[DRY-RUN] Split plan")
    lines.append(f"  dataset:       {context['dataset']}")
    lines.append(f"  pointer:       {context['pointer']}")
    lines.append(
        f"  source roots:  training={context['src_training']} | testing={context['src_testing']}"
    )
    lines.append(
        f"  exts:          {', '.join(context['exts']) if context['exts'] else '<any>'}"
    )
    lines.append(f"  test_frac:     {context['test_frac']}")
    lines.append(f"  seed:          {context['seed']}")
    lines.append(f"  clear_dest:    {context['clear_dest']}")
    outs = context["outputs"]
    lines.append(f"  outputs:       training={outs['training']} | testing={outs['testing']}")
    m = context["mapping"]
    lines.append(
        "  mapping opts:  "
        f"use_dataset_subdir={m['use_dataset_subdir']}, "
        f"write_split_copy={m['write_split_copy']}, "
        f"save_remap_to_project_root={m['save_remap_to_project_root']}"
    )
    lines.append("\n  Per-class plan:")
    for c in plan.classes:
        lines.append(
            f"    {c.name:15s} -> source: {c.source:5d} | train: {c.n_train:5d} | test: {c.n_test:5d}"
        )
    t = context["totals"]
    lines.append(f"\n  Totals -> source: {t['source']} | train: {t['train']} | test: {t['test']}")
    lines.append("\n[DRY-RUN] No files will be created, moved, or modified.")
    return "\n".join(lines)


def build_empty_plan_context(
    *,
    dataset_slug: str,
    pointer: Path,
    exts: Sequence[str] | None,
    test_frac: float,
    seed: int,
    clear_dest: bool,
    out_training: Path,
    out_testing: Path,
    mapping_use_dataset_subdir: bool,
    mapping_write_split_copy: bool,
    save_remap_to_project_root: bool,
):
    """
    Construct an empty SplitPlan + logging context for dry-runs when no pointer/data exist.
    Keeps 'what to print/log' centralized in the planner.
    """
    plan = SplitPlan(classes=tuple(), total_source=0, total_train=0, total_test=0)
    # placeholders for display only
    src_training = Path("<missing>/Training")
    src_testing  = Path("<missing>/Testing")

    extra = make_log_extra(
        plan,
        dataset_slug=dataset_slug,
        pointer=pointer,
        src_training=src_training,
        src_testing=src_testing,
        exts=exts,
        test_frac=test_frac,
        seed=seed,
        clear_dest=clear_dest,
        out_training=out_training,
        out_testing=out_testing,
        mapping_use_dataset_subdir=mapping_use_dataset_subdir,
        mapping_write_split_copy=mapping_write_split_copy,
        save_remap_to_project_root=save_remap_to_project_root,
    )
    return plan, extra
