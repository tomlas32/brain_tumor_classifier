from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple
import random
from pathlib import Path

@dataclass(frozen=True)
class ClassPlan:
    name: str
    source: int
    n_train: int
    n_val: int
    n_test: int

@dataclass(frozen=True)
class SplitPlan:
    classes: Tuple[ClassPlan, ...]
    total_source: int
    total_train: int
    total_val: int
    total_test: int

def plan_split(combined: Mapping[str, Iterable[Path]], test_frac: float, val_frac: float, seed: int,
               balance: str = "none") -> SplitPlan:
    random.seed(seed)

    cap_per_class = None
    if balance == "equalize":
        cap_per_class = min(len({str(p) for p in paths}) for paths in combined.values() if paths)

    classes: List[ClassPlan] = []
    total_src = total_train = total_val = total_test = 0

    for cls in sorted(combined.keys()):
        uniq = sorted({str(p) for p in combined[cls]})
        if cap_per_class is not None and len(uniq) > cap_per_class:
            uniq = random.sample(uniq, cap_per_class)
        n = len(uniq)

        if n == 0:
            continue
        n_test = max(1, int(n * test_frac))
        n_val = int(n * val_frac)
        if n - (n_test + n_val) < 1 and n > 1:
            if n_val > 0:
                n_val = max(0, n - n_test - 1)
            if n - (n_test + n_val) < 1 and n_test > 1:
                n_test = 1
        n_train = n - n_test - n_val

        classes.append(ClassPlan(cls, n, n_train, n_val, n_test))
        total_src += n
        total_train += n_train
        total_val += n_val
        total_test += n_test

    return SplitPlan(tuple(classes), total_src, total_train, total_val, total_test)

def make_log_extra(
    plan: SplitPlan,
    *,
    dataset_slug: str,
    pointer,
    balance: str,
    src_training: Path,
    src_testing: Path,
    exts: str,
    test_frac: float,
    val_frac: float,
    seed: int,
    clear_dest: bool,
    out_training: Path,
    out_validation: Path,
    out_testing: Path,
    mapping_use_dataset_subdir: bool,
    mapping_write_split_copy: bool,
    save_remap_to_project_root: bool,
):
    return {
        "dataset": dataset_slug,
        "sources": {"training": str(src_training), "testing": str(src_testing)},
        "exts": exts,
        "test_frac": test_frac,
        "val_frac": val_frac,
        "balance": balance,
        "seed": seed,
        "clear_dest": clear_dest,
        "outputs": {
            "training": str(out_training),
            "validation": str(out_validation),
            "testing": str(out_testing),
        },
        "mapping": {
            "use_dataset_subdir": bool(mapping_use_dataset_subdir),
            "write_split_copy": bool(mapping_write_split_copy),
            "save_remap_to_project_root": bool(save_remap_to_project_root),
        },
        "totals": {
            "source": plan.total_source,
            "train": plan.total_train,
            "val": plan.total_val,
            "test": plan.total_test,
        },
        "classes": [
            {"class": c.name, "source": c.source, "train": c.n_train, "val": c.n_val, "test": c.n_test}
            for c in plan.classes
        ],
    }

def render_human(plan: SplitPlan, context: Dict) -> str:
    lines = []
    lines.append("Split plan (dry run)")
    lines.append(f"  dataset:       {context['dataset']}")
    lines.append(f"  sources:       training={context['sources']['training']} | testing={context['sources']['testing']}")
    lines.append(f"  exts:          {context['exts']}")
    lines.append(f"  test_frac:     {context['test_frac']}")
    lines.append(f"  val_frac:      {context['val_frac']}")
    lines.append(f"  balance:       {context['balance']}")
    lines.append(f"  seed:          {context['seed']}")
    lines.append(f"  clear_dest:    {context['clear_dest']}")
    outs = context["outputs"]
    lines.append(f"  outputs:       training={outs['training']} | validation={outs['validation']} | testing={outs['testing']}")
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
            f"    {c.name:15s} -> source: {c.source:5d} | train: {c.n_train:5d} | val: {c.n_val:5d} | test: {c.n_test:5d}"
        )
    t = context["totals"]
    lines.append(f"\n  Totals -> source: {t['source']} | train: {t['train']} | val: {t['val']} | test: {t['test']}")
    return "\n".join(lines)