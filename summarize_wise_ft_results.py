#!/usr/bin/env python3
"""Compare base and dev-selected WiSE-FT results for ViCLIP models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SPLITS = ("dev", "test")
REQUIRED_METRICS = (
    "k400_acc1",
    "hmdb51_acc1",
    "ucf101_acc1",
    "msrvtt_t2v_r1",
    "msrvtt_v2t_r1",
    "msvd_t2v_r1",
    "msvd_v2t_r1",
)
COLUMNS = (
    ("k400_acc1", "K400 acc1"),
    ("hmdb51_acc1", "HMDB51 acc1"),
    ("ucf101_acc1", "UCF101 acc1"),
    ("msrvtt_t2v_r1", "MSR-VTT T2V R@1"),
    ("msrvtt_v2t_r1", "MSR-VTT V2T R@1"),
    ("msvd_t2v_r1", "MSVD T2V R@1"),
    ("msvd_v2t_r1", "MSVD V2T R@1"),
    ("avg_dev", "avg_dev"),
    ("avg_test", "avg_test"),
)
SELECTION_METRICS = {
    "avg-dev": (),
    "bvd-10k": ("bvd10k_t2v_r1", "bvd10k_v2t_r1"),
    "kinetics400": ("k400_acc1",),
    "hmdb51": ("hmdb51_acc1",),
    "ucf-101": ("ucf101_acc1",),
    "msr-vtt": ("msrvtt_t2v_r1", "msrvtt_v2t_r1"),
    "msvd": ("msvd_t2v_r1", "msvd_v2t_r1"),
}
SELECTION_DESCRIPTIONS = {
    "avg-dev": "the balanced downstream avg_dev",
    "bvd-10k": "dev BVD-10k mean T2V/V2T R@1",
    "kinetics400": "dev Kinetics-400 accuracy",
    "hmdb51": "dev HMDB51 accuracy",
    "ucf-101": "dev UCF101 accuracy",
    "msr-vtt": "dev MSR-VTT mean T2V/V2T R@1",
    "msvd": "dev MSVD mean T2V/V2T R@1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("wise-ft"))
    parser.add_argument("--original", default="ViCLIP-original")
    parser.add_argument(
        "--selection-metric",
        choices=tuple(SELECTION_METRICS),
        default="avg-dev",
        help=(
            "Dev metric used to select positive WiSE-FT coefficients. "
            "Retrieval datasets use mean bidirectional R@1."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optionally write the Markdown table to this file.",
    )
    parser.add_argument(
        "--no-averages",
        action="store_true",
        help="Hide the avg_dev and avg_test columns from the table.",
    )
    parser.add_argument("--precision", type=int, default=2)
    return parser.parse_args()


def model_label(folder_name: str) -> str:
    return f"Our model ({folder_name})"


def discover_our_models(root: Path, original: str) -> list[tuple[str, Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"WiSE-FT result root does not exist: {root}")

    model_dirs = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and path.name != original
        ),
        key=lambda path: path.name,
    )
    if not model_dirs:
        raise ValueError(f"No model directories found below {root}")

    return [
        (model_label(path.name), path)
        for path in model_dirs
    ]


def canonical_dataset(dataset: str) -> str | None:
    name = dataset.removeprefix("wds/").lower()
    if name == "kinetics400":
        return "k400"
    if name in {"hmdb51", "hmdb51-with-splits"}:
        return "hmdb51"
    if name in {"ucf-101", "ucf-101-with-splits", "ucf101"}:
        return "ucf101"
    if name == "msr-vtt":
        return "msrvtt"
    if name == "msvd":
        return "msvd"
    if name == "bvd-10k":
        return "bvd10k"
    return None


def common_wise_ft_coef(result: dict) -> float | None:
    """Return the reference-checkpoint coefficient used by clip_benchmark."""
    if "wiseft_coef" in result:
        return round(float(result["wiseft_coef"]), 10)
    if result.get("wise_ft"):
        # The original runner stores alpha * ViCLIP + (1 - alpha) * reference.
        return round(1.0 - float(result["wise_ft_alpha"]), 10)
    return None


def result_metrics(result: dict) -> dict[str, float]:
    dataset = canonical_dataset(result["dataset"])
    if dataset is None:
        return {}

    metrics = result["metrics"]
    if dataset in {"k400", "hmdb51", "ucf101"}:
        return {f"{dataset}_acc1": 100.0 * float(metrics["acc1"])}

    # image_retrieval: text queries retrieve images/videos (T2V).
    # text_retrieval: image/video queries retrieve texts (V2T).
    return {
        f"{dataset}_t2v_r1": 100.0
        * float(metrics["image_retrieval_recall@1"]),
        f"{dataset}_v2t_r1": 100.0
        * float(metrics["text_retrieval_recall@1"]),
    }


def load_model_results(model_dir: Path) -> dict[str, dict[float | None, dict[str, float]]]:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model result directory does not exist: {model_dir}")

    loaded: dict[str, dict[float | None, dict[str, float]]] = {}
    for split in SPLITS:
        split_dir = model_dir / split
        if not split_dir.is_dir():
            loaded[split] = {}
            continue

        configurations: dict[float | None, dict[str, float]] = {}
        for path in sorted(split_dir.glob("*.json")):
            with path.open(encoding="utf-8") as handle:
                result = json.load(handle)
            metrics = result_metrics(result)
            if not metrics:
                continue
            coef = common_wise_ft_coef(result)
            destination = configurations.setdefault(coef, {})
            overlap = destination.keys() & metrics.keys()
            if overlap:
                names = ", ".join(sorted(overlap))
                raise ValueError(
                    f"Duplicate metrics for {model_dir.name}, {split}, "
                    f"coef={coef}: {names}"
                )
            destination.update(metrics)
        loaded[split] = configurations
    return loaded


def has_complete_results(
    results: dict[str, dict[float | None, dict[str, float]]],
    coef: float | None,
) -> bool:
    try:
        metrics_for(results, "dev", coef)
        metrics_for(results, "test", coef)
    except ValueError:
        return False
    return True


def metrics_for(
    results: dict[str, dict[float | None, dict[str, float]]],
    split: str,
    coef: float | None,
) -> dict[str, float]:
    try:
        metrics = results[split][coef]
    except KeyError as error:
        mode = "without WiSE-FT" if coef is None else f"with coef={coef:g}"
        raise ValueError(f"Missing {split} results {mode}") from error

    missing = set(REQUIRED_METRICS) - metrics.keys()
    if missing:
        raise ValueError(
            f"Incomplete {split} results for coef={coef}: "
            f"missing {', '.join(sorted(missing))}"
        )
    return metrics


def average(metrics: dict[str, float]) -> float:
    classification = (
        metrics["k400_acc1"]
        + metrics["hmdb51_acc1"]
        + metrics["ucf101_acc1"]
    ) / 3.0
    retrieval = (
        metrics["msrvtt_t2v_r1"]
        + metrics["msrvtt_v2t_r1"]
        + metrics["msvd_t2v_r1"]
        + metrics["msvd_v2t_r1"]
    ) / 4.0
    return 0.5 * classification + 0.5 * retrieval


def select_best_dev_coef(
    results: dict[str, dict[float | None, dict[str, float]]],
    selection_metric: str,
) -> float | None:
    candidates = sorted(
        coef for coef in results["dev"] if coef is not None and coef > 0.0
    )
    if not candidates:
        return None

    selection_keys = SELECTION_METRICS[selection_metric]
    if selection_keys:
        required_selection_keys = set(selection_keys)
        missing_selection = [
            coef
            for coef in candidates
            if not required_selection_keys <= results["dev"][coef].keys()
        ]
        if len(missing_selection) == len(candidates):
            return None
        if missing_selection:
            missing = ", ".join(f"{coef:g}" for coef in missing_selection)
            raise ValueError(
                f"Incomplete {SELECTION_DESCRIPTIONS[selection_metric]} "
                f"coefficient sweep; missing metrics for coefficients: {missing}"
            )

    def selection_score(coef: float) -> float:
        metrics = metrics_for(results, "dev", coef)
        if selection_metric == "avg-dev":
            return average(metrics)
        return sum(metrics[key] for key in selection_keys) / len(selection_keys)

    # Prefer the smaller coefficient if selection scores tie.
    return max(
        candidates,
        key=lambda coef: (selection_score(coef), -coef),
    )


def make_row(
    label: str,
    results: dict[str, dict[float | None, dict[str, float]]],
    coef: float | None,
) -> tuple[str, dict[str, float]]:
    dev = metrics_for(results, "dev", coef)
    test = metrics_for(results, "test", coef)
    row = dict(test)
    row["avg_dev"] = average(dev)
    row["avg_test"] = average(test)
    if coef is not None:
        label = f"{label} + WiSE-FT (coef={coef:g})"
    return label, row


def markdown_table(
    rows: list[tuple[str, dict[str, float]]],
    precision: int,
    show_averages: bool = True,
) -> str:
    columns = (
        COLUMNS
        if show_averages
        else tuple(column for column in COLUMNS if not column[0].startswith("avg_"))
    )
    headers = ["Model / configuration", *(title for _, title in columns)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", *(["---:"] * len(columns))]) + " |",
    ]
    for label, values in rows:
        formatted = [f"{values[key]:.{precision}f}" for key, _ in columns]
        lines.append("| " + " | ".join([label, *formatted]) + " |")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    original = load_model_results(args.root / args.original)
    discovered_models = [
        (label, load_model_results(path))
        for label, path in discover_our_models(args.root, args.original)
    ]
    our_models = [
        (label, results)
        for label, results in discovered_models
        if has_complete_results(results, None)
    ]
    incomplete_models = [
        label
        for label, results in discovered_models
        if not has_complete_results(results, None)
    ]

    original_coef = select_best_dev_coef(original, args.selection_metric)
    our_selected = [
        (label, results, select_best_dev_coef(results, args.selection_metric))
        for label, results in our_models
    ]
    rows = [
        make_row("Original ViCLIP", original, None),
        *(
            make_row(label, results, None)
            for label, results in our_models
        ),
    ]
    selected = [
        ("Original ViCLIP", original, original_coef),
        *our_selected,
    ]
    if original_coef is not None:
        rows.append(make_row("Original ViCLIP", original, original_coef))
    if original_coef != 0.5:
        rows.append(make_row("Original ViCLIP", original, 0.5))
    unavailable_fixed = []
    for label, results, coef in our_selected:
        if coef is not None:
            rows.append(make_row(label, results, coef))
    table = markdown_table(
        rows,
        args.precision,
        show_averages=not args.no_averages,
    )

    selection_description = SELECTION_DESCRIPTIONS[args.selection_metric]
    note = (
        "\n\nMetric columns show test-set percentages. WiSE-FT coefficients are "
        f"selected independently for each model using {selection_description}."
    )
    unavailable = [label for label, _, coef in selected if coef is None]
    if unavailable:
        note += (
            " No selected WiSE-FT row is shown for "
            + ", ".join(unavailable)
            + f" because its {selection_description} sweep is unavailable."
        )
    if incomplete_models:
        note += (
            " No table rows are shown for "
            + ", ".join(incomplete_models)
            + " because its complete dev/test baseline results are unavailable."
        )
    if unavailable_fixed:
        note += (
            " No fixed coef=0.5 row is shown for "
            + ", ".join(unavailable_fixed)
            + " because its complete dev/test coef=0.5 results are unavailable."
        )
    output = table + note
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
