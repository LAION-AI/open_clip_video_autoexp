import argparse
import json
import re
from glob import glob

import pandas as pd
model_profile = pd.read_csv("model_profile.csv").set_index("model")

PERF_DECIMALS = 1
NA_VALUE = "-"
RECALL_KS = (1, 5, 10)
RETR_IMAGE_COLS = [f"image_retrieval_recall@{k}" for k in RECALL_KS]
RETR_TEXT_COLS = [f"text_retrieval_recall@{k}" for k in RECALL_KS]
RETR_MEAN_COLS = [f"mean_r{k}" for k in RECALL_KS]
RETR_IMAGE_COL = RETR_IMAGE_COLS[0]
RETR_TEXT_COL = RETR_TEXT_COLS[0]
ACC1_COL = "acc1"
SCIENTIFIC_NOTATION_COLS = {"flops", "flops_per_sample"}

CLF_DATASETS = {
    "dev": ["kinetics400", "UCF-101-with-splits", "hmdb51-with-splits"],
    "test": ["kinetics400", "UCF-101-with-splits", "hmdb51-with-splits"]
}
RETR_DATASETS = {
    "dev": ["MSR-VTT", "MSVD", "BVD-10k"],
    "test": ["MSR-VTT", "MSVD"]
}
CLF_DATASETS_AVG = {
    "dev": ["kinetics400", "UCF-101-with-splits", "hmdb51-with-splits"],
    "test": ["kinetics400", "UCF-101-with-splits", "hmdb51-with-splits"]
}
RETR_DATASETS_AVG = {
    "dev": ["MSR-VTT", "MSVD"],
    "test": ["MSR-VTT", "MSVD"]
}

CLF_COLS = [ACC1_COL]
RETR_COLS = [
    col
    for columns in zip(RETR_IMAGE_COLS, RETR_TEXT_COLS, RETR_MEAN_COLS)
    for col in columns
]

METRIC_COLS = {
    "CLF": CLF_COLS,
    "RETR": RETR_COLS,
}
SEED_PATTERN = re.compile(r"seed\d+(?=(_|/|$))")


def get_df_baselines():
    rows = []
    for f in glob("baselines/*.json"):
        data = json.load(open(f))
        dic = data
        dic['pretrain_dataset'] = ''
        dic["downstream_dataset"] = dic["dataset"].replace("wds/", "")
        del dic["dataset"]
        dic.update(data["metrics"])
        del dic["metrics"]
        dic["model_folder"] = dic["model"] + "_" + dic["pretrained"]
        dic["flops_per_sample"] =  model_profile.loc[dic["model"], "gflops"]*10**9 if dic["model"] in model_profile.index else None
        dic["model"] = dic["model"] + " baseline"
        if "pt_datacomp" in f:
            dic["model"] = dic["model"] + " (DataComp)"
        elif "pt_openai" in f:
            dic["model"] = dic["model"] + " (OpenAI)"
        dic["split"] = "test"
        rows.append(dic)
    return pd.DataFrame(rows)


def rearrange_dataframe_with_metrics(df):
    # Make a new dataframe such that every row is a model, and columns contain all downstream metrics and the rest
    rows = []
    for model_folder, df_model_folder in df.groupby("model_folder"):
        metrics = {}
        for _, row in df_model_folder.iterrows():
            cols = CLF_COLS if row.downstream_dataset in CLF_DATASETS.get(row.split, {}) else RETR_COLS
            if row.split in CLF_DATASETS and row.downstream_dataset in CLF_DATASETS[row.split]:
                cols = CLF_COLS
            elif row.split in RETR_DATASETS and row.downstream_dataset in RETR_DATASETS[row.split]:
                cols = RETR_COLS
            else:
                cols = []
            for col in cols:
                metrics[row.downstream_dataset + "_" + col + "_" + row.split] = row[col]
        for k, v in metrics.items():
            row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.drop(columns=CLF_COLS + RETR_COLS + ["downstream_dataset", "split"])
    for split in ("dev", "test"):
        clf_cols = [ds + "_" + ACC1_COL + "_" + split for ds in CLF_DATASETS_AVG[split]]
        retr_cols = [ds + "_" + RETR_IMAGE_COL + "_" + split for ds in RETR_DATASETS_AVG[split]] + [ds + "_" + RETR_TEXT_COL + "_" + split for ds in RETR_DATASETS_AVG[split]]
        if set(clf_cols).issubset(set(df.columns)) and set(retr_cols).issubset(set(df.columns)):
            df["avg_clf_"+split] = df.apply(lambda row: row[clf_cols].mean(), axis=1)
            df["avg_ret_"+split] = df.apply(lambda row: row[retr_cols].mean(), axis=1)
            df["avg_overall_"+split] = 0.5 * (df["avg_clf_"+split] + df["avg_ret_"+split])
    if "avg_overall_test" in df:
        df["avg_overall" ] = df["avg_overall_test"]
    return df


def fmt_float(value: float, digits: int = PERF_DECIMALS) -> str:
    if pd.isna(value):
        return NA_VALUE
    return f"{value * 100:.{digits}f}"


def fmt_average_metric(
    values: pd.Series,
    mean_digits: int = PERF_DECIMALS,
    std_digits: int = 2,
) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return NA_VALUE
    mean = values.mean()
    if len(values) < 2:
        return f"{mean * 100:.{mean_digits}f} +/- {NA_VALUE}"
    std = values.std(ddof=1)
    return f"{mean * 100:.{mean_digits}f} +/- {std * 100:.{std_digits}f}"


def fmt_metric_values(values: pd.Series) -> str:
    formatted = []
    for value in values:
        if pd.isna(value):
            formatted.append(NA_VALUE)
        else:
            formatted.append(fmt_float(float(value)))
    return " ".join(formatted)


def fmt_flops(value: float) -> str:
    if pd.isna(value):
        return NA_VALUE
    mantissa, exponent = f"{float(value):.1e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exponent = str(int(exponent))
    return f"{mantissa}e{exponent}"


def fmt_human_number(value: float) -> str:
    if pd.isna(value):
        return NA_VALUE
    suffixes = [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]
    value = float(value)
    for divisor, suffix in suffixes:
        if abs(value) >= divisor:
            scaled = value / divisor
            if scaled.is_integer():
                return f"{int(scaled)}{suffix}"
            return f"{scaled:.1f}".rstrip("0").rstrip(".") + suffix
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}".rstrip("0").rstrip(".")


def format_cell(col: str, value, is_metric: bool = False) -> str:
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return NA_VALUE
    if is_metric:
        return fmt_float(value)
    if format_display_col(col) in SCIENTIFIC_NOTATION_COLS:
        return fmt_flops(value)
    if col == "samples_seen":
        return fmt_human_number(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def get_metric_columns_from_dataframe(df: pd.DataFrame) -> list[str]:
    metric_suffixes = tuple([m + "_" + split for m in tuple(CLF_COLS + RETR_COLS) for split in ("dev", "test", "decontaminated_test")])
    return [
        col for col in df.columns
        if col.startswith("avg_") or col.endswith(metric_suffixes)
    ]


def get_default_metric_cols(split="test") -> list[str]:
    clf_cols = [f"{dataset}_{ACC1_COL}_{split}" for dataset in CLF_DATASETS[split]]
    retr_cols = [
        f"{dataset}_{metric_col}_{split}"
        for dataset in RETR_DATASETS[split]
        for metric_col in RETR_IMAGE_COLS + RETR_TEXT_COLS
    ]
    return clf_cols + retr_cols + ["avg_overall"]


def format_display_col(col: str) -> str:
    if col.endswith(f"_{RETR_IMAGE_COL}"):
        return col.replace(f"_{RETR_IMAGE_COL}", "_video")
    if col.endswith(f"_{RETR_TEXT_COL}"):
        return col.replace(f"_{RETR_TEXT_COL}", "_text")
    return col


def get_text_alignment(df: pd.DataFrame, metric_cols: set[str]) -> list[bool]:
    align_right = []
    for col in df.columns:
        non_na = df[col].dropna()
        is_numeric = pd.api.types.is_numeric_dtype(non_na) if not non_na.empty else False
        align_right.append(col in metric_cols or is_numeric)
    return align_right


def get_display_rows(df: pd.DataFrame, metric_cols=None):
    metric_cols = set(metric_cols or [])
    cols = [format_display_col(col) for col in df.columns]
    rows = [
        [format_cell(col, row[col], is_metric=col in metric_cols) for col in df.columns]
        for _, row in df.iterrows()
    ]
    return cols, rows




def format_seed_values(values: pd.Series) -> str:
    values = values.dropna().drop_duplicates().tolist()
    try:
        values = sorted(values)
    except TypeError:
        values = sorted(values, key=str)

    formatted = []
    for value in values:
        if isinstance(value, float) and value.is_integer():
            formatted.append(str(int(value)))
        else:
            formatted.append(str(value))
    return ",".join(formatted)


def normalize_seed_identifier(value):
    return SEED_PATTERN.sub("seed", value)

def filter_multiple_seed_runs(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "name" not in df.columns or "seed" not in df.columns:
        return df

    df = df.reset_index(drop=True)
    key = df.name.map(normalize_seed_identifier)
    keep_indices = []
    for indices in df.groupby(key, sort=False, dropna=False).indices.values():
        group = df.iloc[list(indices)]
        if group["seed"].dropna().nunique() > 1:
            keep_indices.extend(indices)
    return df.iloc[keep_indices].reset_index(drop=True)


def group_seed_runs(df: pd.DataFrame, metric_cols: list[str], display_mode: str) -> pd.DataFrame:
    metric_cols = [col for col in metric_cols if col in df.columns]
    if df is None or df.empty or "name" not in df.columns or "seed" not in df.columns or not metric_cols:
        return df

    df = df.reset_index(drop=True)
    key = df.name.map(normalize_seed_identifier)

    rows = []
    for indices in df.groupby(key, sort=False, dropna=False).indices.values():
        group = df.iloc[list(indices)].sort_values("seed", kind="stable")
        if group["seed"].dropna().nunique() < 2:
            rows.extend(row.copy() for _, row in group.iterrows())
            continue

        row = group.iloc[0].copy()
        row["name"] = normalize_seed_identifier(row["name"])
        row["seed"] = format_seed_values(group["seed"])
        for col in metric_cols:
            if display_mode == "all":
                row[col] = fmt_metric_values(group[col])
            else:
                row[col] = fmt_average_metric(group[col])
        rows.append(row)

    return pd.DataFrame(rows, columns=df.columns).reset_index(drop=True)


def df_to_markdown(df: pd.DataFrame, metric_cols=None) -> str:
    if df is None or df.empty:
        return None
    cols, rows = get_display_rows(df, metric_cols)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        escaped = [cell.replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)


def df_to_text(df: pd.DataFrame, metric_cols=None) -> str:
    if df is None or df.empty:
        return None
    cols, rows = get_display_rows(df, metric_cols)
    align_right = get_text_alignment(df, set(metric_cols or []))
    widths = [len(col) for col in cols]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def format_row(row, is_header=False):
        formatted = []
        for idx, cell in enumerate(row):
            if not is_header and align_right[idx]:
                formatted.append(cell.rjust(widths[idx]))
            else:
                formatted.append(cell.ljust(widths[idx]))
        return "| " + " | ".join(formatted) + " |"

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    lines = [separator, format_row(cols, is_header=True), separator]
    lines.extend(format_row(row) for row in rows)
    lines.append(separator)
    return "\n".join(lines)


def df_to_plain(df: pd.DataFrame, metric_cols=None) -> str:
    if df is None or df.empty:
        return None
    _, rows = get_display_rows(df, metric_cols)
    return "\n".join(" ".join(row) for row in rows)


def df_to_prettytable(df: pd.DataFrame, metric_cols=None) -> str:
    if df is None or df.empty:
        return None
    try:
        from prettytable import PrettyTable
    except ImportError as exc:
        raise ImportError(
            "prettytable format requires the 'prettytable' package. "
            "Install it with `pip install prettytable`."
        ) from exc

    cols, rows = get_display_rows(df, metric_cols)
    align_right = get_text_alignment(df, set(metric_cols or []))
    table = PrettyTable()
    table.field_names = cols
    for idx, col in enumerate(cols):
        table.align[col] = "r" if align_right[idx] else "l"
    for row in rows:
        table.add_row(row)
    return table.get_string()


def escape_latex(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def df_to_latex(df: pd.DataFrame, metric_cols=None) -> str:
    if df is None or df.empty:
        return None
    cols, rows = get_display_rows(df, metric_cols)
    align_right = get_text_alignment(df, set(metric_cols or []))
    column_spec = "".join("r" if align else "l" for align in align_right)
    lines = [
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        " & ".join(escape_latex(col) for col in cols) + r" \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(" & ".join(escape_latex(cell) for cell in row) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines)


def render_table(df: pd.DataFrame, output_format: str, metric_cols=None) -> str:
    if output_format == "plain":
        return df_to_plain(df, metric_cols)
    if output_format == "text":
        return df_to_text(df, metric_cols)
    if output_format == "prettytable":
        return df_to_prettytable(df, metric_cols)
    if output_format == "latex":
        return df_to_latex(df, metric_cols)
    return df_to_markdown(df, metric_cols)


def calculate_spearman(
    df: pd.DataFrame,
    column_a: str,
    column_b: str,
) -> tuple[float, int]:
    missing = [column for column in (column_a, column_b) if column not in df.columns]
    if missing:
        raise ValueError(f"Missing Spearman column(s): {', '.join(missing)}")

    pairs = df[[column_a, column_b]].apply(pd.to_numeric, errors="coerce")
    pairs = pairs.replace([float("inf"), float("-inf")], pd.NA).dropna()
    if len(pairs) < 2:
        raise ValueError(
            "Spearman correlation requires at least two rows with both "
            "columns present and numeric"
        )

    ranks = pairs.rank(method="average")
    correlation = ranks[column_a].corr(ranks[column_b])
    if pd.isna(correlation):
        raise ValueError(
            "Spearman correlation is undefined because at least one selected "
            "column is constant"
        )
    return float(correlation), len(pairs)


def extra_cols(df):
    for image_col, text_col, mean_col in zip(
        RETR_IMAGE_COLS,
        RETR_TEXT_COLS,
        RETR_MEAN_COLS,
    ):
        df[mean_col] = 0.5 * (df[image_col] + df[text_col])
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize experiment results from a parquet file.")
    parser.add_argument("--parquet", default="results.parquet", help="Path to parquet file.")
    parser.add_argument("--model", default=None, help="Regexp to filter experiments by model")
    parser.add_argument("--dataset", default=None, help="Regexp to filter experiments by dataset")
    parser.add_argument("--name", type=str, help="Regexp to filter experiments by name")
    parser.add_argument("--keep_best", action="store_true")
    parser.add_argument(
        "--selection-metric",
        dest="selection_metric",
        default="avg_overall_dev",
        metavar="COLUMN",
        help=(
            "Metric column to maximize when --keep_best is enabled "
            "(default: avg_overall_dev)."
        ),
    )
    parser.add_argument(
        "--sort-by",
        default="avg_overall",
        metavar="COLUMN",
        help="Column used to sort output rows in ascending order (default: avg_overall).",
    )
    parser.add_argument("--show_name", action="store_true", help="Include the name column in the output table.")
    parser.add_argument("--show_throughput", action="store_true", help="Include the throughput columns")
    parser.add_argument("--short", action="store_true", help="only display model name")
    parser.add_argument("--cols", type=str, nargs="+", help="Columns to show")
    parser.add_argument("--base_cols", type=str, nargs="+", help="Base columns to show before all metric columns")
    parser.add_argument(
        "--spearman",
        nargs=2,
        metavar=("COLUMN_A", "COLUMN_B"),
        help=(
            "Calculate Spearman rank correlation between two columns after "
            "filtering and optional --keep_best selection."
        ),
    )
    parser.add_argument("--seed", type=int, nargs="+", help="Seeds to include")
    parser.add_argument(
        "--only_multiple_seeds",
        action="store_true",
        help="Only show experiments that have multiple seeds in the current selection",
    )
    parser.add_argument(
        "--group_seeds",
        action="store_true",
        help="Group runs that differ only by seed",
    )
    parser.add_argument(
        "--group_seeds_show",
        choices=["mean_std", "all"],
        default="mean_std",
        help="How to display metrics for grouped seeds",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "plain", "text", "prettytable", "latex"],
        default="markdown",
        help="Output table format. Plain emits space-separated rows without a header.",
    )
    args = parser.parse_args()

    df_baselines = extra_cols(get_df_baselines())
    df_baselines = rearrange_dataframe_with_metrics(df_baselines)

    df = pd.read_parquet(args.parquet)
    if args.seed:
        df = df[df.seed.isin(args.seed)]
    df = extra_cols(df)
    df = df[df.epoch == df.total_epochs]
    df = rearrange_dataframe_with_metrics(df)
    df = pd.concat((df, df_baselines))

    if args.sort_by not in df.columns:
        parser.error(f"--sort-by column {args.sort_by!r} is not available")
    df = df.sort_values(by=args.sort_by, ascending=True, na_position="last")
    df.loc[pd.isna(df.name), "name"] = "none"
    default_metric_cols = get_default_metric_cols()
    if args.model:
        df = df[df.model.str.contains(args.model)]
    if args.name:
        df = df[df.name.str.contains(args.name)]
    if args.dataset:
        df = df[df.pretrain_dataset.str.contains(args.dataset)]
    if args.only_multiple_seeds:
        df = filter_multiple_seed_runs(df)
    if args.group_seeds:
        df = group_seed_runs(df, get_metric_columns_from_dataframe(df), args.group_seeds_show)
    if args.keep_best:
        metric_cols = get_metric_columns_from_dataframe(df)
        if args.selection_metric not in metric_cols:
            available = ", ".join(sorted(metric_cols))
            parser.error(
                f"--selection-metric column {args.selection_metric!r} is not "
                f"available. Metric columns: {available}"
            )
        selection_values = pd.to_numeric(
            df[args.selection_metric],
            errors="coerce",
        )
        if not selection_values.notna().any():
            parser.error(
                f"--selection-metric column {args.selection_metric!r} has no "
                "numeric values after applying the requested filters"
            )
        df = df.assign(_keep_best_metric=selection_values)
        df = df.sort_values(
            by="_keep_best_metric",
            ascending=False,
            na_position="last",
        )
        df = df.drop_duplicates(
            subset=["model", "pretrain_dataset", "samples_seen"],
            keep="first",
        )
        df = df.drop(columns="_keep_best_metric")
        df = df.sort_values(
            by=args.sort_by,
            ascending=True,
            na_position="last",
        )
    spearman_result = None
    if args.spearman:
        try:
            correlation, sample_count = calculate_spearman(df, *args.spearman)
        except ValueError as error:
            parser.error(str(error))
        spearman_result = (
            f"Spearman({args.spearman[0]}, {args.spearman[1]}) = "
            f"{correlation:.6f} (n={sample_count})"
        )
    if args.base_cols:
        base_cols = args.base_cols
    elif args.short:
        base_cols = ["model",]
    else:
        base_cols = ["model", "global_batch_size", "pretrain_dataset", "samples_seen", "dataset_resampled", "lr", "wd", "lock_image", "flops_per_sample", "flops", "training_time_hours", "gpus"]
    if not args.base_cols and args.show_name:
        base_cols.append("name")
    if not args.base_cols and args.show_throughput:
        base_cols.extend(["samples_per_sec", "samples_per_sec_per_gpu"])
    if args.cols:
        df = df[args.cols]
    else:
        df = df[base_cols + default_metric_cols]
    metric_cols = get_metric_columns_from_dataframe(df)
    out = render_table(df, args.format, metric_cols=metric_cols)
    print(out)
    if spearman_result:
        print(f"\n{spearman_result}")
