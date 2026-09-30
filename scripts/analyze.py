"""python scripts/analyze.py --config configs/smoke.yaml [--models stub]"""
import argparse

import pandas as pd

from gsi.analysis.figures import all_figures
from gsi.analysis.tables import all_tables, load_scored
from gsi.experiment.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="*",
                    help="analyze selected models into tables/<model>/; omit for combined tables")
    args = ap.parse_args()
    cfg = load_config(args.config)
    df = load_scored(cfg.results_dir, args.models)
    if df.empty:
        print("no scored records found")
        return
    pd.set_option("display.width", 200, "display.max_rows", 500, "display.max_columns", 30)
    # Filtered runs must not replace another model's tables or the combined export.
    # Figures already include the model name in their filenames.
    groups = df.groupby("model") if args.models else [(None, df)]
    for model, records in groups:
        out = cfg.results_dir / "tables"
        if model is not None:
            out = out / model
        out.mkdir(parents=True, exist_ok=True)
        tables = all_tables(records)
        for name, table in tables.items():
            print(f"\n=== {name} ===")
            if table.empty:
                print("(empty)")
                continue
            print(table.to_string(index=False))
            table.to_csv(out / f"{name}.csv", index=False)
        all_figures(tables, cfg.results_dir / "figs")
        print(f"\ntables written to {out}, figures to {cfg.results_dir / 'figs'}")


if __name__ == "__main__":
    main()
