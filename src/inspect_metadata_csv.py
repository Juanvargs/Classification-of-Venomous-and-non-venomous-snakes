import argparse
import csv
import json
from collections import Counter
from pathlib import Path


KEYWORDS = (
    "species",
    "scientific",
    "common",
    "venom",
    "venomous",
    "poison",
    "toxic",
    "danger",
    "risk",
    "class",
    "label",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Inspecciona un CSV de metadata.")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--output-dir", default="reports/audits/dataset_sources/metadata_inspection")
    parser.add_argument("--sample-size", type=int, default=20)
    return parser.parse_args()


def sniff_dialect(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        sample = file.read(4096)
    try:
        return csv.Sniffer().sniff(sample)
    except csv.Error:
        return csv.excel


def inspect_csv(path, sample_size):
    dialect = sniff_dialect(path)
    rows = []
    value_counts = {}
    row_count = 0

    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, dialect=dialect)
        fieldnames = reader.fieldnames or []
        counters = {field: Counter() for field in fieldnames}

        for row in reader:
            row_count += 1
            if len(rows) < sample_size:
                rows.append(row)
            for field in fieldnames:
                value = (row.get(field) or "").strip()
                if value:
                    counters[field][value] += 1

    for field, counter in counters.items():
        lower_field = field.lower()
        if any(keyword in lower_field for keyword in KEYWORDS) or len(counter) <= 20:
            value_counts[field] = dict(counter.most_common(25))

    suspected_columns = [
        field
        for field in fieldnames
        if any(keyword in field.lower() for keyword in KEYWORDS)
    ]

    return {
        "csv_path": str(path),
        "row_count": row_count,
        "columns": fieldnames,
        "suspected_relevant_columns": suspected_columns,
        "sample_rows": rows,
        "value_counts": value_counts,
    }


def main():
    args = parse_args()
    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = inspect_csv(csv_path, args.sample_size)

    with open(output_dir / "metadata_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
