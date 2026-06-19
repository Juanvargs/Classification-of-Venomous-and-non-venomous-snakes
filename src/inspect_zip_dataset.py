import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TABLE_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".json"}


def parse_args():
    parser = argparse.ArgumentParser(description="Inspecciona la estructura de un dataset en ZIP.")
    parser.add_argument("--zip-path", required=True)
    parser.add_argument("--output-dir", default="reports/audits/dataset_sources/zip_inspection")
    parser.add_argument("--sample-size", type=int, default=120)
    return parser.parse_args()


def safe_parts(name):
    return [part for part in Path(name).parts if part not in ("", ".")]


def inspect_zip(zip_path):
    with zipfile.ZipFile(zip_path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]

    extensions = Counter(Path(info.filename).suffix.lower() for info in infos)
    image_entries = [
        info.filename
        for info in infos
        if Path(info.filename).suffix.lower() in IMAGE_EXTENSIONS
    ]
    table_entries = [
        info.filename
        for info in infos
        if Path(info.filename).suffix.lower() in TABLE_EXTENSIONS
    ]

    top_level = Counter()
    first_two_levels = Counter()
    first_three_levels = Counter()

    for info in infos:
        parts = safe_parts(info.filename)
        if parts:
            top_level[parts[0]] += 1
        if len(parts) >= 2:
            first_two_levels["/".join(parts[:2])] += 1
        if len(parts) >= 3:
            first_three_levels["/".join(parts[:3])] += 1

    return {
        "zip_path": str(zip_path),
        "total_files": len(infos),
        "image_files": len(image_entries),
        "table_or_metadata_files": len(table_entries),
        "extensions": dict(sorted(extensions.items())),
        "top_level": dict(top_level.most_common(30)),
        "first_two_levels": dict(first_two_levels.most_common(50)),
        "first_three_levels": dict(first_three_levels.most_common(80)),
        "metadata_files": table_entries[:80],
        "image_samples": image_entries[:80],
        "all_samples": [info.filename for info in infos[:120]],
    }


def main():
    args = parse_args()
    zip_path = Path(args.zip_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = inspect_zip(zip_path)

    with open(output_dir / "zip_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    with open(output_dir / "zip_entries_sample.txt", "w", encoding="utf-8") as file:
        for item in summary["all_samples"][: args.sample_size]:
            file.write(f"{item}\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
