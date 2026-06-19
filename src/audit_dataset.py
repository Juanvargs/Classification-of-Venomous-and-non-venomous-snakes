import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageOps, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(description="Audita un dataset de imagenes por split y clase.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="reports/audits/dataset_audit")
    parser.add_argument("--hash-size", type=int, default=8)
    return parser.parse_args()


def average_hash(image, hash_size):
    image = ImageOps.grayscale(image).resize((hash_size, hash_size))
    pixels = list(image.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if pixel >= avg else "0" for pixel in pixels)
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"


def file_sha1(path):
    digest = hashlib.sha1()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_images(data_dir):
    data_dir = Path(data_dir)
    for split_dir in data_dir.iterdir():
        if not split_dir.is_dir():
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            for path in class_dir.rglob("*"):
                if path.is_file():
                    yield split_dir.name, class_dir.name, path


def audit_dataset(data_dir, hash_size):
    rows = []
    corrupt = []
    exact_hashes = defaultdict(list)
    perceptual_hashes = defaultdict(list)

    for split, class_name, path in iter_images(data_dir):
        extension = path.suffix.lower()
        row = {
            "split": split,
            "class": class_name,
            "path": str(path),
            "file_name": path.name,
            "extension": extension,
            "bytes": path.stat().st_size,
            "width": None,
            "height": None,
            "mode": None,
            "exact_sha1": None,
            "average_hash": None,
            "status": "ok",
        }

        if extension not in IMAGE_EXTENSIONS:
            row["status"] = "unsupported_extension"
            rows.append(row)
            continue

        try:
            exact_hash = file_sha1(path)
            with Image.open(path) as image:
                image.load()
                row["width"], row["height"] = image.size
                row["mode"] = image.mode
                row["exact_sha1"] = exact_hash
                row["average_hash"] = average_hash(image, hash_size)
        except (OSError, UnidentifiedImageError) as error:
            row["status"] = "corrupt"
            row["error"] = str(error)
            corrupt.append(row)

        rows.append(row)
        if row["exact_sha1"]:
            exact_hashes[row["exact_sha1"]].append(row)
        if row["average_hash"]:
            perceptual_hashes[row["average_hash"]].append(row)

    return rows, corrupt, exact_hashes, perceptual_hashes


def summarize(rows, exact_hashes, perceptual_hashes):
    ok_rows = [row for row in rows if row["status"] == "ok"]
    counts = Counter((row["split"], row["class"]) for row in ok_rows)
    extensions = Counter(row["extension"] for row in rows)
    modes = Counter(row["mode"] for row in ok_rows)
    sizes = [(row["width"], row["height"]) for row in ok_rows]
    exact_duplicates = {key: value for key, value in exact_hashes.items() if len(value) > 1}
    perceptual_duplicates = {
        key: value for key, value in perceptual_hashes.items() if len(value) > 1
    }

    return {
        "total_files": len(rows),
        "valid_images": len(ok_rows),
        "problem_files": len(rows) - len(ok_rows),
        "counts_by_split_class": {
            f"{split}/{class_name}": count
            for (split, class_name), count in sorted(counts.items())
        },
        "extensions": dict(extensions),
        "modes": {str(key): value for key, value in modes.items()},
        "unique_sizes": len(set(sizes)),
        "min_width": min((width for width, _ in sizes), default=None),
        "max_width": max((width for width, _ in sizes), default=None),
        "min_height": min((height for _, height in sizes), default=None),
        "max_height": max((height for _, height in sizes), default=None),
        "exact_duplicate_groups": len(exact_duplicates),
        "exact_duplicate_files": sum(len(group) for group in exact_duplicates.values()),
        "average_hash_duplicate_groups": len(perceptual_duplicates),
        "average_hash_duplicate_files": sum(
            len(group) for group in perceptual_duplicates.values()
        ),
    }


def save_csv(rows, path):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_duplicate_groups(groups, path):
    rows = []
    for group_id, group in groups.items():
        for row in group:
            rows.append(
                {
                    "group": group_id,
                    "split": row["split"],
                    "class": row["class"],
                    "path": row["path"],
                    "file_name": row["file_name"],
                }
            )
    save_csv(rows, path)


def save_count_plot(rows, output_dir):
    ok_rows = [row for row in rows if row["status"] == "ok"]
    counts = Counter((row["split"], row["class"]) for row in ok_rows)
    labels = [f"{split}\n{class_name}" for split, class_name in sorted(counts)]
    values = [counts[key] for key in sorted(counts)]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.ylabel("Imagenes")
    plt.title("Distribucion por split y clase")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=160)
    plt.close()


def write_markdown(summary, output_dir):
    lines = [
        "# Auditoria del dataset",
        "",
        "## Resumen",
        "",
        f"- Archivos totales: {summary['total_files']}",
        f"- Imagenes validas: {summary['valid_images']}",
        f"- Archivos con problemas: {summary['problem_files']}",
        f"- Grupos de duplicados exactos: {summary['exact_duplicate_groups']}",
        f"- Archivos en duplicados exactos: {summary['exact_duplicate_files']}",
        f"- Grupos con hash perceptual repetido: {summary['average_hash_duplicate_groups']}",
        f"- Archivos con hash perceptual repetido: {summary['average_hash_duplicate_files']}",
        "",
        "## Distribucion",
        "",
        "| Split / clase | Imagenes |",
        "| --- | ---: |",
    ]
    for key, value in summary["counts_by_split_class"].items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Dimensiones",
            "",
            f"- Tamanos unicos: {summary['unique_sizes']}",
            f"- Ancho minimo: {summary['min_width']}",
            f"- Ancho maximo: {summary['max_width']}",
            f"- Alto minimo: {summary['min_height']}",
            f"- Alto maximo: {summary['max_height']}",
            "",
            "## Archivos generados",
            "",
            "- `dataset_files.csv`",
            "- `exact_duplicates.csv`",
            "- `average_hash_duplicates.csv`",
            "- `summary.json`",
            "- `class_distribution.png`",
        ]
    )

    with open(output_dir / "README.md", "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, corrupt, exact_hashes, perceptual_hashes = audit_dataset(
        args.data_dir,
        args.hash_size,
    )
    exact_duplicates = {key: value for key, value in exact_hashes.items() if len(value) > 1}
    perceptual_duplicates = {
        key: value for key, value in perceptual_hashes.items() if len(value) > 1
    }
    summary = summarize(rows, exact_hashes, perceptual_hashes)

    save_csv(rows, output_dir / "dataset_files.csv")
    save_csv(corrupt, output_dir / "problem_files.csv")
    save_duplicate_groups(exact_duplicates, output_dir / "exact_duplicates.csv")
    save_duplicate_groups(perceptual_duplicates, output_dir / "average_hash_duplicates.csv")
    save_count_plot(rows, output_dir)
    write_markdown(summary, output_dir)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
