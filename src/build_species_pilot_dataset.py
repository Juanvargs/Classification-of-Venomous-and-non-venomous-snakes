import argparse
import csv
import json
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crea un dataset piloto por especie desde DATA_2."
    )
    parser.add_argument("--zip-path", required=True)
    parser.add_argument("--candidates-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument(
        "--selected-species-csv",
        default="",
        help=(
            "CSV opcional con columna scientific_name para construir un piloto "
            "controlado en lugar de elegir especies solo por abundancia."
        ),
    )
    parser.add_argument("--species-per-venom-status", type=int, default=10)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--min-train", type=int, default=40)
    parser.add_argument("--min-test", type=int, default=10)
    return parser.parse_args()


def slugify(value):
    value = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    return re.sub(r"_+", "_", value).strip("_")


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_zip_csv(zip_path, member):
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as file:
            text = file.read().decode("utf-8-sig").splitlines()
    return list(csv.DictReader(text))


def choose_species(candidates, per_status, min_train, min_test):
    selected = []
    for status in ["Venomous", "Non Venomous"]:
        valid = [
            row
            for row in candidates
            if row["venom_status"] == status
            and int(row["train_count"]) >= min_train
            and int(row["test_count"]) >= min_test
        ]
        valid.sort(key=lambda row: (-int(row["total_data2_count"]), row["scientific_name"]))
        selected.extend(valid[:per_status])
    selected.sort(key=lambda row: row["scientific_name"])
    return selected


def choose_selected_species(candidates, selected_species_csv, min_train, min_test):
    selected_rows = read_csv(selected_species_csv)
    requested_names = []
    for row in selected_rows:
        name = row.get("scientific_name", "").strip()
        if name and name not in requested_names:
            requested_names.append(name)

    candidates_by_name = {row["scientific_name"]: row for row in candidates}
    missing = [name for name in requested_names if name not in candidates_by_name]
    if missing:
        raise ValueError(f"Especies no encontradas en candidates-csv: {missing}")

    insufficient = []
    selected = []
    for name in requested_names:
        row = candidates_by_name[name]
        if int(row["train_count"]) < min_train or int(row["test_count"]) < min_test:
            insufficient.append(
                {
                    "scientific_name": name,
                    "train_count": row["train_count"],
                    "test_count": row["test_count"],
                }
            )
            continue
        selected.append(row)

    if insufficient:
        raise ValueError(
            "Especies con datos insuficientes para el piloto controlado: "
            f"{insufficient}"
        )

    selected.sort(key=lambda row: row["scientific_name"])
    return selected


def folder_name(row):
    return f"{int(row['class_id']):03d}_{slugify(row['scientific_name'])}"


def group_rows_by_class(rows, selected_class_ids):
    grouped = defaultdict(list)
    for row in rows:
        class_id = str(row.get("class_id", "")).strip()
        if class_id in selected_class_ids:
            grouped[class_id].append(row)
    return grouped


def split_train_val(rows, val_fraction):
    rows = list(rows)
    random.Random(SEED).shuffle(rows)
    val_count = max(1, int(round(len(rows) * val_fraction)))
    return rows[val_count:], rows[:val_count]


def image_member(split, row):
    return f"{split}/{row['class_id']}/{row['UUID']}.jpg"


def copy_rows_from_zip(zip_path, rows, source_split, destination_split, output_dir, species_row):
    copied = []
    missing = []
    destination_class = folder_name(species_row)
    with zipfile.ZipFile(zip_path) as archive:
        members = set(archive.namelist())
        for row in rows:
            member = image_member(source_split, row)
            if member not in members:
                missing.append(member)
                continue

            destination = (
                Path(output_dir)
                / destination_split
                / destination_class
                / f"{row['UUID']}.jpg"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source_file, open(destination, "wb") as target_file:
                shutil.copyfileobj(source_file, target_file)

            copied.append(
                {
                    "split": destination_split,
                    "class_folder": destination_class,
                    "class_id": species_row["class_id"],
                    "scientific_name": species_row["scientific_name"],
                    "common_name": species_row["common_name"],
                    "venom_status": species_row["venom_status"],
                    "country": row.get("country", "unknown"),
                    "continent": row.get("continent", "unknown"),
                    "uuid": row["UUID"],
                    "source_member": member,
                    "destination_path": str(destination),
                }
            )
    return copied, missing


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames and rows:
        fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(manifest_rows, selected_species, missing):
    by_split = Counter(row["split"] for row in manifest_rows)
    by_status_split = Counter((row["venom_status"], row["split"]) for row in manifest_rows)
    by_species = Counter((row["scientific_name"], row["split"]) for row in manifest_rows)
    return {
        "selected_species": len(selected_species),
        "species_per_venom_status": dict(Counter(row["venom_status"] for row in selected_species)),
        "images_by_split": dict(by_split),
        "images_by_venom_status_and_split": {
            f"{status}/{split}": count
            for (status, split), count in sorted(by_status_split.items())
        },
        "missing_images": len(missing),
        "species": [
            {
                "class_folder": folder_name(row),
                "class_id": row["class_id"],
                "scientific_name": row["scientific_name"],
                "common_name": row["common_name"],
                "venom_status": row["venom_status"],
                "train": by_species[(row["scientific_name"], "train")],
                "val": by_species[(row["scientific_name"], "val")],
                "test": by_species[(row["scientific_name"], "test")],
            }
            for row in selected_species
        ],
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"La carpeta ya existe: {output_dir}")

    candidates = read_csv(args.candidates_csv)
    if args.selected_species_csv:
        selected_species = choose_selected_species(
            candidates,
            args.selected_species_csv,
            args.min_train,
            args.min_test,
        )
    else:
        selected_species = choose_species(
            candidates,
            args.species_per_venom_status,
            args.min_train,
            args.min_test,
        )
    selected_by_class = {row["class_id"]: row for row in selected_species}
    selected_class_ids = set(selected_by_class)

    train_rows = read_zip_csv(args.zip_path, "Csv/train.csv")
    test_rows = read_zip_csv(args.zip_path, "Csv/test.csv")
    train_by_class = group_rows_by_class(train_rows, selected_class_ids)
    test_by_class = group_rows_by_class(test_rows, selected_class_ids)

    manifest_rows = []
    missing = []
    for class_id, species_row in selected_by_class.items():
        train_split, val_split = split_train_val(
            train_by_class[class_id], args.val_fraction
        )
        copied, missing_rows = copy_rows_from_zip(
            args.zip_path, train_split, "train", "train", output_dir, species_row
        )
        manifest_rows.extend(copied)
        missing.extend(missing_rows)

        copied, missing_rows = copy_rows_from_zip(
            args.zip_path, val_split, "train", "val", output_dir, species_row
        )
        manifest_rows.extend(copied)
        missing.extend(missing_rows)

        copied, missing_rows = copy_rows_from_zip(
            args.zip_path, test_by_class[class_id], "test", "test", output_dir, species_row
        )
        manifest_rows.extend(copied)
        missing.extend(missing_rows)

    metadata_rows = []
    for row in selected_species:
        metadata_rows.append(
            {
                "class_folder": folder_name(row),
                "class_id": row["class_id"],
                "scientific_name": row["scientific_name"],
                "common_name": row["common_name"],
                "venom_status": row["venom_status"],
                "countries": row["countries"],
                "continents": row["continents"],
                "train_count_source": row["train_count"],
                "test_count_source": row["test_count"],
            }
        )

    write_csv(args.manifest, manifest_rows)
    write_csv(args.metadata_output, metadata_rows)

    summary = summarize(manifest_rows, selected_species, missing)
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_output, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=True)

    print(f"Dataset written: {output_dir}")
    print(f"Manifest written: {args.manifest}")
    print(f"Metadata written: {args.metadata_output}")
    print(f"Summary written: {args.summary_output}")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
