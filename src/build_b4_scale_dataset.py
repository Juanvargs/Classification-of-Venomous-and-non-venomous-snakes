import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path


BASE_DATA_DIR = Path("data/active/Snake Species Pilot 51 Targeted")
BASE_METADATA_PATH = Path("data/metadata/species_pilot51_targeted_metadata.csv")
BASE_MANIFEST_PATH = Path("data/metadata/species_pilot51_targeted_manifest.csv")
B4_TARGETS_PATH = Path("data/collection_targets/b4_scale_up_targets.csv")
EXTRA_MANIFEST_PATH = Path("data/raw/b4_scale_up_inaturalist/manifest.csv")
OUTPUT_DATA_DIR = Path("data/active/Snake Species Pilot 51 Targeted B4 Scale")
OUTPUT_EXTERNAL_DIR = Path("data/external_tests/b4_scale_holdout")
OUTPUT_METADATA_PATH = Path("data/metadata/species_pilot51_b4_scale_metadata.csv")
OUTPUT_MANIFEST_PATH = Path("data/metadata/species_pilot51_b4_scale_manifest.csv")
OUTPUT_SUMMARY_PATH = Path("data/metadata/species_pilot51_b4_scale_summary.json")
OUTPUT_ZIP_PATH = Path("data/raw/snake_species_pilot51_b4_scale.zip")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Construye B4 agregando imagenes generales a train y reservando holdout externo."
    )
    parser.add_argument("--base-data-dir", default=str(BASE_DATA_DIR))
    parser.add_argument("--base-metadata", default=str(BASE_METADATA_PATH))
    parser.add_argument("--base-manifest", default=str(BASE_MANIFEST_PATH))
    parser.add_argument("--targets", default=str(B4_TARGETS_PATH))
    parser.add_argument("--extra-manifest", default=str(EXTRA_MANIFEST_PATH))
    parser.add_argument("--output-data-dir", default=str(OUTPUT_DATA_DIR))
    parser.add_argument("--output-external-dir", default=str(OUTPUT_EXTERNAL_DIR))
    parser.add_argument("--output-metadata", default=str(OUTPUT_METADATA_PATH))
    parser.add_argument("--output-manifest", default=str(OUTPUT_MANIFEST_PATH))
    parser.add_argument("--output-summary", default=str(OUTPUT_SUMMARY_PATH))
    parser.add_argument("--output-zip", default=str(OUTPUT_ZIP_PATH))
    parser.add_argument("--target-train-per-species", type=int, default=300)
    parser.add_argument("--external-holdout-per-species", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--make-zip", action="store_true")
    return parser.parse_args()


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_files(root):
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def copytree_fresh(source, destination, overwrite):
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"{destination} ya existe. Usa --overwrite para reconstruir."
            )
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def fresh_dir(destination, overwrite):
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"{destination} ya existe. Usa --overwrite para reconstruir."
            )
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)


def safe_name(value):
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def zip_dir(source_dir, zip_path):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(image_files(source_dir)):
            archive.write(path, path.relative_to(source_dir.parent))


def main():
    args = parse_args()
    base_data_dir = Path(args.base_data_dir)
    output_data_dir = Path(args.output_data_dir)
    output_external_dir = Path(args.output_external_dir)

    base_metadata = read_csv(Path(args.base_metadata))
    base_manifest = read_csv(Path(args.base_manifest))
    targets = read_csv(Path(args.targets))
    extra_manifest = read_csv(Path(args.extra_manifest))

    metadata_by_species = {row["scientific_name"]: row for row in base_metadata}
    metadata_by_class = {row["class_folder"]: row for row in base_metadata}
    targets_by_species = {row["true_species"]: row for row in targets}

    copytree_fresh(base_data_dir, output_data_dir, args.overwrite)
    fresh_dir(output_external_dir, args.overwrite)

    base_hashes = {file_hash(path) for path in image_files(base_data_dir)}

    output_manifest = []
    for row in base_manifest:
        copied = dict(row)
        source_path = Path(row["destination_path"])
        relative = source_path.relative_to(base_data_dir)
        copied["destination_path"] = str(output_data_dir / relative)
        output_manifest.append(copied)

    rows_by_species = defaultdict(list)
    skipped = []
    seen_extra_hashes = set()

    for row in extra_manifest:
        species = row["scientific_name"]
        meta = metadata_by_species.get(species)
        if not meta or species not in targets_by_species:
            skipped.append({**row, "reason": "species not in B4 targets"})
            continue

        source = Path(row["image_path"])
        if not source.exists():
            skipped.append({**row, "reason": "source image missing"})
            continue

        digest = file_hash(source)
        if digest in base_hashes:
            skipped.append({**row, "reason": "duplicate of base dataset"})
            continue
        if digest in seen_extra_hashes:
            skipped.append({**row, "reason": "duplicate inside B4 extras"})
            continue
        seen_extra_hashes.add(digest)

        prepared = dict(row)
        prepared["sha256"] = digest
        rows_by_species[species].append(prepared)

    added_by_class = defaultdict(int)
    external_rows = []
    remaining_gap_by_species = {}

    for species, meta in metadata_by_species.items():
        class_folder = meta["class_folder"]
        base_train = int(metadata_by_class[class_folder]["train_count_source"])
        target_train = args.target_train_per_species
        train_needed = max(0, target_train - base_train)
        remaining_gap_by_species[species] = train_needed

        candidates = sorted(
            rows_by_species.get(species, []),
            key=lambda row: (row.get("source_page", ""), row["image_path"]),
        )
        holdout = candidates[: args.external_holdout_per_species]
        train_candidates = candidates[args.external_holdout_per_species :]
        selected_train = train_candidates[:train_needed]

        external_class_dir = output_external_dir / meta["venom_status"].replace(" ", "_")
        external_class_dir.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(holdout, start=1):
            source = Path(row["image_path"])
            destination = external_class_dir / (
                f"{meta['class_id']}_{safe_name(species)}_holdout_{index:03d}{source.suffix.lower()}"
            )
            shutil.copy2(source, destination)
            external_rows.append(
                {
                    "id": f"b4_{meta['class_id']}_{index:03d}",
                    "image_path": str(destination),
                    "true_class": meta["venom_status"],
                    "true_species": species,
                    "common_name": meta["common_name"],
                    "country_for_test": row.get("observed_place", ""),
                    "source_page": row.get("source_page", ""),
                    "license": row.get("license", ""),
                    "sha256": row["sha256"],
                }
            )

        train_class_dir = output_data_dir / "train" / class_folder
        train_class_dir.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(selected_train, start=1):
            source = Path(row["image_path"])
            destination = train_class_dir / (
                f"b4extra_{safe_name(species)}_{index:04d}{source.suffix.lower()}"
            )
            shutil.copy2(source, destination)
            added_by_class[class_folder] += 1
            output_manifest.append(
                {
                    "split": "train",
                    "class_folder": class_folder,
                    "class_id": meta["class_id"],
                    "scientific_name": species,
                    "common_name": meta["common_name"],
                    "venom_status": meta["venom_status"],
                    "country": row.get("observed_place", ""),
                    "continent": "",
                    "uuid": source.stem,
                    "source_member": row.get("source_page", ""),
                    "destination_path": str(destination),
                }
            )

    output_metadata = []
    species_summary = []
    for row in base_metadata:
        copied = dict(row)
        class_folder = row["class_folder"]
        added = added_by_class[class_folder]
        copied["train_count_source"] = str(int(copied["train_count_source"]) + added)
        output_metadata.append(copied)
        species_summary.append(
            {
                "class_folder": class_folder,
                "scientific_name": row["scientific_name"],
                "venom_status": row["venom_status"],
                "base_train_count_source": int(metadata_by_class[class_folder]["train_count_source"]),
                "added_train": added,
                "target_train": args.target_train_per_species,
                "remaining_train_gap": max(0, remaining_gap_by_species[row["scientific_name"]] - added),
                "external_holdout": sum(
                    1 for item in external_rows if item["true_species"] == row["scientific_name"]
                ),
                "available_unique_extra": len(rows_by_species.get(row["scientific_name"], [])),
            }
        )

    images_by_split = {
        split: len(image_files(output_data_dir / split)) for split in ["train", "val", "test"]
    }
    summary = {
        "base_data_dir": str(base_data_dir),
        "output_data_dir": str(output_data_dir),
        "output_external_dir": str(output_external_dir),
        "extra_manifest": str(args.extra_manifest),
        "target_train_per_species": args.target_train_per_species,
        "external_holdout_per_species": args.external_holdout_per_species,
        "added_train_images": sum(added_by_class.values()),
        "external_holdout_images": len(external_rows),
        "skipped_images": len(skipped),
        "images_by_split": images_by_split,
        "species": species_summary,
        "skipped": skipped,
    }

    write_csv(Path(args.output_metadata), output_metadata)
    write_csv(Path(args.output_manifest), output_manifest)
    write_csv(output_external_dir / "manifest.csv", external_rows)
    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.make_zip:
        zip_dir(output_data_dir, Path(args.output_zip))
        summary["zip_path"] = str(args.output_zip)
        Path(args.output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
