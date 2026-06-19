import argparse
import csv
import json
from pathlib import Path


BASE_SUMMARY_PATH = Path("data/metadata/species_pilot51_targeted_summary.json")
BASE_METADATA_PATH = Path("data/metadata/species_pilot51_targeted_metadata.csv")
OUTPUT_TARGETS_PATH = Path("data/collection_targets/b4_scale_up_targets.csv")
OUTPUT_SUMMARY_PATH = Path("data/collection_targets/b4_scale_up_targets_summary.json")
TAXON_QUERY_ALIASES = {
    "Corallus hortulanus": "Corallus hortulana",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crea objetivos B4 para ampliar todas las especies de forma balanceada."
    )
    parser.add_argument("--base-summary", default=str(BASE_SUMMARY_PATH))
    parser.add_argument("--base-metadata", default=str(BASE_METADATA_PATH))
    parser.add_argument("--output-targets", default=str(OUTPUT_TARGETS_PATH))
    parser.add_argument("--output-summary", default=str(OUTPUT_SUMMARY_PATH))
    parser.add_argument("--target-train-per-species", type=int, default=300)
    parser.add_argument("--external-holdout-per-species", type=int, default=5)
    parser.add_argument("--min-download-per-species", type=int, default=25)
    return parser.parse_args()


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def priority_for_gap(gap):
    if gap >= 150:
        return 1
    if gap >= 100:
        return 2
    return 3


def main():
    args = parse_args()
    base_summary = json.loads(Path(args.base_summary).read_text(encoding="utf-8"))
    metadata_rows = read_csv(Path(args.base_metadata))
    metadata_by_species = {row["scientific_name"]: row for row in metadata_rows}

    target_rows = []
    summary_rows = []
    for species in base_summary["species"]:
        scientific_name = species["scientific_name"]
        meta = metadata_by_species[scientific_name]
        query_species = TAXON_QUERY_ALIASES.get(scientific_name, scientific_name)
        current_train = int(species["train"])
        train_gap = max(0, args.target_train_per_species - current_train)
        requested = max(args.min_download_per_species, train_gap + args.external_holdout_per_species)

        row = {
            "priority": priority_for_gap(train_gap),
            "model_target": "species_b4_scale",
            "true_species": scientific_name,
            "query_species": query_species,
            "true_class": meta["venom_status"],
            "min_images": requested,
            "issue": "b4_general_scale_up",
            "confused_with": "",
            "region_focus": "",
            "current_train": current_train,
            "target_train": args.target_train_per_species,
            "train_gap": train_gap,
            "external_holdout": args.external_holdout_per_species,
            "notes": "General B4 scale-up target. Add to train only after reserving external holdout.",
        }
        target_rows.append(row)
        summary_rows.append(
            {
                "scientific_name": scientific_name,
                "query_species": query_species,
                "venom_status": meta["venom_status"],
                "current_train": current_train,
                "train_gap": train_gap,
                "requested_downloads": requested,
                "priority": row["priority"],
            }
        )

    target_rows.sort(key=lambda row: (int(row["priority"]), row["true_class"], row["true_species"]))
    summary_rows.sort(key=lambda row: (int(row["priority"]), row["venom_status"], row["scientific_name"]))

    payload = {
        "base_summary": str(args.base_summary),
        "base_metadata": str(args.base_metadata),
        "output_targets": str(args.output_targets),
        "target_train_per_species": args.target_train_per_species,
        "external_holdout_per_species": args.external_holdout_per_species,
        "min_download_per_species": args.min_download_per_species,
        "species": len(target_rows),
        "total_requested_downloads": sum(int(row["min_images"]) for row in target_rows),
        "total_train_gap": sum(int(row["train_gap"]) for row in target_rows),
        "priority_counts": {
            str(priority): sum(1 for row in target_rows if int(row["priority"]) == priority)
            for priority in [1, 2, 3]
        },
        "venom_status_counts": {
            status: sum(1 for row in target_rows if row["true_class"] == status)
            for status in ["Venomous", "Non Venomous"]
        },
        "targets": summary_rows,
    }

    write_csv(Path(args.output_targets), target_rows)
    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_summary).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
