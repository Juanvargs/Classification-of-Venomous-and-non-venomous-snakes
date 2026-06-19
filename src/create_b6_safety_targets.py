import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


B5_RESULTS_PATH = Path(
    "data/external_tests/b4_scale_holdout/reports/current_app_flow_b5/current_app_flow_results.csv"
)
B4_METADATA_PATH = Path("data/metadata/species_pilot51_b4_scale_metadata.csv")
OUTPUT_TARGETS_PATH = Path("data/collection_targets/b6_safety_targets.csv")
OUTPUT_SUMMARY_PATH = Path("data/collection_targets/b6_safety_targets_summary.json")

TAXON_QUERY_ALIASES = {
    "Corallus hortulanus": "Corallus hortulana",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Crea objetivos B6 desde errores B5 donde especies venenosas caen "
            "visualmente como no venenosas en el flujo sin region."
        )
    )
    parser.add_argument("--b5-results", default=str(B5_RESULTS_PATH))
    parser.add_argument("--metadata", default=str(B4_METADATA_PATH))
    parser.add_argument("--output-targets", default=str(OUTPUT_TARGETS_PATH))
    parser.add_argument("--output-summary", default=str(OUTPUT_SUMMARY_PATH))
    parser.add_argument("--priority1-images", type=int, default=180)
    parser.add_argument("--priority2-images", type=int, default=130)
    parser.add_argument("--priority3-images", type=int, default=90)
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


def priority_for_species(rows):
    dangerous_false_safe = any(row["integrated_decision"] == "Non Venomous" for row in rows)
    high_confidence_miss = any(float(row["species_confidence"]) >= 0.50 for row in rows)
    if dangerous_false_safe:
        return 1
    if high_confidence_miss or len(rows) >= 2:
        return 2
    return 3


def requested_images(priority, args):
    if priority == 1:
        return args.priority1_images
    if priority == 2:
        return args.priority2_images
    return args.priority3_images


def main():
    args = parse_args()
    results = read_csv(Path(args.b5_results))
    metadata = read_csv(Path(args.metadata))
    metadata_by_species = {row["scientific_name"]: row for row in metadata}

    risky_rows = [
        row
        for row in results
        if row["flow"] == "sin_region"
        and row["true_class"] == "Venomous"
        and row["species_risk"] == "Non Venomous"
    ]

    rows_by_species = defaultdict(list)
    confused_by_species = defaultdict(Counter)
    for row in risky_rows:
        rows_by_species[row["true_species"]].append(row)
        confused_by_species[row["true_species"]][row["species_prediction"]] += 1

    target_rows = []
    summary_rows = []
    for species, rows in rows_by_species.items():
        meta = metadata_by_species[species]
        priority = priority_for_species(rows)
        min_images = requested_images(priority, args)
        confused_with = " | ".join(
            species_name
            for species_name, _ in confused_by_species[species].most_common()
        )
        query_species = TAXON_QUERY_ALIASES.get(species, species)
        dangerous_count = sum(1 for row in rows if row["integrated_decision"] == "Non Venomous")
        high_confidence_count = sum(1 for row in rows if float(row["species_confidence"]) >= 0.50)

        target_rows.append(
            {
                "priority": priority,
                "model_target": "species_b6_safety",
                "true_species": species,
                "query_species": query_species,
                "true_class": meta["venom_status"],
                "min_images": min_images,
                "issue": "b6_venomous_as_non_venomous",
                "confused_with": confused_with,
                "region_focus": "",
                "current_train": meta["train_count_source"],
                "target_train": int(meta["train_count_source"]) + min_images,
                "train_gap": min_images,
                "external_holdout": 0,
                "dangerous_false_safe_count": dangerous_count,
                "visual_false_non_venomous_count": len(rows),
                "high_confidence_miss_count": high_confidence_count,
                "notes": (
                    "B6 safety target from B5 holdout sin_region. Add to train; "
                    "keep B4 holdout fixed for evaluation."
                ),
            }
        )
        summary_rows.append(
            {
                "scientific_name": species,
                "priority": priority,
                "current_train": int(meta["train_count_source"]),
                "requested_images": min_images,
                "visual_false_non_venomous_count": len(rows),
                "dangerous_false_safe_count": dangerous_count,
                "high_confidence_miss_count": high_confidence_count,
                "confused_with": confused_with,
            }
        )

    target_rows.sort(key=lambda row: (int(row["priority"]), row["true_species"]))
    summary_rows.sort(key=lambda row: (int(row["priority"]), row["scientific_name"]))

    payload = {
        "source_results": str(args.b5_results),
        "metadata": str(args.metadata),
        "output_targets": str(args.output_targets),
        "model_target": "species_b6_safety",
        "selection_rule": (
            "Venomous true class predicted visually as Non Venomous by B5 in sin_region."
        ),
        "visual_false_non_venomous_cases": len(risky_rows),
        "target_species": len(target_rows),
        "total_requested_images": sum(int(row["min_images"]) for row in target_rows),
        "priority_counts": {
            str(priority): sum(1 for row in target_rows if int(row["priority"]) == priority)
            for priority in [1, 2, 3]
        },
        "targets": summary_rows,
    }

    write_csv(Path(args.output_targets), target_rows)
    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_summary).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
