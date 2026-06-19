import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Construye un catalogo de especies a partir de la metadata disponible."
    )
    parser.add_argument("--species-json", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    return parser.parse_args()


def normalize_text(value):
    value = "" if value is None else str(value).strip()
    return value if value else "unknown"


def read_species_json(path):
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    species_by_class = {}
    for item in payload.get("species", []):
        class_id = str(item.get("class_id", "")).strip()
        taxonomy = item.get("taxonomy", {})
        characteristics = item.get("characteristics", {})
        distribution = item.get("distribution", {})
        dataset_info = item.get("dataset_info", {})

        countries = [normalize_text(country) for country in distribution.get("countries", [])]
        continents = [normalize_text(continent) for continent in distribution.get("continents", [])]
        species_by_class[class_id] = {
            "class_id": class_id,
            "scientific_name": normalize_text(taxonomy.get("binomial_name")),
            "common_name": normalize_text(taxonomy.get("common_name")),
            "genus": normalize_text(taxonomy.get("genus")),
            "family": normalize_text(taxonomy.get("family")),
            "snake_sub_family": normalize_text(taxonomy.get("snake_sub_family")),
            "poisonous": bool(characteristics.get("poisonous")),
            "data1_sample_count": int(dataset_info.get("sample_count") or 0),
            "countries": set(countries),
            "continents": set(continents),
        }
    return species_by_class


def read_split_metadata(path, split_name):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {
                    "split": split_name,
                    "class_id": normalize_text(row.get("class_id")),
                    "scientific_name": normalize_text(row.get("binomial")),
                    "country": normalize_text(row.get("country")),
                    "continent": normalize_text(row.get("continent")),
                    "poisonous": normalize_text(row.get("poisonous")),
                }
            )
    return rows


def join_values(values):
    known = sorted(value for value in values if value and value.lower() != "unknown")
    unknown = sorted(value for value in values if value and value.lower() == "unknown")
    return " | ".join(known + unknown)


def build_catalog(species_by_class, split_rows):
    counts_by_class = defaultdict(Counter)
    countries_by_class = defaultdict(set)
    continents_by_class = defaultdict(set)
    names_by_class = defaultdict(Counter)
    poisonous_by_class = defaultdict(Counter)

    for row in split_rows:
        class_id = row["class_id"]
        counts_by_class[class_id][row["split"]] += 1
        countries_by_class[class_id].add(row["country"])
        continents_by_class[class_id].add(row["continent"])
        names_by_class[class_id][row["scientific_name"]] += 1
        poisonous_by_class[class_id][row["poisonous"]] += 1

    all_class_ids = sorted(
        set(species_by_class.keys()) | set(counts_by_class.keys()),
        key=lambda value: int(value) if value.isdigit() else value,
    )

    catalog = []
    for class_id in all_class_ids:
        species = species_by_class.get(class_id, {})
        scientific_name = species.get("scientific_name")
        if not scientific_name or scientific_name == "unknown":
            scientific_name = names_by_class[class_id].most_common(1)[0][0]

        countries = set(species.get("countries", set())) | countries_by_class[class_id]
        continents = set(species.get("continents", set())) | continents_by_class[class_id]
        poisonous = species.get("poisonous")
        if poisonous is None:
            poisonous = poisonous_by_class[class_id].most_common(1)[0][0] == "1"

        train_count = counts_by_class[class_id]["train"]
        test_count = counts_by_class[class_id]["test"]
        total_count = train_count + test_count
        known_countries = [country for country in countries if country.lower() != "unknown"]

        catalog.append(
            {
                "class_id": class_id,
                "scientific_name": scientific_name,
                "common_name": species.get("common_name", "unknown"),
                "venom_status": "Venomous" if poisonous else "Non Venomous",
                "poisonous": "1" if poisonous else "0",
                "genus": species.get("genus", "unknown"),
                "family": species.get("family", "unknown"),
                "snake_sub_family": species.get("snake_sub_family", "unknown"),
                "countries": join_values(countries),
                "continents": join_values(continents),
                "known_country_count": len(set(known_countries)),
                "train_count": train_count,
                "test_count": test_count,
                "total_data2_count": total_count,
                "data1_sample_count": species.get("data1_sample_count", 0),
                "region_metadata_available": "yes" if known_countries else "no",
                "source_note": "DATA_1 metadata + DATA_2 train/test metadata",
            }
        )
    return catalog


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "class_id",
        "scientific_name",
        "common_name",
        "venom_status",
        "poisonous",
        "genus",
        "family",
        "snake_sub_family",
        "countries",
        "continents",
        "known_country_count",
        "train_count",
        "test_count",
        "total_data2_count",
        "data1_sample_count",
        "region_metadata_available",
        "source_note",
    ]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    venom_counter = Counter(row["venom_status"] for row in rows)
    train_ready = [
        row for row in rows if int(row["train_count"]) >= 40 and int(row["test_count"]) >= 10
    ]
    payload = {
        "species_total": len(rows),
        "venom_status_counts": dict(venom_counter),
        "species_with_region_metadata": sum(
            1 for row in rows if row["region_metadata_available"] == "yes"
        ),
        "species_with_data2_images": sum(1 for row in rows if int(row["total_data2_count"]) > 0),
        "species_train_ready_min_40_train_10_test": len(train_ready),
        "top_species_by_images": sorted(
            rows, key=lambda row: int(row["total_data2_count"]), reverse=True
        )[:15],
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=True)


def main():
    args = parse_args()
    species_by_class = read_species_json(args.species_json)
    split_rows = read_split_metadata(args.train_csv, "train") + read_split_metadata(
        args.test_csv, "test"
    )
    catalog = build_catalog(species_by_class, split_rows)
    write_csv(args.output_csv, catalog)
    write_summary(args.summary_json, catalog)
    print(f"Species catalog written: {args.output_csv}")
    print(f"Summary written: {args.summary_json}")


if __name__ == "__main__":
    main()
