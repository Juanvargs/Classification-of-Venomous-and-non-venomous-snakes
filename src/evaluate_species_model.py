import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from tensorflow import keras


IMG_SIZE = (224, 224)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evalua un modelo multiclase por especie y su conversion a riesgo."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--classes", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--region-top-k", type=int, default=5)
    return parser.parse_args()


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def normalize(value):
    return str(value or "").lower().replace("_", " ").strip()


def load_metadata(path):
    rows = read_csv(path)
    return {row["class_folder"]: row for row in rows}


def load_manifest_regions(path):
    if not path:
        return {}
    regions = {}
    for row in read_csv(path):
        destination = Path(row["destination_path"])
        regions[str(destination).lower()] = {
            "country": row.get("country", "unknown"),
            "continent": row.get("continent", "unknown"),
        }
    return regions


def iter_images(test_dir):
    test_dir = Path(test_dir)
    for class_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
        for image_path in sorted(class_dir.glob("*")):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                yield image_path, class_dir.name


def load_image(path):
    image = keras.utils.load_img(path, target_size=IMG_SIZE)
    array = keras.utils.img_to_array(image)
    return np.expand_dims(array, axis=0)


def compatible_region(metadata_row, region):
    country = normalize(region.get("country", ""))
    continent = normalize(region.get("continent", ""))
    if not country and not continent:
        return False
    countries = normalize(metadata_row.get("countries", ""))
    continents = normalize(metadata_row.get("continents", ""))
    return (country and country != "unknown" and country in countries) or (
        continent and continent != "unknown" and continent in continents
    )


def choose_region_aware(top_indices, class_names, metadata_by_class, region):
    compatible = []
    for index in top_indices:
        class_name = class_names[index]
        metadata_row = metadata_by_class[class_name]
        if compatible_region(metadata_row, region):
            compatible.append(index)
    return compatible[0] if compatible else top_indices[0]


def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def summarize(rows, prefix):
    total = len(rows)
    top1_correct = sum(row[f"{prefix}_top1_correct"] for row in rows)
    top3_correct = sum(row[f"{prefix}_top3_correct"] for row in rows)
    top5_correct = sum(row[f"{prefix}_top5_correct"] for row in rows)
    risk_correct = sum(row[f"{prefix}_risk_correct"] for row in rows)

    venomous_rows = [row for row in rows if row["true_venom_status"] == "Venomous"]
    non_venomous_rows = [row for row in rows if row["true_venom_status"] == "Non Venomous"]
    venomous_risk_hits = sum(
        row[f"{prefix}_pred_venom_status"] == "Venomous" for row in venomous_rows
    )
    non_venomous_risk_hits = sum(
        row[f"{prefix}_pred_venom_status"] == "Non Venomous" for row in non_venomous_rows
    )

    return {
        "species_top1_accuracy": safe_divide(top1_correct, total),
        "species_top3_accuracy": safe_divide(top3_correct, total),
        "species_top5_accuracy": safe_divide(top5_correct, total),
        "risk_accuracy": safe_divide(risk_correct, total),
        "venomous_risk_recall": safe_divide(venomous_risk_hits, len(venomous_rows)),
        "non_venomous_risk_recall": safe_divide(
            non_venomous_risk_hits, len(non_venomous_rows)
        ),
        "support": total,
        "venomous_support": len(venomous_rows),
        "non_venomous_support": len(non_venomous_rows),
    }


def main():
    args = parse_args()
    model = keras.models.load_model(args.model)
    with open(args.classes, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    metadata_by_class = load_metadata(args.metadata)
    manifest_regions = load_manifest_regions(args.manifest)
    rows = []

    for image_path, true_class in iter_images(Path(args.data_dir) / "test"):
        predictions = model.predict(load_image(image_path), verbose=0)[0]
        top_indices = np.argsort(predictions)[::-1]
        top_indices = [int(index) for index in top_indices]
        top_class = class_names[top_indices[0]]
        true_metadata = metadata_by_class[true_class]
        top_metadata = metadata_by_class[top_class]

        region = manifest_regions.get(str(image_path).lower(), {})
        region_source_available = bool(region) and normalize(region.get("country")) != "unknown"
        region_indices = top_indices[: args.region_top_k]
        region_index = choose_region_aware(
            region_indices, class_names, metadata_by_class, region
        )
        region_class = class_names[region_index]
        region_metadata = metadata_by_class[region_class]

        rows.append(
            {
                "image_path": str(image_path),
                "true_class": true_class,
                "true_scientific_name": true_metadata["scientific_name"],
                "true_venom_status": true_metadata["venom_status"],
                "direct_pred_class": top_class,
                "direct_pred_scientific_name": top_metadata["scientific_name"],
                "direct_pred_venom_status": top_metadata["venom_status"],
                "direct_confidence": float(predictions[top_indices[0]]),
                "direct_top1_correct": top_class == true_class,
                "direct_top3_correct": true_class in [class_names[i] for i in top_indices[:3]],
                "direct_top5_correct": true_class in [class_names[i] for i in top_indices[:5]],
                "direct_risk_correct": top_metadata["venom_status"]
                == true_metadata["venom_status"],
                "region_pred_class": region_class,
                "region_pred_scientific_name": region_metadata["scientific_name"],
                "region_pred_venom_status": region_metadata["venom_status"],
                "region_top1_correct": region_class == true_class,
                "region_top3_correct": true_class in [class_names[i] for i in top_indices[:3]],
                "region_top5_correct": true_class in [class_names[i] for i in top_indices[:5]],
                "region_risk_correct": region_metadata["venom_status"]
                == true_metadata["venom_status"],
                "region_source_available": region_source_available,
                "country": region.get("country", ""),
                "continent": region.get("continent", ""),
            }
        )

    region_rows = [row for row in rows if row["region_source_available"]]
    payload = {
        "model": args.model,
        "classes": len(class_names),
        "top_k": args.top_k,
        "region_top_k": args.region_top_k,
        "direct": summarize(rows, "direct"),
        "region_aware_all_rows": summarize(rows, "region"),
        "region_aware_known_country_rows": summarize(region_rows, "region"),
        "true_venom_status_counts": dict(Counter(row["true_venom_status"] for row in rows)),
        "notes": [
            "direct usa solo la imagen.",
            "region_aware reordena dentro del top-k usando pais/continente cuando esta disponible.",
            "la region no reemplaza a la imagen; solo ayuda a elegir entre especies candidatas.",
        ],
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=True)

    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
