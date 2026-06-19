import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tensorflow import keras


IMG_SIZE = (224, 224)
EPSILON = 1e-9


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compara formas de usar pais/region en prediccion por especie."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--classes", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--soft-boosts", default="1.5,2,3,5")
    parser.add_argument("--bayes-alphas", default="0.25,0.5,1")
    return parser.parse_args()


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def normalize(value):
    return str(value or "").lower().replace("_", " ").strip()


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def load_metadata(path):
    return {row["class_folder"]: row for row in read_csv(path)}


def load_manifest(path):
    rows = read_csv(path)
    by_path = {}
    for row in rows:
        destination = str(Path(row["destination_path"])).lower()
        by_path[destination] = row
    return rows, by_path


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
    if (not country or country == "unknown") and (
        not continent or continent == "unknown"
    ):
        return False

    countries = normalize(metadata_row.get("countries", ""))
    continents = normalize(metadata_row.get("continents", ""))
    country_match = country and country != "unknown" and country in countries
    continent_match = continent and continent != "unknown" and continent in continents
    return bool(country_match or continent_match)


def build_geo_priors(manifest_rows, class_names):
    train_rows = [row for row in manifest_rows if row.get("split") in {"train", "val"}]
    class_counts = Counter(row["class_folder"] for row in train_rows)
    total_class_count = sum(class_counts.values())
    class_prior = {
        class_name: (class_counts[class_name] + 1) / (total_class_count + len(class_names))
        for class_name in class_names
    }

    country_counts = defaultdict(Counter)
    continent_counts = defaultdict(Counter)
    for row in train_rows:
        class_name = row["class_folder"]
        country = normalize(row.get("country", ""))
        continent = normalize(row.get("continent", ""))
        if country and country != "unknown":
            country_counts[country][class_name] += 1
        if continent and continent != "unknown":
            continent_counts[continent][class_name] += 1

    return {
        "class_prior": class_prior,
        "country_counts": country_counts,
        "continent_counts": continent_counts,
    }


def location_prior_for_class(class_name, region, priors, class_count):
    country = normalize(region.get("country", ""))
    continent = normalize(region.get("continent", ""))
    counts = None

    if country and country != "unknown" and country in priors["country_counts"]:
        counts = priors["country_counts"][country]
    elif continent and continent != "unknown" and continent in priors["continent_counts"]:
        counts = priors["continent_counts"][continent]

    if not counts:
        return None

    total = sum(counts.values())
    return (counts[class_name] + 1) / (total + class_count)


def choose_top_region_match(predictions, class_names, metadata_by_class, region, top_k):
    top_indices = np.argsort(predictions)[::-1]
    top_indices = [int(index) for index in top_indices[:top_k]]
    for index in top_indices:
        if compatible_region(metadata_by_class[class_names[index]], region):
            return index
    return top_indices[0]


def choose_hard_filter(predictions, class_names, metadata_by_class, region):
    compatible_indices = [
        index
        for index, class_name in enumerate(class_names)
        if compatible_region(metadata_by_class[class_name], region)
    ]
    if not compatible_indices:
        return int(np.argmax(predictions))
    return max(compatible_indices, key=lambda index: predictions[index])


def choose_soft_boost(predictions, class_names, metadata_by_class, region, boost):
    scores = np.array(predictions, dtype="float64")
    for index, class_name in enumerate(class_names):
        if compatible_region(metadata_by_class[class_name], region):
            scores[index] *= boost
    return int(np.argmax(scores))


def choose_bayes_prior(predictions, class_names, region, priors, alpha):
    scores = []
    for index, class_name in enumerate(class_names):
        location_prior = location_prior_for_class(
            class_name, region, priors, len(class_names)
        )
        if location_prior is None:
            return int(np.argmax(predictions))

        visual = max(float(predictions[index]), EPSILON)
        class_prior = max(float(priors["class_prior"][class_name]), EPSILON)
        geo_ratio = max(location_prior / class_prior, EPSILON)
        score = math.log(visual) + alpha * math.log(geo_ratio)
        scores.append(score)
    return int(np.argmax(scores))


def empty_summary():
    return {
        "support": 0,
        "species_top1_accuracy": 0.0,
        "risk_accuracy": 0.0,
        "venomous_risk_recall": 0.0,
        "non_venomous_risk_recall": 0.0,
        "venomous_false_negatives": 0,
        "non_venomous_false_positives": 0,
        "changed_from_image_only": 0,
        "risk_improved_vs_image_only": 0,
        "risk_worsened_vs_image_only": 0,
    }


def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def summarize(rows, strategy):
    if not rows:
        return empty_summary()

    total = len(rows)
    true_venomous = [row for row in rows if row["true_venom_status"] == "Venomous"]
    true_non_venomous = [
        row for row in rows if row["true_venom_status"] == "Non Venomous"
    ]

    species_hits = sum(row[strategy]["pred_class"] == row["true_class"] for row in rows)
    risk_hits = sum(
        row[strategy]["pred_venom_status"] == row["true_venom_status"] for row in rows
    )
    venomous_hits = sum(
        row[strategy]["pred_venom_status"] == "Venomous" for row in true_venomous
    )
    non_venomous_hits = sum(
        row[strategy]["pred_venom_status"] == "Non Venomous"
        for row in true_non_venomous
    )
    venomous_false_negatives = sum(
        row[strategy]["pred_venom_status"] == "Non Venomous" for row in true_venomous
    )
    non_venomous_false_positives = sum(
        row[strategy]["pred_venom_status"] == "Venomous" for row in true_non_venomous
    )

    changed = sum(
        row[strategy]["pred_class"] != row["image_only"]["pred_class"] for row in rows
    )
    improved = sum(
        row[strategy]["pred_venom_status"] == row["true_venom_status"]
        and row["image_only"]["pred_venom_status"] != row["true_venom_status"]
        for row in rows
    )
    worsened = sum(
        row[strategy]["pred_venom_status"] != row["true_venom_status"]
        and row["image_only"]["pred_venom_status"] == row["true_venom_status"]
        for row in rows
    )

    return {
        "support": total,
        "species_top1_accuracy": safe_divide(species_hits, total),
        "risk_accuracy": safe_divide(risk_hits, total),
        "venomous_risk_recall": safe_divide(venomous_hits, len(true_venomous)),
        "non_venomous_risk_recall": safe_divide(
            non_venomous_hits, len(true_non_venomous)
        ),
        "venomous_false_negatives": venomous_false_negatives,
        "non_venomous_false_positives": non_venomous_false_positives,
        "changed_from_image_only": changed,
        "risk_improved_vs_image_only": improved,
        "risk_worsened_vs_image_only": worsened,
    }


def prediction_payload(index, class_names, metadata_by_class, predictions):
    class_name = class_names[index]
    metadata = metadata_by_class[class_name]
    return {
        "pred_class": class_name,
        "pred_scientific_name": metadata["scientific_name"],
        "pred_venom_status": metadata["venom_status"],
        "confidence": float(predictions[index]),
    }


def rank_strategies(summary_by_strategy):
    def ranking_key(item):
        _, metrics = item
        return (
            metrics["venomous_risk_recall"],
            metrics["risk_accuracy"],
            metrics["non_venomous_risk_recall"],
        )

    return [
        {
            "strategy": name,
            "venomous_risk_recall": metrics["venomous_risk_recall"],
            "risk_accuracy": metrics["risk_accuracy"],
            "non_venomous_risk_recall": metrics["non_venomous_risk_recall"],
            "venomous_false_negatives": metrics["venomous_false_negatives"],
            "non_venomous_false_positives": metrics["non_venomous_false_positives"],
        }
        for name, metrics in sorted(
            summary_by_strategy.items(), key=ranking_key, reverse=True
        )
    ]


def main():
    args = parse_args()
    model = keras.models.load_model(args.model)
    with open(args.classes, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    metadata_by_class = load_metadata(args.metadata)
    manifest_rows, manifest_by_path = load_manifest(args.manifest)
    priors = build_geo_priors(manifest_rows, class_names)
    soft_boosts = parse_float_list(args.soft_boosts)
    bayes_alphas = parse_float_list(args.bayes_alphas)

    rows = []
    strategies = ["image_only", f"top{args.top_k}_first_region_match", "hard_filter"]
    strategies += [f"soft_boost_{boost:g}" for boost in soft_boosts]
    strategies += [f"bayes_prior_alpha_{alpha:g}" for alpha in bayes_alphas]

    for image_path, true_class in iter_images(Path(args.data_dir) / "test"):
        predictions = model.predict(load_image(image_path), verbose=0)[0]
        true_metadata = metadata_by_class[true_class]
        manifest_row = manifest_by_path.get(str(image_path).lower(), {})
        region = {
            "country": manifest_row.get("country", ""),
            "continent": manifest_row.get("continent", ""),
        }
        has_known_country = normalize(region.get("country")) not in {"", "unknown"}

        choices = {
            "image_only": int(np.argmax(predictions)),
            f"top{args.top_k}_first_region_match": choose_top_region_match(
                predictions, class_names, metadata_by_class, region, args.top_k
            ),
            "hard_filter": choose_hard_filter(
                predictions, class_names, metadata_by_class, region
            ),
        }
        for boost in soft_boosts:
            choices[f"soft_boost_{boost:g}"] = choose_soft_boost(
                predictions, class_names, metadata_by_class, region, boost
            )
        for alpha in bayes_alphas:
            choices[f"bayes_prior_alpha_{alpha:g}"] = choose_bayes_prior(
                predictions, class_names, region, priors, alpha
            )

        row = {
            "image_path": str(image_path),
            "true_class": true_class,
            "true_scientific_name": true_metadata["scientific_name"],
            "true_venom_status": true_metadata["venom_status"],
            "country": region["country"],
            "continent": region["continent"],
            "has_known_country": has_known_country,
        }
        for strategy, index in choices.items():
            row[strategy] = prediction_payload(
                index, class_names, metadata_by_class, predictions
            )
        rows.append(row)

    known_country_rows = [row for row in rows if row["has_known_country"]]
    summary_all = {strategy: summarize(rows, strategy) for strategy in strategies}
    summary_known_country = {
        strategy: summarize(known_country_rows, strategy) for strategy in strategies
    }

    payload = {
        "model": args.model,
        "classes": len(class_names),
        "support": len(rows),
        "known_country_support": len(known_country_rows),
        "strategies": strategies,
        "all_rows": summary_all,
        "known_country_rows": summary_known_country,
        "ranking_known_country_by_safety": rank_strategies(summary_known_country),
        "notes": [
            "image_only usa solo probabilidad visual.",
            "topK_first_region_match es el flujo actual de la app: solo reordena dentro del top-k.",
            "hard_filter elimina especies no compatibles con pais/continente cuando hay metadata.",
            "soft_boost aumenta la puntuacion visual de especies compatibles con la region.",
            "bayes_prior combina probabilidad visual con frecuencia geografica estimada desde train/val.",
            "La region debe usarse como apoyo, no como veredicto independiente.",
        ],
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=True)

    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
