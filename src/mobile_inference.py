import csv
import io
import json
import math
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMG_SIZE = (224, 224)
EPSILON = 1e-9
GEO_PRIOR_ALPHA = 0.75
GEO_PRIOR_TOP_N = 10
SPECIES_CONFIDENCE_FLOOR = 0.35
NON_VENOMOUS_SAFE_CONFIDENCE = 0.50
NON_VENOMOUS_TOP5_MIN = 4
NON_VENOMOUS_BINARY_ALERT_MAX = 0.45
NON_VENOMOUS_TOP5_VENOMOUS_ALERT = 0.07


SPECIES_PROFILE = {
    "model": PROJECT_ROOT
    / "models/experiments/species_pilot51_b4_efficientnetv2b1_colab/best_model.keras",
    "classes": PROJECT_ROOT
    / "models/experiments/species_pilot51_b4_efficientnetv2b1_colab/class_names.json",
    "metadata": PROJECT_ROOT / "data/metadata/species_pilot51_b4_scale_metadata.csv",
    "manifest": PROJECT_ROOT / "data/metadata/species_pilot51_b4_scale_manifest.csv",
    "species_count": 51,
    "image_size": (260, 260),
    "name": "51 especies - B4 EfficientNetV2B1",
}

BINARY_PROFILE = {
    "model": PROJECT_ROOT
    / "models/experiments/outputs_targeted_efficientnetb0_lowlr/best_model.keras",
    "classes": PROJECT_ROOT
    / "models/experiments/outputs_targeted_efficientnetb0_lowlr/class_names.json",
    "venomous_threshold": 0.27,
    "confidence_threshold": 0.90,
}


def ensure_assets_exist():
    paths = [
        SPECIES_PROFILE["model"],
        SPECIES_PROFILE["classes"],
        SPECIES_PROFILE["metadata"],
        SPECIES_PROFILE["manifest"],
        BINARY_PROFILE["model"],
        BINARY_PROFILE["classes"],
    ]
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Faltan archivos para inferencia movil: " + ", ".join(missing))


@lru_cache(maxsize=4)
def load_model(model_path):
    return keras.models.load_model(model_path)


@lru_cache(maxsize=8)
def load_classes(classes_path):
    with open(classes_path, "r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=8)
def load_species_metadata(metadata_path):
    metadata_path = Path(metadata_path)
    with open(metadata_path, "r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return {row["class_folder"]: row for row in rows}


@lru_cache(maxsize=8)
def load_species_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


@lru_cache(maxsize=8)
def build_geo_priors(manifest_path, class_names_tuple):
    rows = load_species_manifest(manifest_path)
    train_rows = [row for row in rows if row.get("split") in {"train", "val"}]
    class_names = list(class_names_tuple)
    class_counts = Counter(row["class_folder"] for row in train_rows)
    total_class_count = sum(class_counts.values())
    class_count = len(class_names)
    class_prior = {
        class_name: (class_counts[class_name] + 1) / (total_class_count + class_count)
        for class_name in class_names
    }

    country_counts = defaultdict(Counter)
    continent_counts = defaultdict(Counter)
    for row in train_rows:
        class_name = row["class_folder"]
        country = normalize_region_text(row.get("country", ""))
        continent = normalize_region_text(row.get("continent", ""))
        if country and country != "unknown":
            country_counts[country][class_name] += 1
        if continent and continent != "unknown":
            continent_counts[continent][class_name] += 1

    return {
        "class_prior": dict(class_prior),
        "country_counts": {key: dict(value) for key, value in country_counts.items()},
        "continent_counts": {key: dict(value) for key, value in continent_counts.items()},
    }


def image_from_bytes(image_bytes):
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def prepare_image(image, image_size=IMG_SIZE):
    image = image.convert("RGB").resize(image_size)
    array = keras.utils.img_to_array(image)
    return np.expand_dims(array, axis=0)


def normalize_class_name(class_name):
    return str(class_name or "").lower().replace("_", " ").strip()


def normalize_region_text(value):
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = "".join(character for character in value if not unicodedata.combining(character))
    return normalize_class_name(value)


def negative_class_name(class_names, positive_class):
    positive = normalize_class_name(positive_class)
    for class_name in class_names:
        if normalize_class_name(class_name) != positive:
            return class_name
    return "Non Venomous"


def is_region_compatible(metadata_row, region_query):
    query = normalize_region_text(region_query)
    if not query:
        return False
    countries = normalize_region_text(metadata_row.get("countries", ""))
    continents = normalize_region_text(metadata_row.get("continents", ""))
    return query in countries or query in continents


def find_geo_counts(region_query, geo_priors):
    query = normalize_region_text(region_query)
    if not query:
        return None, ""
    country_counts = geo_priors["country_counts"].get(query)
    if country_counts:
        return country_counts, "pais"
    continent_counts = geo_priors["continent_counts"].get(query)
    if continent_counts:
        return continent_counts, "continente"
    return None, ""


def select_species_by_geo_prior(
    predictions,
    class_names,
    region_query,
    geo_priors,
    alpha=GEO_PRIOR_ALPHA,
    top_n=GEO_PRIOR_TOP_N,
):
    geo_counts, geo_level = find_geo_counts(region_query, geo_priors)
    if not geo_counts:
        return None, ""

    total_geo = sum(geo_counts.values())
    class_count = len(class_names)
    candidate_indices = np.argsort(predictions)[::-1][:top_n]
    scores = []
    for index in candidate_indices:
        class_name = class_names[int(index)]
        location_prior = (geo_counts.get(class_name, 0) + 1) / (total_geo + class_count)
        class_prior = max(float(geo_priors["class_prior"].get(class_name, EPSILON)), EPSILON)
        geo_ratio = max(location_prior / class_prior, EPSILON)
        visual_probability = max(float(predictions[index]), EPSILON)
        scores.append(math.log(visual_probability) + alpha * math.log(geo_ratio))

    return int(candidate_indices[int(np.argmax(scores))]), geo_level


def select_species_with_region(
    predictions,
    class_names,
    metadata_by_class,
    region_query,
    geo_priors,
    top_k=5,
):
    sorted_indices = np.argsort(predictions)[::-1]
    top_indices = [int(index) for index in sorted_indices[:top_k]]
    selected_index = top_indices[0]
    region_used = False
    region_method = "solo imagen"

    if region_query:
        geo_index, geo_level = select_species_by_geo_prior(
            predictions,
            class_names,
            region_query,
            geo_priors,
            alpha=GEO_PRIOR_ALPHA,
        )
        if geo_index is not None:
            return (
                geo_index,
                top_indices,
                geo_index != selected_index,
                f"prior geografico suave por {geo_level}",
            )

        for index in top_indices:
            class_name = class_names[index]
            metadata_row = metadata_by_class.get(class_name, {})
            if is_region_compatible(metadata_row, region_query):
                selected_index = index
                region_used = index != top_indices[0]
                region_method = "compatibilidad en top-5"
                break

    return selected_index, top_indices, region_used, region_method


def species_label(metadata_row):
    common_name = metadata_row.get("common_name", "unknown")
    if common_name and common_name != "unknown":
        return common_name
    return metadata_row.get("scientific_name", "unknown")


def predict_species(image, region_query):
    species_model = load_model(str(SPECIES_PROFILE["model"]))
    species_classes = load_classes(str(SPECIES_PROFILE["classes"]))
    metadata_by_class = load_species_metadata(str(SPECIES_PROFILE["metadata"]))
    geo_priors = build_geo_priors(str(SPECIES_PROFILE["manifest"]), tuple(species_classes))
    predictions = species_model.predict(
        prepare_image(image, SPECIES_PROFILE["image_size"]),
        verbose=0,
    )[0]

    selected_index, top_indices, region_used, region_method = select_species_with_region(
        predictions,
        species_classes,
        metadata_by_class,
        region_query,
        geo_priors,
    )

    direct_index = top_indices[0]
    selected_class = species_classes[selected_index]
    direct_class = species_classes[direct_index]
    selected_metadata = metadata_by_class[selected_class]
    direct_metadata = metadata_by_class[direct_class]

    top_candidates = []
    display_indices = list(top_indices)
    if selected_index not in display_indices:
        display_indices.insert(0, selected_index)
    for index in display_indices:
        class_name = species_classes[index]
        row = metadata_by_class[class_name]
        top_candidates.append(
            {
                "species": species_label(row),
                "scientific_name": row.get("scientific_name", "unknown"),
                "risk": row.get("venom_status", "unknown"),
                "confidence": float(predictions[index]),
                "region_compatible": bool(region_query and is_region_compatible(row, region_query)),
            }
        )

    return {
        "selected_display_name": species_label(selected_metadata),
        "selected_scientific_name": selected_metadata.get("scientific_name", "unknown"),
        "selected_risk": selected_metadata.get("venom_status", "unknown"),
        "selected_confidence": float(predictions[selected_index]),
        "direct_display_name": species_label(direct_metadata),
        "direct_scientific_name": direct_metadata.get("scientific_name", "unknown"),
        "direct_risk": direct_metadata.get("venom_status", "unknown"),
        "direct_confidence": float(predictions[direct_index]),
        "region_used": region_used,
        "region_method": region_method,
        "top_candidates": top_candidates,
        "species_count": SPECIES_PROFILE["species_count"],
    }


def predict_binary(image):
    model = load_model(str(BINARY_PROFILE["model"]))
    class_names = load_classes(str(BINARY_PROFILE["classes"]))
    predictions = model.predict(prepare_image(image), verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    argmax_class = class_names[predicted_index]
    argmax_confidence = float(predictions[predicted_index])
    normalized_names = [normalize_class_name(name) for name in class_names]
    venomous_index = normalized_names.index("venomous")
    venomous_probability = float(predictions[venomous_index])
    fallback_class = negative_class_name(class_names, "Venomous")
    decision = (
        "Venomous"
        if venomous_probability >= BINARY_PROFILE["venomous_threshold"]
        else fallback_class
    )

    return {
        "decision": decision,
        "argmax_class": argmax_class,
        "argmax_confidence": argmax_confidence,
        "venomous_probability": venomous_probability,
        "probabilities": [
            {"class_name": class_name, "probability": float(probability)}
            for class_name, probability in zip(class_names, predictions)
        ],
    }


def top_candidate_risk_counts(species_result, limit=5):
    counts = Counter()
    for candidate in species_result.get("top_candidates", [])[:limit]:
        counts[candidate.get("risk", "unknown")] += 1
    return counts


def top_candidate_max_confidence_for_risk(species_result, risk, limit=5):
    confidence = 0.0
    for candidate in species_result.get("top_candidates", [])[:limit]:
        if candidate.get("risk") == risk:
            confidence = max(confidence, float(candidate.get("confidence", 0.0)))
    return confidence


def integrated_decision(species_result, binary_result):
    species_risk = species_result["selected_risk"]
    species_confidence = species_result["selected_confidence"]
    top5_risk_counts = top_candidate_risk_counts(species_result)
    top5_venomous_confidence = top_candidate_max_confidence_for_risk(
        species_result,
        "Venomous",
    )

    if species_risk == "Venomous":
        return {
            "decision": "Venomous",
            "title": "Serpiente venenosa",
            "tone": "red",
            "reason": "La prediccion por especie indica riesgo venenoso.",
        }

    if binary_result["decision"] == "Venomous":
        if (
            species_risk == "Non Venomous"
            and species_confidence >= NON_VENOMOUS_SAFE_CONFIDENCE
            and top5_venomous_confidence < NON_VENOMOUS_TOP5_VENOMOUS_ALERT
            and (
                top5_risk_counts["Non Venomous"] >= NON_VENOMOUS_TOP5_MIN
                or binary_result["venomous_probability"] < NON_VENOMOUS_BINARY_ALERT_MAX
            )
        ):
            return {
                "decision": "Non Venomous",
                "title": "Riesgo venenoso bajo",
                "tone": "green",
                "reason": "La especie no venenosa tiene alta confianza y la alerta preventiva no domina.",
            }
        return {
            "decision": "No concluyente",
            "title": "No concluyente",
            "tone": "amber",
            "reason": "La especie sugiere bajo riesgo, pero el respaldo preventivo detecta alerta.",
        }

    if species_confidence < SPECIES_CONFIDENCE_FLOOR:
        return {
            "decision": "No concluyente",
            "title": "No concluyente",
            "tone": "amber",
            "reason": "La confianza de especie esta por debajo del minimo configurado.",
        }

    if top5_venomous_confidence >= NON_VENOMOUS_TOP5_VENOMOUS_ALERT:
        return {
            "decision": "No concluyente",
            "title": "No concluyente",
            "tone": "amber",
            "reason": "Hay una especie venenosa con peso suficiente dentro del top-5.",
        }

    return {
        "decision": "Non Venomous",
        "title": "Riesgo venenoso bajo",
        "tone": "green",
        "reason": "La especie y el respaldo preventivo coinciden en bajo riesgo.",
    }


def display_risk(value):
    return {
        "Venomous": "Venenosa",
        "Non Venomous": "No venenosa",
        "unknown": "Desconocido",
    }.get(str(value or ""), str(value or ""))


def predict_image_bytes(image_bytes, country=""):
    ensure_assets_exist()
    image = image_from_bytes(image_bytes)
    species_result = predict_species(image, country)
    binary_result = predict_binary(image)
    decision_result = integrated_decision(species_result, binary_result)

    return {
        "decision": {
            **decision_result,
            "display": {
                "decision": {
                    "Venomous": "Venenosa",
                    "Non Venomous": "No venenosa",
                    "No concluyente": "No concluyente",
                }.get(decision_result["decision"], decision_result["decision"]),
                "risk": display_risk(species_result["selected_risk"]),
            },
        },
        "species": species_result,
        "binary": binary_result,
        "model": {
            "species_profile": SPECIES_PROFILE["name"],
            "decision_policy": "B4 safety-first",
        },
        "safety_note": (
            "No manipules la serpiente. Si existe riesgo de mordedura, "
            "busca ayuda profesional."
        ),
    }
