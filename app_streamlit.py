import csv
import hashlib
import html
import io
import json
import math
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras


IMG_SIZE = (224, 224)
EPSILON = 1e-9
GEO_PRIOR_ALPHA = 0.75
GEO_PRIOR_TOP_N = 10
SPECIES_CONFIDENCE_FLOOR = 0.35
NON_VENOMOUS_SAFE_CONFIDENCE = 0.50
NON_VENOMOUS_TOP5_MIN = 4
NON_VENOMOUS_BINARY_ALERT_MAX = 0.45
NON_VENOMOUS_TOP5_VENOMOUS_ALERT = 0.07
APP_LOG_PATH = Path("reports/app_predictions/prediction_log.csv")
PREDICTION_LOG_FIELDS = [
    "timestamp_utc",
    "image_name",
    "image_sha256",
    "location_mode",
    "selected_country",
    "location_label",
    "species_profile",
    "binary_profile",
    "use_species_model",
    "venomous_threshold",
    "binary_confidence_threshold",
    "species_prediction",
    "species_scientific_name",
    "species_risk",
    "species_confidence",
    "direct_species",
    "direct_scientific_name",
    "direct_risk",
    "direct_confidence",
    "region_used",
    "region_method",
    "binary_decision",
    "binary_p_venomous",
    "binary_argmax_class",
    "binary_argmax_confidence",
    "integrated_decision",
    "integrated_reason",
]

SPECIES_PROFILES = {
    "51 especies - B4 EfficientNetV2B1": {
        "model": Path("models/experiments/species_pilot51_b4_efficientnetv2b1_colab/best_model.keras"),
        "classes": Path("models/experiments/species_pilot51_b4_efficientnetv2b1_colab/class_names.json"),
        "metadata": Path("data/metadata/species_pilot51_b4_scale_metadata.csv"),
        "manifest": Path("data/metadata/species_pilot51_b4_scale_manifest.csv"),
        "species_count": 51,
        "image_size": (260, 260),
        "description": "Modelo principal actual; ampliacion general de train con holdout externo reservado.",
    },
    "51 especies - B5 EfficientNetV2B3 300": {
        "model": Path("models/experiments/species_pilot51_b5_efficientnetv2b3_300_colab/best_model.keras"),
        "classes": Path("models/experiments/species_pilot51_b5_efficientnetv2b3_300_colab/class_names.json"),
        "metadata": Path("data/metadata/species_pilot51_b4_scale_metadata.csv"),
        "manifest": Path("data/metadata/species_pilot51_b4_scale_manifest.csv"),
        "species_count": 51,
        "image_size": (300, 300),
        "description": "Candidato B5; EfficientNetV2B3 a 300x300 entrenado sobre B4 Scale.",
    },
}

MODEL_PROFILES = {
    "Preventivo dirigido": {
        "model": Path("models/experiments/outputs_targeted_efficientnetb0_lowlr/best_model.keras"),
        "classes": Path("models/experiments/outputs_targeted_efficientnetb0_lowlr/class_names.json"),
        "venomous_threshold": 0.27,
        "description": "Reduce falsos negativos de Venomous; acepta mas falsos positivos.",
    },
}


@st.cache_resource(show_spinner=False)
def load_model(model_path):
    return keras.models.load_model(model_path)


@st.cache_data(show_spinner=False)
def load_classes(classes_path):
    with open(classes_path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def load_species_metadata(metadata_path):
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return {row["class_folder"]: row for row in rows}


@st.cache_data(show_spinner=False)
def load_species_manifest(manifest_path):
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return []
    with open(manifest_path, "r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


@st.cache_data(show_spinner=False)
def load_available_countries(manifest_path):
    rows = load_species_manifest(manifest_path)
    countries = {
        row.get("country", "").strip()
        for row in rows
        if row.get("country", "").strip()
        and normalize_region_text(row.get("country", "")) != "unknown"
    }
    return sorted(countries)


@st.cache_data(show_spinner=False)
def build_geo_priors(manifest_path, class_names):
    rows = load_species_manifest(manifest_path)
    train_rows = [row for row in rows if row.get("split") in {"train", "val"}]
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


def apply_theme():
    st.markdown(
        """
        <style>
        .stApp,
        [data-testid="stAppViewContainer"] {
            background: #f8fafc;
            color: #0f172a;
        }
        [data-testid="stHeader"] {
            background: rgba(248, 250, 252, 0);
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 2.5rem;
        }
        [data-testid="stSidebar"] {
            background: #eef2f7;
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #0f172a !important;
        }
        [data-testid="stRadio"] p,
        [data-testid="stRadio"] span {
            color: #0f172a !important;
        }
        [data-testid="stFileUploader"] section {
            background: #ffffff !important;
            border: 1px solid #e5e7eb !important;
            border-radius: 8px !important;
        }
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] section p,
        [data-testid="stFileUploader"] section span {
            color: #334155 !important;
        }
        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploader"] button * {
            background: #0f172a !important;
            color: #f8fafc !important;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="select"] span {
            color: #f8fafc !important;
        }
        .app-kicker {
            color: #64748b;
            font-size: 0.86rem;
            margin-bottom: 0.2rem;
        }
        .app-title {
            color: #0f172a;
            font-size: 2.25rem;
            font-weight: 750;
            letter-spacing: 0;
            line-height: 1.08;
            margin: 0;
        }
        .app-subtitle {
            color: #475569;
            font-size: 0.98rem;
            margin-top: 0.55rem;
            margin-bottom: 1.55rem;
        }
        .section-label {
            color: #334155;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0;
            margin: 0.2rem 0 0.35rem 0;
        }
        .result-card {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            background: #ffffff;
            padding: 1.05rem 1.1rem;
            margin: 0.35rem 0 0.9rem 0;
        }
        .result-card h3 {
            color: #0f172a;
            font-size: 1.35rem;
            line-height: 1.15;
            margin: 0 0 0.35rem 0;
        }
        .muted {
            color: #64748b;
            font-size: 0.9rem;
            margin: 0;
        }
        .pill {
            display: inline-block;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            padding: 0.2rem 0.55rem;
            margin-bottom: 0.65rem;
        }
        .pill-red {
            background: #fee2e2;
            color: #991b1b;
        }
        .pill-green {
            background: #dcfce7;
            color: #166534;
        }
        .pill-amber {
            background: #fef3c7;
            color: #92400e;
        }
        .species-name {
            color: #0f172a;
            display: block;
            font-size: 1.55rem;
            font-weight: 750;
            line-height: 1.18;
            margin: 0.25rem 0;
        }
        .scientific-name {
            color: #64748b;
            display: block;
            font-size: 0.98rem;
            font-style: italic;
            margin-bottom: 0.95rem;
        }
        .status-label {
            border-top: 1px solid #e5e7eb;
            color: #334155;
            display: block;
            font-size: 0.88rem;
            font-weight: 700;
            padding-top: 0.85rem;
        }
        .candidate-note {
            color: #64748b;
            display: block;
            font-size: 0.86rem;
            line-height: 1.35;
            margin-top: 0.65rem;
        }
        .placeholder {
            border: 1px dashed #cbd5e1;
            border-radius: 8px;
            color: #64748b;
            padding: 1.2rem;
            background: #f8fafc;
        }
        div.stButton > button {
            border-radius: 8px;
            font-weight: 700;
            min-height: 2.8rem;
        }
        div[data-testid="stMetricValue"] {
            color: #0f172a;
            font-size: 1.35rem;
        }
        @media (max-width: 760px) {
            .app-title {
                font-size: 1.8rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def prepare_image(image, image_size=IMG_SIZE):
    image = image.convert("RGB").resize(image_size)
    array = keras.utils.img_to_array(image)
    return np.expand_dims(array, axis=0)


def normalize_class_name(class_name):
    return class_name.lower().replace("_", " ").strip()


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
    scores = []
    candidate_indices = np.argsort(predictions)[::-1][:top_n]
    for index in candidate_indices:
        index = int(index)
        class_name = class_names[index]
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


def predict_species(image, region_query, species_profile):
    species_model_path = species_profile["model"]
    species_classes_path = species_profile["classes"]
    species_metadata_path = species_profile["metadata"]
    species_manifest_path = species_profile["manifest"]
    species_image_size = species_profile.get("image_size", IMG_SIZE)

    species_model = load_model(species_model_path)
    species_classes = load_classes(species_classes_path)
    metadata_by_class = load_species_metadata(species_metadata_path)
    geo_priors = build_geo_priors(species_manifest_path, tuple(species_classes))
    predictions = species_model.predict(prepare_image(image, species_image_size), verbose=0)[0]

    selected_index, top_indices, region_used, region_method = select_species_with_region(
        predictions,
        species_classes,
        metadata_by_class,
        region_query,
        geo_priors,
        top_k=5,
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
                "Especie": species_label(row),
                "Nombre cientifico": row.get("scientific_name", "unknown"),
                "Riesgo": row.get("venom_status", "unknown"),
                "Confianza": float(predictions[index]),
                "Compatible con region": bool(region_query and is_region_compatible(row, region_query)),
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
        "species_count": species_profile["species_count"],
    }


def predict_binary(image, model_path, classes_path, venomous_threshold):
    model = load_model(model_path)
    class_names = load_classes(classes_path)
    predictions = model.predict(prepare_image(image), verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    argmax_class = class_names[predicted_index]
    argmax_confidence = float(predictions[predicted_index])
    normalized_names = [normalize_class_name(name) for name in class_names]
    venomous_index = normalized_names.index("venomous")
    venomous_probability = float(predictions[venomous_index])
    fallback_class = negative_class_name(class_names, "Venomous")
    decision = "Venomous" if venomous_probability >= venomous_threshold else fallback_class

    probabilities = [
        {"Clase": class_name, "Probabilidad": float(probability)}
        for class_name, probability in zip(class_names, predictions)
    ]
    return {
        "decision": decision,
        "argmax_class": argmax_class,
        "argmax_confidence": argmax_confidence,
        "venomous_probability": venomous_probability,
        "probabilities": probabilities,
    }


def top_candidate_risk_counts(species_result, limit=5):
    counts = Counter()
    for candidate in species_result.get("top_candidates", [])[:limit]:
        counts[candidate.get("Riesgo", "unknown")] += 1
    return counts


def top_candidate_max_confidence_for_risk(species_result, risk, limit=5):
    confidence = 0.0
    for candidate in species_result.get("top_candidates", [])[:limit]:
        if candidate.get("Riesgo") == risk:
            confidence = max(confidence, float(candidate.get("Confianza", 0.0)))
    return confidence


def integrated_decision(species_result, binary_result, binary_confidence_threshold):
    if species_result:
        species_risk = species_result["selected_risk"]
        species_confidence = species_result["selected_confidence"]
        top5_risk_counts = top_candidate_risk_counts(species_result)
        top5_venomous_confidence = top_candidate_max_confidence_for_risk(
            species_result, "Venomous"
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
                    "reason": (
                        "La especie no venenosa tiene alta confianza y la alerta "
                        "preventiva no es dominante."
                    ),
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
                "reason": "Hay una candidata venenosa relevante dentro del top 5.",
            }
        return {
            "decision": "Non Venomous",
            "title": "Riesgo venenoso bajo",
            "tone": "green",
            "reason": "La especie y el respaldo preventivo coinciden en bajo riesgo.",
        }

    if binary_result["decision"] == "Venomous":
        return {
            "decision": "Venomous",
            "title": "Posible serpiente venenosa",
            "tone": "red",
            "reason": "El modelo preventivo supera el umbral de riesgo venenoso.",
        }
    if binary_result["argmax_confidence"] < binary_confidence_threshold:
        return {
            "decision": "No concluyente",
            "title": "No concluyente",
            "tone": "amber",
            "reason": "La confianza del modelo binario no alcanza el umbral minimo.",
        }
    return {
        "decision": "Non Venomous",
        "title": "Riesgo venenoso bajo",
        "tone": "green",
        "reason": "El modelo preventivo no supera el umbral de riesgo venenoso.",
    }


def safe_text(value):
    return html.escape(str(value or ""))


def safe_float(value):
    if value is None:
        return ""
    return f"{float(value):.8f}"


def display_risk(value):
    return {
        "Venomous": "Venenosa",
        "Non Venomous": "No venenosa",
        "unknown": "Desconocido",
    }.get(str(value or ""), str(value or ""))


def build_prediction_log_row(
    image_name,
    image_sha256,
    location_mode,
    selected_country,
    location_label,
    species_profile_name,
    binary_profile_name,
    use_species_model,
    venomous_threshold,
    confidence_threshold,
    species_result,
    binary_result,
    decision_result,
):
    species_result = species_result or {}
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "image_name": image_name,
        "image_sha256": image_sha256,
        "location_mode": location_mode,
        "selected_country": selected_country,
        "location_label": location_label,
        "species_profile": species_profile_name,
        "binary_profile": binary_profile_name,
        "use_species_model": str(bool(use_species_model)),
        "venomous_threshold": safe_float(venomous_threshold),
        "binary_confidence_threshold": safe_float(confidence_threshold),
        "species_prediction": species_result.get("selected_display_name", ""),
        "species_scientific_name": species_result.get("selected_scientific_name", ""),
        "species_risk": species_result.get("selected_risk", ""),
        "species_confidence": safe_float(species_result.get("selected_confidence")),
        "direct_species": species_result.get("direct_display_name", ""),
        "direct_scientific_name": species_result.get("direct_scientific_name", ""),
        "direct_risk": species_result.get("direct_risk", ""),
        "direct_confidence": safe_float(species_result.get("direct_confidence")),
        "region_used": str(bool(species_result.get("region_used", False))),
        "region_method": species_result.get("region_method", ""),
        "binary_decision": binary_result["decision"],
        "binary_p_venomous": safe_float(binary_result["venomous_probability"]),
        "binary_argmax_class": binary_result["argmax_class"],
        "binary_argmax_confidence": safe_float(binary_result["argmax_confidence"]),
        "integrated_decision": decision_result["decision"],
        "integrated_reason": decision_result["reason"],
    }


def append_prediction_log(row, log_path=APP_LOG_PATH):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with open(log_path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PREDICTION_LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in PREDICTION_LOG_FIELDS})


def render_decision_card(decision_result, species_result, binary_result, location_label):
    del binary_result, location_label

    pill_class = {
        "red": "pill-red",
        "green": "pill-green",
        "amber": "pill-amber",
    }.get(decision_result["tone"], "pill-amber")

    status_label = {
        "Venomous": "Venenosa",
        "Non Venomous": "No venenosa",
        "No concluyente": "No concluyente",
    }.get(decision_result["decision"], decision_result["decision"])

    if species_result:
        species_name = species_result["selected_display_name"]
        scientific_name = species_result["selected_scientific_name"]
    else:
        species_name = "No disponible"
        scientific_name = "Modelo por especie desactivado"

    if decision_result["decision"] == "No concluyente":
        species_heading = "Especie candidata"
        note = (
            "La imagen no permite una identificacion confiable. "
            "La especie mostrada es solo una candidata del modelo."
        )
    else:
        species_heading = "Especie"
        note = ""

    html_block = f"""
    <div class="result-card">
        <span class="pill {pill_class}">{safe_text(status_label)}</span>
        <span class="status-label">{safe_text(species_heading)}</span>
        <span class="species-name">{safe_text(species_name)}</span>
        <span class="scientific-name">{safe_text(scientific_name)}</span>
        <span class="status-label">Clasificacion: {safe_text(status_label)}</span>
        <span class="candidate-note">{safe_text(note)}</span>
    </div>
    """
    st.markdown(html_block, unsafe_allow_html=True)

    if species_result and species_result.get("top_candidates"):
        rows = []
        for rank, candidate in enumerate(species_result["top_candidates"][:5], start=1):
            rows.append(
                {
                    "#": rank,
                    "Especie": candidate["Especie"],
                    "Nombre cientifico": candidate["Nombre cientifico"],
                    "Riesgo": display_risk(candidate["Riesgo"]),
                    "Confianza": f"{candidate['Confianza']:.2%}",
                }
            )
        st.markdown('<p class="section-label">Top 5 especies candidatas</p>', unsafe_allow_html=True)
        st.dataframe(rows, hide_index=True, use_container_width=True)

    st.caption(
        "No manipules la serpiente. Si existe riesgo de mordedura, busca ayuda profesional."
    )


def missing_paths(paths):
    return [str(path) for path in paths if not Path(path).exists()]


def main():
    st.set_page_config(page_title="Identificador de serpientes", layout="wide")
    apply_theme()

    st.markdown('<p class="app-kicker">Clasificacion visual con contexto geografico opcional</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="app-title">Identificador de serpientes</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-subtitle">Carga una imagen, indica si conoces el pais y ejecuta la busqueda.</p>',
        unsafe_allow_html=True,
    )

    species_profile_name = "51 especies - B4 EfficientNetV2B1"
    species_profile = SPECIES_PROFILES[species_profile_name]
    profile_name = "Preventivo dirigido"
    profile = MODEL_PROFILES[profile_name]
    use_species_model = True
    model_path = profile["model"]
    classes_path = profile["classes"]
    venomous_threshold = profile["venomous_threshold"]
    confidence_threshold = 0.90
    country_options = ["Selecciona un pais"] + load_available_countries(species_profile["manifest"])

    input_col, result_col = st.columns([0.9, 1.1], gap="large")

    with input_col:
        st.markdown('<p class="section-label">Imagen</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Imagen de serpiente",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        image = None
        image_sha256 = ""
        if uploaded_file:
            image_bytes = uploaded_file.getvalue()
            image_sha256 = hashlib.sha256(image_bytes).hexdigest()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            st.image(image, caption=uploaded_file.name, use_container_width=True)

        st.markdown('<p class="section-label">Ubicacion</p>', unsafe_allow_html=True)
        location_mode = st.radio(
            "Contexto geografico",
            ["Selecciona una opcion", "Sin region", "Con pais"],
            horizontal=True,
            label_visibility="collapsed",
        )
        selected_country = ""
        if location_mode == "Con pais":
            selected_country = st.selectbox("Pais", country_options)

        region_query = selected_country if location_mode == "Con pais" else ""
        location_ready = location_mode == "Sin region" or (
            location_mode == "Con pais" and selected_country != "Selecciona un pais"
        )
        location_label = "Sin region" if location_mode == "Sin region" else selected_country

        required_paths = [model_path, classes_path]
        if use_species_model:
            required_paths.extend(
                [
                    species_profile["model"],
                    species_profile["classes"],
                    species_profile["metadata"],
                    species_profile["manifest"],
                ]
            )
        missing = missing_paths(required_paths)
        ready = uploaded_file is not None and location_ready and not missing

        signature = "|".join(
            [
                uploaded_file.name if uploaded_file else "",
                location_mode,
                selected_country,
                species_profile_name,
                str(model_path),
                str(classes_path),
                str(venomous_threshold),
                str(confidence_threshold),
                str(use_species_model),
            ]
        )

        search_clicked = st.button(
            "Buscar",
            type="primary",
            disabled=not ready,
            use_container_width=True,
        )

        if missing:
            st.error("Faltan archivos de modelo o metadata en la configuracion actual.")
            with st.expander("Ver rutas faltantes"):
                for path in missing:
                    st.code(path)
        elif uploaded_file is None:
            st.caption("Carga una imagen para activar la busqueda.")
        elif not location_ready:
            st.caption("Selecciona `Sin region` o escoge un pais disponible.")

    with result_col:
        if "prediction_signature" not in st.session_state:
            st.session_state.prediction_signature = ""
            st.session_state.prediction_payload = None

        if search_clicked and image is not None:
            with st.spinner("Buscando coincidencias..."):
                species_result = (
                    predict_species(image, region_query, species_profile)
                    if use_species_model
                    else None
                )
                binary_result = predict_binary(
                    image,
                    model_path,
                    classes_path,
                    venomous_threshold,
                )
                decision_result = integrated_decision(
                    species_result,
                    binary_result,
                    confidence_threshold,
                )
                append_prediction_log(
                    build_prediction_log_row(
                        uploaded_file.name,
                        image_sha256,
                        location_mode,
                        selected_country,
                        location_label,
                        species_profile_name,
                        profile_name,
                        use_species_model,
                        venomous_threshold,
                        confidence_threshold,
                        species_result,
                        binary_result,
                        decision_result,
                    )
                )
                st.session_state.prediction_signature = signature
                st.session_state.prediction_payload = {
                    "species": species_result,
                    "binary": binary_result,
                    "decision": decision_result,
                    "location_label": location_label,
                }

        payload = st.session_state.prediction_payload
        if payload and st.session_state.prediction_signature == signature:
            render_decision_card(
                payload["decision"],
                payload["species"],
                payload["binary"],
                payload["location_label"],
            )
        else:
            st.markdown(
                """
                <div class="placeholder">
                    El resultado aparecera aqui despues de cargar la imagen, definir la ubicacion y pulsar Buscar.
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
