import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app_streamlit as app  # noqa: E402


DEFAULT_EXTERNAL_DIR = Path("data/external_tests/manual_30_species_region")
DEFAULT_OUTPUT_DIR = DEFAULT_EXTERNAL_DIR / "reports"
DEFAULT_SPECIES_PROFILE = "51 especies - B4 EfficientNetV2B1"
DEFAULT_BINARY_PROFILE = "Preventivo dirigido"
DEFAULT_BINARY_CONFIDENCE_THRESHOLD = 0.90


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=True)


def normalize(value):
    return app.normalize_region_text(value)


def result_ok(true_class, predicted_class):
    return "OK" if true_class == predicted_class else "ERROR"


def integrated_ok(true_class, decision):
    if decision == "No concluyente":
        return "NO_CONCLUYENTE"
    return result_ok(true_class, decision)


def choose_wrong_country(correct_country, available_countries):
    correct = normalize(correct_country)
    preferred = [
        "Australia",
        "United States of America",
        "Mexico",
        "India",
        "Brazil",
        "South Africa",
        "Thailand",
        "Colombia",
    ]
    for country in preferred:
        if country in available_countries and normalize(country) != correct:
            return country
    for country in available_countries:
        if normalize(country) != correct:
            return country
    return ""


def class_counts(rows):
    return dict(Counter(row["true_class"] for row in rows))


def summarize_direct_risk(rows, risk_column, ok_column):
    total = len(rows)
    ok = sum(row[ok_column] == "OK" for row in rows)
    venomous_rows = [row for row in rows if row["true_class"] == "Venomous"]
    non_venomous_rows = [row for row in rows if row["true_class"] == "Non Venomous"]
    fn = sum(row[risk_column] == "Non Venomous" for row in venomous_rows)
    fp = sum(row[risk_column] == "Venomous" for row in non_venomous_rows)
    return {
        "support": total,
        "accuracy": ok / total if total else 0,
        "correct": ok,
        "errors": total - ok,
        "venomous_false_negatives": fn,
        "non_venomous_false_positives": fp,
        "venomous_recall": 1 - (fn / len(venomous_rows)) if venomous_rows else 0,
        "non_venomous_recall": 1 - (fp / len(non_venomous_rows))
        if non_venomous_rows
        else 0,
    }


def summarize_integrated(rows):
    total = len(rows)
    resolved = [row for row in rows if row["integrated_decision"] != "No concluyente"]
    no_conclusive = total - len(resolved)
    resolved_correct = sum(row["integrated_ok"] == "OK" for row in resolved)
    true_venomous = [row for row in rows if row["true_class"] == "Venomous"]
    true_non_venomous = [row for row in rows if row["true_class"] == "Non Venomous"]
    dangerous_fn = sum(
        row["true_class"] == "Venomous"
        and row["integrated_decision"] == "Non Venomous"
        for row in rows
    )
    non_venomous_as_venomous = sum(
        row["true_class"] == "Non Venomous"
        and row["integrated_decision"] == "Venomous"
        for row in rows
    )
    venomous_no_conclusive = sum(
        row["true_class"] == "Venomous"
        and row["integrated_decision"] == "No concluyente"
        for row in rows
    )
    non_venomous_no_conclusive = sum(
        row["true_class"] == "Non Venomous"
        and row["integrated_decision"] == "No concluyente"
        for row in rows
    )
    return {
        "support": total,
        "resolved": len(resolved),
        "no_conclusive": no_conclusive,
        "no_conclusive_rate": no_conclusive / total if total else 0,
        "resolved_accuracy": resolved_correct / len(resolved) if resolved else 0,
        "dangerous_false_non_venomous": dangerous_fn,
        "non_venomous_as_venomous": non_venomous_as_venomous,
        "venomous_no_conclusive": venomous_no_conclusive,
        "non_venomous_no_conclusive": non_venomous_no_conclusive,
        "venomous_safety_recall": 1 - (dangerous_fn / len(true_venomous))
        if true_venomous
        else 0,
        "non_venomous_resolved_as_safe_rate": sum(
            row["true_class"] == "Non Venomous"
            and row["integrated_decision"] == "Non Venomous"
            for row in rows
        )
        / len(true_non_venomous)
        if true_non_venomous
        else 0,
    }


def summarize_by_flow(rows):
    by_flow = defaultdict(list)
    for row in rows:
        by_flow[row["flow"]].append(row)

    summary = {}
    for flow, flow_rows in by_flow.items():
        exact_species_ok = sum(row["species_exact_ok"] == "OK" for row in flow_rows)
        summary[flow] = {
            "class_counts": class_counts(flow_rows),
            "species_exact_accuracy": exact_species_ok / len(flow_rows)
            if flow_rows
            else 0,
            "species_risk": summarize_direct_risk(
                flow_rows, "species_risk", "species_risk_ok"
            ),
            "binary_preventive": summarize_direct_risk(
                flow_rows, "binary_decision", "binary_ok"
            ),
            "integrated": summarize_integrated(flow_rows),
        }
    return summary


def build_improvement_targets(rows):
    targets = []
    for row in rows:
        issue = ""
        priority = 4
        action = ""

        if (
            row["true_class"] == "Venomous"
            and row["species_risk"] == "Non Venomous"
        ):
            issue = "species_false_non_venomous"
            priority = 1
            action = (
                "Agregar ejemplos y hard negatives de la especie real y del grupo "
                "confundido; revisar si la region debe bloquear esta salida."
            )
        elif (
            row["true_class"] == "Venomous"
            and row["integrated_decision"] == "No concluyente"
        ):
            issue = "venomous_no_conclusive"
            priority = 2
            action = (
                "Mantener como salida preventiva por ahora; mejorar especie para "
                "convertir este caso en Venomous sin depender del binario."
            )
        elif (
            row["true_class"] == "Non Venomous"
            and row["integrated_decision"] == "Venomous"
        ):
            issue = "non_venomous_as_venomous"
            priority = 2
            action = (
                "Reforzar especies no venenosas visualmente parecidas a venenosas "
                "para reducir falsos positivos sin bajar recall Venomous."
            )
        elif (
            row["true_class"] == "Non Venomous"
            and row["integrated_decision"] == "No concluyente"
        ):
            issue = "non_venomous_no_conclusive"
            priority = 3
            action = (
                "Ajustar respaldo binario o regla integrada solo si no aumenta "
                "falsos Non Venomous en serpientes venenosas."
            )
        elif row["species_exact_ok"] == "ERROR":
            issue = "species_exact_miss"
            priority = 4
            action = "Mejorar discriminacion de especie exacta con mas datos por especie."

        if issue:
            targets.append(
                {
                    "priority": priority,
                    "issue": issue,
                    "flow": row["flow"],
                    "id": row["id"],
                    "image_path": row["image_path"],
                    "true_class": row["true_class"],
                    "true_species": row["true_species"],
                    "country_for_test": row["country_for_test"],
                    "region_query": row["region_query"],
                    "species_prediction": row["species_prediction"],
                    "species_risk": row["species_risk"],
                    "species_confidence": row["species_confidence"],
                    "binary_decision": row["binary_decision"],
                    "binary_p_venomous": row["binary_p_venomous"],
                    "integrated_decision": row["integrated_decision"],
                    "action": action,
                }
            )

    return sorted(
        targets,
        key=lambda row: (row["priority"], row["flow"], row["id"], row["issue"]),
    )


def summarize_targets(targets):
    by_issue = Counter(row["issue"] for row in targets)
    by_priority = Counter(str(row["priority"]) for row in targets)
    return {
        "total_targets": len(targets),
        "by_issue": dict(by_issue),
        "by_priority": dict(by_priority),
        "priority_1_count": by_priority.get("1", 0),
        "priority_2_count": by_priority.get("2", 0),
    }


def prediction_row(
    manifest_row,
    flow,
    region_query,
    species_profile_name,
    binary_profile_name,
    species_result,
    binary_result,
    decision_result,
):
    true_class = manifest_row["true_class"]
    true_species = manifest_row.get("scientific_name") or manifest_row.get("true_species", "")
    species_prediction = species_result["selected_scientific_name"]
    species_risk = species_result["selected_risk"]
    binary_decision = binary_result["decision"]
    integrated_decision = decision_result["decision"]
    return {
        "id": manifest_row["id"],
        "flow": flow,
        "image_path": manifest_row["image_path"],
        "true_class": true_class,
        "true_species": true_species,
        "common_name": manifest_row.get("common_name", ""),
        "country_for_test": manifest_row.get("country_for_test", ""),
        "region_query": region_query,
        "species_profile": species_profile_name,
        "binary_profile": binary_profile_name,
        "species_prediction": species_prediction,
        "species_display_name": species_result["selected_display_name"],
        "species_risk": species_risk,
        "species_confidence": round(species_result["selected_confidence"], 6),
        "species_exact_ok": result_ok(true_species, species_prediction),
        "species_risk_ok": result_ok(true_class, species_risk),
        "direct_species": species_result["direct_scientific_name"],
        "direct_risk": species_result["direct_risk"],
        "direct_confidence": round(species_result["direct_confidence"], 6),
        "region_used": species_result["region_used"],
        "region_method": species_result["region_method"],
        "binary_decision": binary_decision,
        "binary_p_venomous": round(binary_result["venomous_probability"], 6),
        "binary_argmax_class": binary_result["argmax_class"],
        "binary_argmax_confidence": round(binary_result["argmax_confidence"], 6),
        "binary_ok": result_ok(true_class, binary_decision),
        "integrated_decision": integrated_decision,
        "integrated_ok": integrated_ok(true_class, integrated_decision),
        "integrated_reason": decision_result["reason"],
    }


def evaluate(manifest_rows, species_profile_name, binary_profile_name, include_wrong_country):
    species_profile = app.SPECIES_PROFILES[species_profile_name]
    binary_profile = app.MODEL_PROFILES[binary_profile_name]
    available_countries = app.load_available_countries(species_profile["manifest"])
    rows = []

    for manifest_row in manifest_rows:
        image = Image.open(manifest_row["image_path"]).convert("RGB")
        binary_result = app.predict_binary(
            image,
            binary_profile["model"],
            binary_profile["classes"],
            binary_profile["venomous_threshold"],
        )

        flows = [
            ("sin_region", ""),
            ("pais_correcto", manifest_row.get("country_for_test", "")),
        ]
        if include_wrong_country:
            wrong_country = choose_wrong_country(
                manifest_row.get("country_for_test", ""), available_countries
            )
            if wrong_country:
                flows.append(("pais_incorrecto", wrong_country))

        for flow, region_query in flows:
            species_result = app.predict_species(image, region_query, species_profile)
            decision_result = app.integrated_decision(
                species_result,
                binary_result,
                DEFAULT_BINARY_CONFIDENCE_THRESHOLD,
            )
            rows.append(
                prediction_row(
                    manifest_row,
                    flow,
                    region_query,
                    species_profile_name,
                    binary_profile_name,
                    species_result,
                    binary_result,
                    decision_result,
                )
            )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evalua automaticamente el flujo actual de app_streamlit.py."
    )
    parser.add_argument("--external-dir", default=str(DEFAULT_EXTERNAL_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--species-profile", default=DEFAULT_SPECIES_PROFILE)
    parser.add_argument("--binary-profile", default=DEFAULT_BINARY_PROFILE)
    parser.add_argument("--include-wrong-country", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    external_dir = Path(args.external_dir)
    output_dir = Path(args.output_dir)
    manifest_path = external_dir / "manifest.csv"
    manifest_rows = read_csv(manifest_path)
    rows = evaluate(
        manifest_rows,
        args.species_profile,
        args.binary_profile,
        args.include_wrong_country,
    )
    targets = build_improvement_targets(rows)
    summary = {
        "external_dir": str(external_dir),
        "manifest": str(manifest_path),
        "species_profile": args.species_profile,
        "binary_profile": args.binary_profile,
        "support_images": len(manifest_rows),
        "support_predictions": len(rows),
        "flows": summarize_by_flow(rows),
        "improvement_targets": summarize_targets(targets),
        "next_improvement_rule": (
            "Mejorar primero los grupos con falsos Venomous->Non Venomous "
            "y reducir No concluyente en Non Venomous sin sacrificar safety recall."
        ),
    }
    write_csv(output_dir / "current_app_flow_results.csv", rows)
    write_csv(output_dir / "current_app_flow_improvement_targets.csv", targets)
    write_json(output_dir / "current_app_flow_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
