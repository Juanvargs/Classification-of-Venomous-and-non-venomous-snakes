import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.client import RemoteDisconnected
from pathlib import Path
from urllib.error import HTTPError, URLError

from PIL import Image


TARGETS_PATH = Path("data/collection_targets/confusion_targets_b1.csv")
OUTPUT_DIR = Path("data/raw/b1_confusion_targets_inaturalist")
USER_AGENT = "SnakeClassificationStudentProject/1.0 (educational dataset improvement)"
IMAGE_SUFFIX = ".jpg"
OPEN_LICENSES = {"cc0", "cc-by", "cc-by-sa"}
REQUEST_ERRORS = (HTTPError, URLError, TimeoutError, RemoteDisconnected, ConnectionError, OSError)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Descarga imagenes abiertas de iNaturalist para pares confundidos del B1."
    )
    parser.add_argument("--targets", default=str(TARGETS_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--min-priority", type=int, default=1)
    parser.add_argument("--max-priority", type=int, default=2)
    parser.add_argument("--model-target", default="species_b1")
    parser.add_argument("--max-per-species", type=int, default=None)
    parser.add_argument("--per-page", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild-manifest-only", action="store_true")
    return parser.parse_args()


def request_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def download(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=90) as response:
        destination.write_bytes(response.read())


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


def sanitize(value):
    value = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    return re.sub(r"_+", "_", value).strip("_")


def species_from_targets(rows, min_priority, max_priority, model_target):
    selected = {}
    for row in rows:
        priority = int(row["priority"])
        if priority < min_priority or priority > max_priority:
            continue
        if row["model_target"] != model_target:
            continue
        species = row["true_species"].strip()
        query_species = row.get("query_species", "").strip() or species
        current = selected.get(species)
        min_images = int(row["min_images"])
        if not current or min_images > current["min_images"]:
            selected[species] = {
                "scientific_name": species,
                "query_species": query_species,
                "true_class": row["true_class"],
                "priority": priority,
                "min_images": min_images,
                "issues": set(),
                "confused_with": set(),
                "region_focus": set(),
            }
        selected[species]["issues"].add(row["issue"])
        selected[species]["confused_with"].update(
            item.strip() for item in row["confused_with"].split("|") if item.strip()
        )
        selected[species]["region_focus"].update(
            item.strip() for item in row["region_focus"].split("|") if item.strip()
        )
        selected[species].setdefault("accepted_taxon_names", set()).update(
            {species, query_species}
        )

    normalized = []
    for item in selected.values():
        item["issues"] = " | ".join(sorted(item["issues"]))
        item["confused_with"] = " | ".join(sorted(item["confused_with"]))
        item["region_focus"] = " | ".join(sorted(item["region_focus"]))
        item["accepted_taxon_names"] = sorted(item.get("accepted_taxon_names", {item["scientific_name"]}))
        normalized.append(item)
    return sorted(normalized, key=lambda row: (row["priority"], row["scientific_name"]))


def inaturalist_observations(scientific_name, page, per_page):
    params = {
        "taxon_name": scientific_name,
        "photos": "true",
        "quality_grade": "research",
        "photo_license": ",".join(sorted(OPEN_LICENSES)),
        "license": ",".join(sorted(OPEN_LICENSES)),
        "order": "desc",
        "order_by": "created_at",
        "per_page": str(per_page),
        "page": str(page),
    }
    url = "https://api.inaturalist.org/v1/observations?" + urllib.parse.urlencode(params)
    return request_json(url)


def photo_large_url(photo):
    url = photo.get("url", "")
    if not url:
        return ""
    return (
        url.replace("/square.", "/large.")
        .replace("/small.", "/large.")
        .replace("/medium.", "/large.")
    )


def license_code(observation, photo):
    return (photo.get("license_code") or observation.get("license_code") or "").lower()


def observation_candidates(scientific_name, query_species, accepted_taxon_names, per_page, sleep_seconds):
    page = 1
    seen = set()
    while True:
        try:
            payload = inaturalist_observations(query_species, page, per_page)
        except REQUEST_ERRORS as error:
            yield {"error": str(error)}
            return

        results = payload.get("results", [])
        if not results:
            return

        for observation in results:
            taxon_name = observation.get("taxon", {}).get("name", "")
            if taxon_name not in accepted_taxon_names:
                continue
            observation_id = str(observation.get("id", ""))
            if observation_id in seen:
                continue
            seen.add(observation_id)
            photos = observation.get("photos") or []
            if not photos:
                continue
            photo = photos[0]
            license_name = license_code(observation, photo)
            if license_name not in OPEN_LICENSES:
                continue
            image_url = photo_large_url(photo)
            if not image_url:
                continue
            yield {
                "observation_id": observation_id,
                "scientific_name": scientific_name,
                "query_species": query_species,
                "taxon_name": taxon_name,
                "image_url": image_url,
                "source_page": f"https://www.inaturalist.org/observations/{observation_id}",
                "license": license_name,
                "attribution": photo.get("attribution")
                or observation.get("user", {}).get("login", ""),
                "observed_place": observation.get("place_guess", ""),
            }

        total_results = int(payload.get("total_results", 0) or 0)
        if page * per_page >= total_results:
            return
        page += 1
        time.sleep(sleep_seconds)


def valid_image(path):
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        return width >= 250 and height >= 180
    except Exception:
        return False


def manifest_row_for_existing(path, target):
    observation_id = path.stem.split("_")[-1]
    source_page = (
        f"https://www.inaturalist.org/observations/{observation_id}"
        if observation_id.isdigit()
        else ""
    )
    return {
        "image_path": str(path),
        "scientific_name": target["scientific_name"],
        "query_species": target.get("query_species", target["scientific_name"]),
        "taxon_name": target.get("query_species", target["scientific_name"]),
        "true_class": target["true_class"],
        "priority": target["priority"],
        "issues": target["issues"],
        "confused_with": target["confused_with"],
        "region_focus": target["region_focus"],
        "source": "iNaturalist",
        "source_page": source_page,
        "image_url": "",
        "license": "",
        "attribution": "",
        "observed_place": "",
        "downloaded_at": "",
        "notes": "Existing local image reconstructed into manifest.",
    }


def collect_for_species(
    target,
    output_dir,
    limit,
    per_page,
    sleep_seconds,
    dry_run,
    rebuild_manifest_only=False,
):
    species_slug = sanitize(target["scientific_name"])
    species_dir = output_dir / species_slug
    species_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    failures = []
    checked = 0

    collected_paths = set()
    existing = sorted(species_dir.glob(f"*{IMAGE_SUFFIX}"))
    for path in existing:
        if valid_image(path):
            collected_paths.add(path)
            manifest_rows.append(manifest_row_for_existing(path, target))

    if rebuild_manifest_only or len(collected_paths) >= limit:
        return manifest_rows, failures, {
            "species": target["scientific_name"],
            "available": len(collected_paths),
        }

    for candidate in observation_candidates(
        target["scientific_name"],
        target.get("query_species", target["scientific_name"]),
        set(target.get("accepted_taxon_names", [target["scientific_name"]])),
        per_page,
        sleep_seconds,
    ):
        if "error" in candidate:
            failures.append(
                {
                    "scientific_name": target["scientific_name"],
                    "reason": candidate["error"],
                }
            )
            break

        checked += 1
        filename = f"{species_slug}_{candidate['observation_id']}{IMAGE_SUFFIX}"
        destination = species_dir / filename
        if dry_run:
            collected_paths.add(destination)
        elif destination.exists() and valid_image(destination):
            collected_paths.add(destination)
        else:
            try:
                download(candidate["image_url"], destination)
            except REQUEST_ERRORS as error:
                destination.unlink(missing_ok=True)
                failures.append(
                    {
                        "scientific_name": target["scientific_name"],
                        "source_page": candidate["source_page"],
                        "reason": f"download failed: {error}",
                    }
                )
                continue
            if not valid_image(destination):
                destination.unlink(missing_ok=True)
                failures.append(
                    {
                        "scientific_name": target["scientific_name"],
                        "source_page": candidate["source_page"],
                        "reason": "invalid or too small image",
                    }
                )
                continue
            collected_paths.add(destination)

        manifest_rows.append(
            {
                "image_path": str(destination),
                "scientific_name": target["scientific_name"],
                "query_species": candidate.get("query_species", target["scientific_name"]),
                "taxon_name": candidate.get("taxon_name", target["scientific_name"]),
                "true_class": target["true_class"],
                "priority": target["priority"],
                "issues": target["issues"],
                "confused_with": target["confused_with"],
                "region_focus": target["region_focus"],
                "source": "iNaturalist",
                "source_page": candidate["source_page"],
                "image_url": candidate["image_url"],
                "license": candidate["license"],
                "attribution": candidate["attribution"],
                "observed_place": candidate["observed_place"],
                "downloaded_at": datetime.now(timezone.utc).isoformat()
                if not dry_run
                else "",
                "notes": "B1 confusion-target open-license collection.",
            }
        )

        if len(collected_paths) >= limit:
            break
        time.sleep(sleep_seconds)

    return manifest_rows, failures, {
        "species": target["scientific_name"],
        "target": limit,
        "downloaded_or_available": len(collected_paths),
        "checked_candidates": checked,
    }


def main():
    args = parse_args()
    targets_path = Path(args.targets)
    output_dir = Path(args.output_dir)
    report_dir = output_dir / "reports"
    targets = species_from_targets(
        read_csv(targets_path),
        min_priority=args.min_priority,
        max_priority=args.max_priority,
        model_target=args.model_target,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_failures = []
    summaries = []

    for target in targets:
        limit = args.max_per_species or target["min_images"]
        rows, failures, summary = collect_for_species(
            target,
            output_dir,
            limit,
            args.per_page,
            args.sleep,
            args.dry_run,
            args.rebuild_manifest_only,
        )
        all_rows.extend(rows)
        all_failures.extend(failures)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=True))
        if not args.dry_run:
            write_csv(output_dir / "manifest.csv", all_rows)
            write_csv(report_dir / "download_failures.csv", all_failures)

    if not args.dry_run:
        write_csv(output_dir / "manifest.csv", all_rows)
        write_csv(report_dir / "download_failures.csv", all_failures)

    summary_payload = {
        "targets_path": str(targets_path),
        "output_dir": str(output_dir),
        "dry_run": args.dry_run,
        "target_species": len(targets),
        "manifest_rows": len(all_rows),
        "failures": len(all_failures),
        "species": summaries,
    }
    summary_path = report_dir / ("dry_run_summary.json" if args.dry_run else "summary.json")
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
