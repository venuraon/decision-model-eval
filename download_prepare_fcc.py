#!/usr/bin/env python3
"""Download the newest FCC complaints and prepare Strands input JSONL.

The Socrata API query asks for only the newest 500 rows; the full multi-million
row FCC export is never downloaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


FCC_API = "https://opendata.fcc.gov/resource/3xyp-aqkj.json"
DEFAULT_LIMIT = 500
OPTIONS = [
    "billing_and_charges",
    "service_outage_or_quality",
    "device_or_equipment",
    "account_or_number_porting",
    "fraud_or_unwanted_contact",
    "human_review_or_other",
]
CRITERIA = {
    "billing_and_charges": "Charges, fees, bills, or unauthorized billing.",
    "service_outage_or_quality": "Availability, coverage, speed, connection, or service quality.",
    "device_or_equipment": "Devices, equipment, installation, or hardware.",
    "account_or_number_porting": "Accounts, carriers, or keeping/transferring a phone number.",
    "fraud_or_unwanted_contact": "Robocalls, telemarketing, spoofing, or unwanted messages.",
    "human_review_or_other": "Anything not covered above or too ambiguous to route automatically.",
}

# The FCC Issue field is used exclusively to derive a transparent, repeatable
# gold label. It is intentionally omitted from state_text so the model cannot
# simply read back the label it is being scored against.
ISSUE_MAP = {
    "billing": "billing_and_charges",
    "charges": "billing_and_charges",
    "cramming": "billing_and_charges",
    "internet service": "service_outage_or_quality",
    "service quality": "service_outage_or_quality",
    "availability": "service_outage_or_quality",
    "phone service": "service_outage_or_quality",
    "robocalls": "fraud_or_unwanted_contact",
    "telemarketing": "fraud_or_unwanted_contact",
    "unwanted calls": "fraud_or_unwanted_contact",
    "spoofing": "fraud_or_unwanted_contact",
    "number portability": "account_or_number_porting",
    "porting": "account_or_number_porting",
    "equipment": "device_or_equipment",
    "device": "device_or_equipment",
}


def get_json(url: str) -> list[dict]:
    request = Request(url, headers={"User-Agent": "decision-model-eval/1.0"})
    with urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("FCC API returned a non-list response")
    return payload


def normalize(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def queue_label(row: dict) -> str:
    text = " ".join(
        normalize(row.get(field)).lower()
        for field in ("issue", "issue_type", "method")
    )
    for phrase, label in ISSUE_MAP.items():
        if phrase in text:
            return label
    return "human_review_or_other"


def state_text(row: dict) -> str:
    pieces = [
        f"Service/form: {normalize(row.get('issue_type'))}",
        f"Method: {normalize(row.get('method'))}",
        f"Call or message type: {normalize(row.get('type_of_call_or_messge'))}",
        "Property/goods/service type: "
        f"{normalize(row.get('type_of_property_goods_or_services'))}",
    ]
    return "\n".join(piece for piece in pieces if not piece.endswith(": "))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_outputs(rows: list[dict], output_dir: Path, source_url: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = []
    for row in rows:
        prepared.append(
            {
                "record_id": normalize(row.get("id")),
                "filed_date": normalize(row.get("ticket_created")),
                "state": state_text(row),
                "question": "Which queue should own this complaint?",
                "options": OPTIONS,
                "criteria": CRITERIA,
                "gold_label": queue_label(row),
                "source_fields": {
                    "issue": normalize(row.get("issue")),
                    "issue_type": normalize(row.get("issue_type")),
                    "method": normalize(row.get("method")),
                    "type_of_call_or_messge": normalize(
                        row.get("type_of_call_or_messge")
                    ),
                    "type_of_property_goods_or_services": normalize(
                        row.get("type_of_property_goods_or_services")
                    ),
                },
            }
        )
    jsonl_path = output_dir / "fcc_latest_500_prepared.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for item in prepared:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    csv_path = output_dir / "fcc_latest_500_prepared.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["record_id", "filed_date", "state", "gold_label"],
        )
        writer.writeheader()
        writer.writerows(
            {key: item[key] for key in writer.fieldnames} for item in prepared
        )
    manifest = {
        "source_url": source_url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": len(prepared),
        "newest_filed_date": prepared[0]["filed_date"] if prepared else None,
        "oldest_filed_date": prepared[-1]["filed_date"] if prepared else None,
        "record_ids": [item["record_id"] for item in prepared],
        "options": OPTIONS,
        "criteria": CRITERIA,
        "label_mapping": ISSUE_MAP,
        "jsonl_sha256": sha256(jsonl_path),
    }
    (output_dir / "fcc_latest_500_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = Path(__file__).resolve().parent / "data"
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 5000:
        parser.error("--limit must be between 1 and 5000")
    query = urlencode(
        {
            "$limit": args.limit,
            "$order": "ticket_created DESC",
            "$select": (
                "id,ticket_created,issue,issue_type,method,"
                "type_of_call_or_messge,type_of_property_goods_or_services"
            ),
        }
    )
    url = f"{FCC_API}?{query}"
    print(f"Requesting {args.limit} newest records from FCC Socrata API...")
    rows = get_json(url)
    if not rows:
        print("FCC API returned no rows", file=sys.stderr)
        return 1
    write_outputs(rows, args.output_dir, url)
    print(f"Wrote {len(rows)} prepared records to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
