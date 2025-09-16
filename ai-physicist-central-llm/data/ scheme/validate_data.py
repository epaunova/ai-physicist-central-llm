#!/usr/bin/env python3
"""
Validate data files against JSON Schemas.
Usage:
  python scripts/validate_data.py
"""

import json
from pathlib import Path
from typing import Tuple
from jsonschema import validate, Draft7Validator

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = ROOT / "data" / "schema"

DATA_CORPUS = ROOT / "data" / "corpus" / "physics_abstracts.json"
SCHEMA_CORPUS = SCHEMA_DIR / "corpus.schema.json"

DATA_QA = ROOT / "data" / "evaluation" / "physics_qa_dataset.json"
SCHEMA_QA = SCHEMA_DIR / "qa_dataset.schema.json"

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def validate_json(data, schema) -> Tuple[bool, str]:
    v = Draft7Validator(schema)
    errors = sorted(v.iter_errors(data), key=lambda e: e.path)
    if not errors:
        return True, "OK"
    msg = "\n".join([f"- {'/'.join(map(str, e.path))}: {e.message}" for e in errors])
    return False, msg

def main():
    print("=" * 60)
    print("VALIDATING DATA AGAINST SCHEMAS")
    print("=" * 60)

    # Corpus
    corpus = load_json(DATA_CORPUS)
    corpus_schema = load_json(SCHEMA_CORPUS)
    ok, msg = validate_json(corpus, corpus_schema)
    print(f"\n[corpus] {DATA_CORPUS.name}: {'✓ VALID' if ok else '✗ INVALID'}")
    if not ok:
        print(msg)

    # QA dataset
    qa = load_json(DATA_QA)
    qa_schema = load_json(SCHEMA_QA)
    ok2, msg2 = validate_json(qa, qa_schema)
    print(f"\n[qa] {DATA_QA.name}: {'✓ VALID' if ok2 else '✗ INVALID'}")
    if not ok2:
        print(msg2)

    if ok and ok2:
        print("\nAll datasets valid ✓")
    else:
        print("\nSome datasets failed validation ✗")

if __name__ == "__main__":
    main()
