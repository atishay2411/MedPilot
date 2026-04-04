#!/usr/bin/env python3
"""Quick test for the deterministic classifier."""
import json
from app.services.deterministic_classifier import try_deterministic_classify

tests = [
    ("What can you do?", None, None),
    ("How many patients are present?", None, None),
    ("Add a patient named Nesh Rochwani", None, None),
    ("Name: Nesh Rochwani, Birthdate: 24 Apr 2000", "create_patient", {"given_name": "Nesh", "family_name": "Rochwani"}),
    ("Full name is Nesh Rochwani and dob is 24 April 2000", "create_patient", {"given_name": "Nesh", "family_name": "Rochwani"}),
    ("List all patients", None, None),
    ("Show conditions for Maria Santos", None, None),
    ("Search for John Doe", None, None),
    ("Tell me about all patients", None, None),
    ("Show me all patients", None, None),
    ("View vitals", None, None),
    ("Add patient John Smith born 1990-01-15", None, None),
]

for prompt, pending, collected in tests:
    result = try_deterministic_classify(prompt, pending_intent=pending, collected_entities=collected)
    if result:
        print(f"PROMPT: {prompt!r}")
        print(f"  => mode={result.mode}, intent={result.intent}, scope={result.scope}")
        print(f"     entities={json.dumps(result.entities)}")
        msg = result.response_message[:80].replace("\n", " ")
        print(f"     message={msg}...")
        print()
    else:
        print(f"PROMPT: {prompt!r} => FALLBACK TO LLM")
        print()
