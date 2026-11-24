import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

test_cases = [
    {
        "id": "Test 1 (Multi-attribute)",
        "text": "The suspect had a sharp jawline, short dark curly hair, and piercing blue eyes.",
    },
    {
        "id": "Test 2 (Unfamiliar wording)",
        "text": "The fleeing person displayed an aquiline nose and a notched left eyebrow.",
    },
    {
        "id": "Test 3 (Negation safety)",
        "text": "He was clean-shaven with no beard or mustache, but had prominent cheekbones.",
    },
    {
        "id": "Test 4 (Uncertainty)",
        "text": "I think he might have had a receding hairline, or maybe hazel eyes.",
    },
    {
        "id": "Test 5 (No recognizable facial features)",
        "text": "The suspect was wearing a dark blue hoodie and running very fast.",
    },
]

print("=" * 80)
print("PHASE 4 LIVE INFERENCE VERIFICATION (TESTS B & C)")
print("=" * 80)

results = []

for case in test_cases:
    print(f"\n--- {case['id']} ---")
    print(f"INPUT: \"{case['text']}\"")
    resp = client.post("/analyze-description", json={"witness_text": case["text"]})
    assert resp.status_code == 200, f"Failed with {resp.status_code}: {resp.text}"
    data = resp.json()

    attrs = data.get("attributes", {})
    extracted_summary = []
    for cat, info in attrs.items():
        extracted_summary.append({
            "category": cat,
            "value": info.get("value"),
            "state": info.get("state"),
            "confidence": info.get("confidence"),
            "spans": info.get("source_spans"),
            "start": info.get("start_char"),
            "end": info.get("end_char"),
            "by": info.get("extracted_by"),
        })

    model_info = data.get("model", {})
    print(f"Model: {model_info.get('name')} | Architecture: {model_info.get('architecture')} | Phase: {model_info.get('phase')}")
    print(f"Checkpoint SHA-256: {model_info.get('checkpoint_sha256')}")
    print(f"Overall Confidence: {data.get('overall_confidence')}")
    print(f"Uncertainties detected: {data.get('uncertainties')}")
    print(f"Contradictions detected: {data.get('contradictions')}")
    print(f"Extracted Attributes Count: {len(extracted_summary)}")
    for a in extracted_summary:
        print(f"  • {a['category']:<20} | value={str(a['value']):<20} | state={a['state']:<12} | offsets=[{a['start']}:{a['end']}] | span={a['spans']} | by={a['by']}")
        # Verify exact character offsets if provided
        if a["start"] is not None and a["end"] is not None:
            slice_text = case["text"][a["start"]:a["end"]]
            assert slice_text == a["spans"][0], f"Offset mismatch: '{slice_text}' != '{a['spans'][0]}'"

    results.append({
        "case": case["id"],
        "input": case["text"],
        "attributes": extracted_summary,
        "overall_confidence": data.get("overall_confidence"),
        "uncertainties": data.get("uncertainties"),
    })

print("\n" + "=" * 80)
print("TEST C DYNAMIC VARIATION CHECK:")
print("Are outputs dynamic and non-identical across inputs?")
active_attr_counts = set(
    len([a for a in r["attributes"] if a["value"] is not None or a["state"] != "unknown"])
    for r in results
)
unique_confidences = set(r["overall_confidence"] for r in results)
print(f"Distinct active attribute counts across cases: {active_attr_counts}")
print(f"Distinct overall confidences: {unique_confidences}")
assert len(active_attr_counts) > 1, "Outputs are static/hardcoded!"
assert len(unique_confidences) > 1, "Confidences are static/hardcoded!"
print("PASSED: Model produces dynamic, input-sensitive forensic predictions!")
print("=" * 80)
