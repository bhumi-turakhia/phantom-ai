import json
from schemas.attributes import AttributeCategory, EvidenceState
from app.main import nlp_service

new_cases = [
    {
        "id": "Corpus Case 1 (Formal Police Blotter)",
        "text": "Complainant stated the fleeing suspect possessed an oval face, thick dark eyebrows, and a hooked nose with a small scar above the lip.",
    },
    {
        "id": "Corpus Case 2 (Informal / Colloquial with Negation)",
        "text": "I saw this guy sprinting down the alley; he was totally clean-shaven, definitely no mustache or beard, but had these really prominent high cheekbones.",
    },
    {
        "id": "Corpus Case 3 (Hedged / Uncertain Witness)",
        "text": "It was pretty dark so I am not certain, but I think he possibly had sandy blonde hair and sort of wide-set eyes.",
    },
    {
        "id": "Corpus Case 4 (Conflicting / Contradictory Testimony)",
        "text": "The first bystander shouted the robber was completely bald, while the cashier swore he had a long messy ponytail.",
    },
    {
        "id": "Corpus Case 5 (Scene & Non-Facial Observables Only)",
        "text": "The perpetrator bolted past wearing an oversized black leather jacket with silver zipper pulls and carried a neon green duffel bag.",
    },
]

print("=" * 80)
print("PHASE 4.1 AUDIT: 5 PREVIOUSLY UNUSED WITNESS DESCRIPTIONS")
print("=" * 80)

for c in new_cases:
    print(f"\n--- {c['id']} ---")
    print(f"INPUT TEXT: \"{c['text']}\"")
    status, is_trained, attrs, contradictions, uncertainties, overall_conf, notes = nlp_service.analyze_description(c["text"])

    active_extracted = [
        (k, ev.value, ev.state.value if hasattr(ev.state, "value") else str(ev.state), ev.confidence, ev.start_char, ev.end_char, ev.source_spans)
        for k, ev in attrs.items()
        if ev.value is not None or (ev.state.value if hasattr(ev.state, "value") else str(ev.state)) != "unknown"
    ]

    print(f"Status: {status} | Trained: {is_trained} | Model Score: {overall_conf}")
    print(f"Contradictions detected: {contradictions}")
    print(f"Uncertainty cues detected: {uncertainties}")
    print(f"Active Extracted Features ({len(active_extracted)}):")
    for cat, val, st, conf, s, e, spans in active_extracted:
        span_str = spans[0] if spans else ""
        offsets_str = f"[{s}:{e}]" if s is not None and e is not None else "[N/A]"
        print(f"  • {cat:<22} | value={str(val):<26} | state={st:<14} | offsets={offsets_str:<10} | span='{span_str}'")
        if s is not None and e is not None and spans:
            extracted_sub = c["text"][s:e]
            assert extracted_sub == span_str, f"Mismatch: '{extracted_sub}' != '{span_str}'"
            print(f"    -> Zero-drift verification: text[{s}:{e}] == '{extracted_sub}' (PASS)")

print("\n" + "=" * 80)
print("ALL 5 AUDIT DESCRIPTIONS VERIFIED THROUGH ACTUAL MODEL INFERENCE PATH.")
print("=" * 80)
