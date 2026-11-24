"""
generate_out_of_frame_dataset.py
PHANTOM AI 3.0 — Phase 3B.2

Generates an independently authored out-of-frame evaluation corpus of ≥ 100
synthetic witness descriptions. NONE of these descriptions share sentence frames
with the training/validation/test generator.

Linguistic registers intentionally different from the existing dataset:
  - 911 emergency dispatch dialogue (Operator / Caller turn structure)
  - Courtroom direct & cross-examination Q&A
  - Security guard shift log entries (terse, note-taking style)
  - Detective handwritten notebook fragments
  - Interrupted spoken recall with discourse markers
  - Single-word / two-word telegraphic attribute statements
  - Complex negations ("not the kind of … but rather …")
  - Multi-witness contradiction format
  - Passive constructions ("it was noted that …")
  - Present-tense news-style reporting ("The suspect is described as …")

CONSTRAINTS:
  - Zero gradient updates from this corpus at any time.
  - Zero word-for-word frame re-use from existing synthetic generator.
  - All character offsets verified: text[start:end] == source_span.
  - All samples marked is_synthetic = True.
  - Annotator ID = "OUT-OF-FRAME-AUTHOR-INDEPENDENT"
"""

import json
import sys
import os
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

# ---------------------------------------------------------------------------
# Verified helper: ensures offset integrity before committing a sample
# ---------------------------------------------------------------------------

def _ann(
    sample_id: str,
    text: str,
    category: str,
    span: str,
    certainty: str = "explicit",
    notes: str = "",
) -> Dict[str, Any]:
    """
    Build an annotation dict with verified character offsets.
    Raises ValueError if the span is not exactly present in text.
    """
    idx = text.find(span)
    if idx == -1:
        raise ValueError(
            f"[OFFSET ERROR] Sample {sample_id!r}: span {span!r} not found in text:\n  {text!r}"
        )
    end_idx = idx + len(span)
    assert text[idx:end_idx] == span, "Offset verification failed"
    return {
        "annotation_id": f"{sample_id}-ANN-{uuid.uuid4().hex[:8]}",
        "category": category,
        "attribute_value": span,
        "source_span": span,
        "start_char": idx,
        "end_char": end_idx,
        "certainty": certainty,
        "annotator_id": "OUT-OF-FRAME-AUTHOR-INDEPENDENT",
        "annotation_timestamp": datetime.now(timezone.utc).isoformat(),
        "uncertainty_cues": [],
        "notes": notes or f"Out-of-frame independent evaluation annotation – category: {category}",
    }


def _unk(sample_id: str, category: str) -> Dict[str, Any]:
    return {
        "annotation_id": f"{sample_id}-UNK-{uuid.uuid4().hex[:8]}",
        "category": category,
        "attribute_value": None,
        "source_span": None,
        "start_char": None,
        "end_char": None,
        "certainty": "unknown",
        "annotator_id": "OUT-OF-FRAME-AUTHOR-INDEPENDENT",
        "annotation_timestamp": datetime.now(timezone.utc).isoformat(),
        "uncertainty_cues": [],
        "notes": "Feature not mentioned in this out-of-frame description.",
    }


def _sample(sid: str, text: str, annotations: List[Dict]) -> Dict[str, Any]:
    return {
        "sample_id": sid,
        "witness_text": text,
        "annotations": annotations,
        "metadata": {
            "dataset_version": "3.0.0-oof",
            "source": "Phantom AI Out-Of-Frame Independent Evaluation Corpus",
            "license": "CC-BY-4.0-Synthetic",
            "is_synthetic": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "notes": (
                "Independent out-of-frame evaluation corpus for Phase 3B.2 generalisation audit. "
                "NOT used for training, validation, or hyperparameter tuning."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Build all samples — hand-authored, linguistically diverse
# ---------------------------------------------------------------------------

def build_samples() -> List[Dict[str, Any]]:
    samples = []
    n = [0]  # mutable counter

    def sid() -> str:
        n[0] += 1
        return f"OOF-{n[0]:04d}"

    # -----------------------------------------------------------------------
    # REGISTER 1: 911 Dispatch dialogue format
    # -----------------------------------------------------------------------
    def dispatch(text: str, anns: List[Dict]) -> Dict:
        return _sample(sid(), text, anns)

    s = "Operator: Can you describe the individual? Caller: Yeah, he had this really bushy beard, like super thick, and his hair was completely shaved on the sides."
    samples.append(dispatch(s, [
        _ann(s.split("|")[0] if "|" in s else sid(), s, "facial_hair", "bushy beard"),
        _ann(s.split("|")[0] if "|" in s else sid(), s, "hair", "shaved on the sides"),
    ]))
    # Fix: rebuild with correct sid
    samples[-1]["sample_id"] = f"OOF-{n[0]:04d}"
    # The above approach is awkward; redo with direct sid calls properly:
    samples.clear()
    n[0] = 0

    # ---- proper generation loop ----

    # --- REGISTER 1: 911 Dispatch ---
    raw = [
        (
            "Operator: Can you describe the person? Caller: He had this really bushy beard, super thick, and his hair was shaved on the sides.",
            [("facial_hair", "bushy beard"), ("hair", "shaved on the sides")],
        ),
        (
            "Operator: What did the suspect look like? Caller: I only caught a glimpse — pale face, really prominent nose, you know, kind of hooked.",
            [("nose", "prominent nose")],
        ),
        (
            "Caller: The guy had a scar, like a diagonal cut across his left cheek. Operator: Anything else? Caller: His eyebrows were really thin, almost invisible.",
            [("distinguishing_marks", "scar"), ("eyebrows", "thin")],
        ),
        (
            "Caller: He was bald, completely bald on top, but had this full goatee. Operator: Eye colour? Caller: Brown, definitely brown eyes.",
            [("hair", "bald"), ("facial_hair", "goatee"), ("eyes", "brown eyes")],
        ),
        (
            "Operator: Hair? Caller: Dark, straight hair. Not long though. Operator: Facial features? Caller: Wide nose. That's what I remember.",
            [("hair", "Dark, straight hair"), ("nose", "Wide nose")],
        ),
        (
            "Caller: She had really high cheekbones — you know, like a model's face — and her lips were painted bright red.",
            [("face_shape", "high cheekbones"), ("lips", "bright red")],
        ),
        (
            "Operator: Anything distinctive about the face? Caller: He had one of those unibrows, you know, where they meet in the middle.",
            [("eyebrows", "unibrow")],
        ),
        (
            "Caller: Wait — he had a gap between his front teeth. And his jaw was really square, blocky. Operator: Hair colour? Caller: Black.",
            [("distinguishing_marks", "gap between his front teeth"), ("face_shape", "square"), ("hair", "Black")],
        ),
    ]
    for text, spans in raw:
        i = sid()
        anns = [_ann(i, text, cat, span) for cat, span in spans]
        samples.append(_sample(i, text, anns))

    # --- REGISTER 2: Courtroom direct/cross-examination ---
    raw = [
        (
            "Q: Can you describe the defendant's hair? A: It was auburn, cut close on the sides but longer on top. Q: And his facial hair? A: A thin moustache, neatly trimmed.",
            [("hair", "auburn"), ("facial_hair", "thin moustache")],
        ),
        (
            "Q: To the best of your recollection, what colour were her eyes? A: A light green, almost grey depending on the light.",
            [("eyes", "light green")],
        ),
        (
            "Q: Did you observe any distinguishing marks on the individual? A: There was a tattoo on the left side of his neck.",
            [("distinguishing_marks", "tattoo on the left side of his neck")],
        ),
        (
            "Q: Describe the nose. A: Broad, with a slight bump at the bridge. Q: The ears? A: Protruding slightly, like they stuck out.",
            [("nose", "Broad"), ("other_observable", "ears")],
        ),
        (
            "Q: What can you tell the court about the defendant's lips? A: Full lips, lower lip noticeably fuller than the upper.",
            [("lips", "Full lips")],
        ),
        (
            "Q: You say you saw the assailant's face — describe it. A: Oval-shaped face, I'd say. Somewhat gaunt. Hollow cheeks.",
            [("face_shape", "Oval-shaped face")],
        ),
        (
            "Defence counsel: Isn't it true you only saw the suspect briefly? Witness: I saw enough. He had a receding hairline and a cleft chin.",
            [("hairline", "receding hairline"), ("face_shape", "cleft chin")],
        ),
        (
            "Q: Did you see any facial hair? A: Not exactly. Some stubble, a few days' growth at most, on his chin.",
            [("facial_hair", "stubble")],
        ),
        (
            "Q: The eyebrows — describe them. A: Arched, rather dramatically arched. And very dark compared to his lighter hair.",
            [("eyebrows", "Arched"), ("hair", "lighter hair")],
        ),
        (
            "Q: Was there anything unusual about the face shape? A: It was quite long. A long, narrow face. A narrow chin.",
            [("face_shape", "long, narrow face")],
        ),
    ]
    for text, spans in raw:
        i = sid()
        anns = [_ann(i, text, cat, span) for cat, span in spans]
        samples.append(_sample(i, text, anns))

    # --- REGISTER 3: Security guard shift log (terse note-taking) ---
    raw = [
        (
            "21:42 — Male suspect exited rear entrance. Shaved head. Heavy stubble. Approximately 30–35.",
            [("hair", "Shaved head"), ("facial_hair", "Heavy stubble")],
        ),
        (
            "Incident log 14-B: Subject described by cashier as having red curly hair and freckles across the nose.",
            [("hair", "red curly hair"), ("distinguishing_marks", "freckles")],
        ),
        (
            "CCTV review note: Individual had prominent ears, ears stuck out noticeably. Wore no hat. Hair: cropped, dark brown.",
            [("other_observable", "prominent ears"), ("hair", "cropped, dark brown")],
        ),
        (
            "Shift report: Suspicious male, bald patch at crown, thick black eyebrows, no visible facial hair.",
            [("hair", "bald patch at crown"), ("eyebrows", "thick black eyebrows")],
        ),
        (
            "Guard 3 observation: Female, long straight blonde hair, blue eyes, small upturned nose.",
            [("hair", "long straight blonde hair"), ("eyes", "blue eyes"), ("nose", "small upturned nose")],
        ),
        (
            "Entry 07: Male with distinctive features — deep-set eyes, prominent brow ridge, square jaw.",
            [("eyes", "deep-set eyes"), ("face_shape", "square jaw")],
        ),
        (
            "Suspect #2: Short grey hair. Trimmed beard. Scar running from left ear toward chin.",
            [("hair", "Short grey hair"), ("facial_hair", "Trimmed beard"), ("distinguishing_marks", "Scar running from left ear toward chin")],
        ),
        (
            "Note: Witness unsure of hair colour — possibly dark, possibly brown. Eyes described as 'dull', hazel maybe.",
            [("eyes", "hazel")],
        ),
    ]
    for text, spans in raw:
        i = sid()
        anns = [_ann(i, text, cat, span) for cat, span in spans]
        samples.append(_sample(i, text, anns))

    # --- REGISTER 4: Detective notebook fragments ---
    raw = [
        (
            "Subject: mid-40s. Widow's peak hairline. Grey at temples. Wire-rimmed glasses.",
            [("hairline", "Widow's peak hairline"), ("hair", "Grey at temples"), ("other_observable", "Wire-rimmed glasses")],
        ),
        (
            "Description from informant: small eyes — narrow, close-set — and a wide, flat nose.",
            [("eyes", "narrow, close-set"), ("nose", "wide, flat nose")],
        ),
        (
            "DB's notes: 3rd witness says large mole near right eyebrow. Corroborated by 1st witness.",
            [("distinguishing_marks", "large mole near right eyebrow")],
        ),
        (
            "Suspect physical: heart-shaped face, thin lips, barely any eyebrows.",
            [("face_shape", "heart-shaped face"), ("lips", "thin lips"), ("eyebrows", "barely any eyebrows")],
        ),
        (
            "Johnson's description: pockmarked skin on cheeks. Nose broken previously — visible deviation to the right.",
            [("distinguishing_marks", "pockmarked skin on cheeks"), ("nose", "deviation to the right")],
        ),
        (
            "Confidential informant: bushy white eyebrows. Big ears. No beard.",
            [("eyebrows", "bushy white eyebrows"), ("other_observable", "Big ears")],
        ),
        (
            "Rough sketch notes: triangular chin, prominent cheekbones, deep-set pale eyes.",
            [("face_shape", "triangular chin"), ("face_shape", "prominent cheekbones"), ("eyes", "deep-set pale eyes")],
        ),
    ]
    for text, spans in raw:
        i = sid()
        anns = [_ann(i, text, cat, span) for cat, span in spans]
        samples.append(_sample(i, text, anns))

    # --- REGISTER 5: Interrupted recall with discourse markers ---
    raw = [
        (
            "He had — wait — brown hair, no, it was more of a dark auburn. And, um, a beard. Yeah, a full beard.",
            [("hair", "dark auburn"), ("facial_hair", "full beard")],
        ),
        (
            "The eyes — I keep coming back to the eyes — they were this icy blue. Pale. Almost white-blue.",
            [("eyes", "icy blue")],
        ),
        (
            "Hold on, let me think. She was — her face was round, definitely round. And her hair was — I want to say curly? Brown and curly.",
            [("face_shape", "round"), ("hair", "Brown and curly")],
        ),
        (
            "His nose, right, it was like — have you ever seen someone who broke their nose? It kind of went sideways.",
            [("nose", "broke their nose")],
        ),
        (
            "Something about the lips — they were thin. Very thin. Like he was pursing them. And — oh — a small goatee.",
            [("lips", "thin"), ("facial_hair", "small goatee")],
        ),
        (
            "I didn't get a great look but — the hairline, yeah — he had this widow's peak. Very distinctive.",
            [("hairline", "widow's peak")],
        ),
        (
            "Okay so — the eyebrows. They were, like, super thick. Almost a monobrow situation. You know what I mean?",
            [("eyebrows", "super thick")],
        ),
        (
            "Wait — tattoo. There was something on his neck. Like a symbol or lettering — I couldn't read it.",
            [("distinguishing_marks", "tattoo")],
        ),
    ]
    for text, spans in raw:
        i = sid()
        anns = [_ann(i, text, cat, span) for cat, span in spans]
        samples.append(_sample(i, text, anns))

    # --- REGISTER 6: Telegraphic single / two-word attribute statements ---
    raw = [
        ("Clean-shaven. Dark eyes. Curly hair.", [("facial_hair", "Clean-shaven"), ("eyes", "Dark eyes"), ("hair", "Curly hair")]),
        ("Bald. Scar on chin.", [("hair", "Bald"), ("distinguishing_marks", "Scar on chin")]),
        ("Blonde. Blue-eyed. Slim nose.", [("hair", "Blonde"), ("eyes", "Blue-eyed"), ("nose", "Slim nose")]),
        ("Stubble. Square face.", [("facial_hair", "Stubble"), ("face_shape", "Square face")]),
        ("Tattoos. Shaved head.", [("distinguishing_marks", "Tattoos"), ("hair", "Shaved head")]),
        ("Full beard. Thick eyebrows. Broad forehead.", [("facial_hair", "Full beard"), ("eyebrows", "Thick eyebrows"), ("hairline", "Broad forehead")]),
        ("Pale. Freckles. Thin lips.", [("distinguishing_marks", "Freckles"), ("lips", "Thin lips")]),
        ("Wavy hair. Hazel eyes. Crooked nose.", [("hair", "Wavy hair"), ("eyes", "Hazel eyes"), ("nose", "Crooked nose")]),
        ("Heavy brow. Deep-set eyes.", [("face_shape", "Heavy brow"), ("eyes", "Deep-set eyes")]),
        ("Dreadlocks. Sparse beard.", [("hair", "Dreadlocks"), ("facial_hair", "Sparse beard")]),
    ]
    for text, spans in raw:
        i = sid()
        anns = [_ann(i, text, cat, span) for cat, span in spans]
        samples.append(_sample(i, text, anns))

    # --- REGISTER 7: Complex negations ---
    raw = [
        (
            "He didn't have a conventional beard — it was more of a patchy, sparse growth across the cheeks.",
            [("facial_hair", "patchy, sparse growth across the cheeks")],
        ),
        (
            "The hair wasn't dark exactly — not black, not brown — more of a very dark auburn.",
            [("hair", "very dark auburn")],
        ),
        (
            "Not slim, not wide — the nose was perfectly average, I'd say, except for a small bump at the top.",
            [("nose", "small bump at the top")],
        ),
        (
            "I wouldn't call the eyes blue — they were more grey, steel grey — definitely not brown.",
            [("eyes", "steel grey")],
        ),
        (
            "His face wasn't round, wasn't square — it was more of an angular diamond shape.",
            [("face_shape", "angular diamond shape")],
        ),
        (
            "She didn't have full lips — quite the opposite. Very thin, almost disappearing.",
            [("lips", "Very thin, almost disappearing")],
        ),
        (
            "No beard at all. Not even stubble. Completely smooth chin.",
            [("facial_hair", "Completely smooth chin")],
        ),
        (
            "His hairline hadn't receded — if anything it was very low, almost meeting the eyebrows.",
            [("hairline", "very low, almost meeting the eyebrows")],
        ),
    ]
    for text, spans in raw:
        i = sid()
        anns = [_ann(i, text, cat, span) for cat, span in spans]
        samples.append(_sample(i, text, anns))

    # --- REGISTER 8: Multi-witness contradiction / discrepancy ---
    raw = [
        (
            "First witness said the hair was brown; second witness insisted it was black. Both agreed on the heavy stubble.",
            [("facial_hair", "heavy stubble")],
        ),
        (
            "Witness A described round eyes; witness B said they were distinctly almond-shaped.",
            [],  # contradictory — no definitive annotation
        ),
        (
            "Officer's note: two witnesses disagree on beard. One says full beard, the other says clean-shaven. Face shape consistent across both: oval.",
            [("face_shape", "oval")],
        ),
        (
            "Discrepancy flagged: first account says blond hair, second says brown. Both witnesses mention the broad flat nose.",
            [("nose", "broad flat nose")],
        ),
        (
            "Conflicting statements on scar location: one says left cheek scar, another says right. Nose shape agreed as bulbous.",
            [("nose", "bulbous")],
        ),
    ]
    for text, spans in raw:
        i = sid()
        anns = [_ann(i, text, cat, span) for cat, span in spans]
        samples.append(_sample(i, text, anns))

    # --- REGISTER 9: Passive / news-style constructions ---
    raw = [
        (
            "The suspect is described by police as having shoulder-length brown hair and green eyes.",
            [("hair", "shoulder-length brown hair"), ("eyes", "green eyes")],
        ),
        (
            "It was noted during the interview that the individual possessed an unusually prominent brow ridge.",
            [("face_shape", "prominent brow ridge")],
        ),
        (
            "A distinguishing feature was reported to be a birthmark located below the left eye.",
            [("distinguishing_marks", "birthmark located below the left eye")],
        ),
        (
            "The subject's hair was observed to be thinning at the crown, with a noticeably receding hairline at the temples.",
            [("hair", "thinning at the crown"), ("hairline", "receding hairline at the temples")],
        ),
        (
            "His facial features were described as angular: a pointed chin, sharp cheekbones, and a thin, straight nose.",
            [("face_shape", "pointed chin"), ("face_shape", "sharp cheekbones"), ("nose", "thin, straight nose")],
        ),
        (
            "She was noted to have a gap between her two front teeth and noticeably full lips.",
            [("distinguishing_marks", "gap between her two front teeth"), ("lips", "noticeably full lips")],
        ),
        (
            "It was reported that the eyebrows were heavily tattooed and appeared unnaturally dark.",
            [("eyebrows", "heavily tattooed")],
        ),
        (
            "The man's ears were described as cauliflower ears, consistent with a history of contact sport.",
            [("other_observable", "cauliflower ears")],
        ),
        (
            "A permanent frown line between the eyebrows was considered a distinguishing feature.",
            [("distinguishing_marks", "frown line between the eyebrows")],
        ),
        (
            "The jawline was described as particularly sharp, almost knife-like, with a defined square chin.",
            [("face_shape", "square chin")],
        ),
    ]
    for text, spans in raw:
        i = sid()
        anns = [_ann(i, text, cat, span) for cat, span in spans]
        samples.append(_sample(i, text, anns))

    # --- REGISTER 10: Uncertain / hedged observations ---
    raw = [
        (
            "I think his hair was dark — maybe black, maybe very dark brown. I can't be certain about that.",
            [],  # explicitly uncertain — no reliable annotation
        ),
        (
            "Could have been a beard, might have just been stubble — it was dim and I didn't look long.",
            [],
        ),
        (
            "Her eyes — I want to say blue? Or maybe grey? Hard to tell under fluorescent lighting.",
            [],
        ),
        (
            "The nose — it seemed wide, but I might be wrong. The face shape was definitely oval, that I'm sure of.",
            [("face_shape", "oval")],
        ),
        (
            "Possibly a scar above the left eyebrow, but I wouldn't swear to it. What I'm certain of is the red hair.",
            [("hair", "red hair")],
        ),
        (
            "I believe the eyebrows were dark and bushy, but the overall lighting made it difficult to judge.",
            [("eyebrows", "dark and bushy")],
        ),
        (
            "He might have had a moustache — it was brief and from a distance. The hair was definitely grey.",
            [("hair", "grey")],
        ),
        (
            "Her lips seemed thin, but I'm not confident. She definitely had prominent cheekbones though.",
            [("lips", "thin"), ("face_shape", "prominent cheekbones")],
        ),
    ]
    for text, spans in raw:
        i = sid()
        anns = [_ann(i, text, cat, span) for cat, span in spans]
        samples.append(_sample(i, text, anns))

    # --- REGISTER 11: Complex multi-attribute sentences ---
    raw = [
        (
            "The man who entered first had a neatly trimmed goatee, thick eyebrows that nearly met in the middle, and piercing green eyes.",
            [("facial_hair", "neatly trimmed goatee"), ("eyebrows", "thick eyebrows"), ("eyes", "piercing green eyes")],
        ),
        (
            "Her platinum blonde hair was cut in a bob, framing a narrow oval face with high-arched eyebrows and a small straight nose.",
            [("hair", "platinum blonde hair"), ("face_shape", "narrow oval face"), ("eyebrows", "high-arched eyebrows"), ("nose", "small straight nose")],
        ),
        (
            "He sported a horseshoe moustache, had a receding hairline, and his face was marked by several old acne scars along both cheeks.",
            [("facial_hair", "horseshoe moustache"), ("hairline", "receding hairline"), ("distinguishing_marks", "old acne scars along both cheeks")],
        ),
        (
            "Something about his whole face — a strong brow, deep-set dark eyes, a heavy jaw — made him look intimidating.",
            [("face_shape", "strong brow"), ("eyes", "deep-set dark eyes"), ("face_shape", "heavy jaw")],
        ),
        (
            "She had a pixie cut — very short dark hair — wide-set almond eyes, and thin but well-defined lips.",
            [("hair", "very short dark hair"), ("eyes", "wide-set almond eyes"), ("lips", "thin but well-defined lips")],
        ),
        (
            "The suspect had a large mole on the right side of his chin, full dark eyebrows, and a broad nose with flared nostrils.",
            [("distinguishing_marks", "large mole on the right side of his chin"), ("eyebrows", "full dark eyebrows"), ("nose", "broad nose with flared nostrils")],
        ),
        (
            "Her hair was dyed an unnatural platinum, her lips were lined with dark lip liner, and her eyebrows appeared microbladed.",
            [("hair", "dyed an unnatural platinum"), ("lips", "dark lip liner"), ("eyebrows", "microbladed")],
        ),
        (
            "He had a cauliflower left ear, a flat crooked nose from a previous break, and salt-and-pepper stubble.",
            [("other_observable", "cauliflower left ear"), ("nose", "flat crooked nose"), ("facial_hair", "salt-and-pepper stubble")],
        ),
    ]
    for text, spans in raw:
        i = sid()
        anns = [_ann(i, text, cat, span) for cat, span in spans]
        samples.append(_sample(i, text, anns))

    # --- REGISTER 12: Present-tense police bulletin style ---
    raw = [
        (
            "Police are looking for a white male with a shaved head and a full red beard.",
            [("hair", "shaved head"), ("facial_hair", "full red beard")],
        ),
        (
            "The missing person has shoulder-length dark hair, brown eyes, and a small scar near the right temple.",
            [("hair", "shoulder-length dark hair"), ("eyes", "brown eyes"), ("distinguishing_marks", "small scar near the right temple")],
        ),
        (
            "Officers are seeking a woman with a round face, curly auburn hair, and distinctive green eyes.",
            [("face_shape", "round face"), ("hair", "curly auburn hair"), ("eyes", "distinctive green eyes")],
        ),
        (
            "The individual is known to have a large tribal tattoo on the right side of his neck and a thick black beard.",
            [("distinguishing_marks", "large tribal tattoo on the right side of his neck"), ("facial_hair", "thick black beard")],
        ),
        (
            "Be advised: the suspect has a distinctly upturned nose, wide-set eyes, and wears his hair in tight cornrows.",
            [("nose", "distinctly upturned nose"), ("eyes", "wide-set eyes"), ("hair", "tight cornrows")],
        ),
        (
            "The described person has a square jaw, a prominent Adam's apple, and thinning grey hair swept to one side.",
            [("face_shape", "square jaw"), ("hair", "thinning grey hair")],
        ),
    ]
    for text, spans in raw:
        i = sid()
        anns = [_ann(i, text, cat, span) for cat, span in spans]
        samples.append(_sample(i, text, anns))

    # --- REGISTER 13: Additional mixed styles ---
    raw = [
        (
            "According to the barman, she had an angular face — almost like a model — and striking grey eyes with long lashes.",
            [("face_shape", "angular face"), ("eyes", "striking grey eyes")],
        ),
        (
            "The delivery driver told us: 'His forehead was huge, like really wide, and he had this tiny little goatee.'",
            [("hairline", "forehead was huge"), ("facial_hair", "tiny little goatee")],
        ),
        (
            "Reviewing the statement again — the witness says the lips were noticeably uneven, the lower lip much thicker.",
            [("lips", "noticeably uneven")],
        ),
        (
            "Both his ears stuck out like handles, and he had this very neat centre parting in his brown hair.",
            [("other_observable", "ears stuck out"), ("hair", "brown hair")],
        ),
        (
            "Deposition extract: 'I'd know those eyes anywhere — pale, almost colourless, very unsettling.'",
            [("eyes", "pale, almost colourless")],
        ),
        (
            "The neighbour described a young woman with a button nose, dimples on both cheeks, and shoulder-length copper hair.",
            [("nose", "button nose"), ("distinguishing_marks", "dimples on both cheeks"), ("hair", "shoulder-length copper hair")],
        ),
        (
            "Noteworthy: the earlobe on the right had an oversized piercing hole — stretched, like a gauge. Hair was dark and unkempt.",
            [("other_observable", "oversized piercing hole"), ("hair", "dark and unkempt")],
        ),
        (
            "She walked quickly and kept her head down but the cashier got a clear look — wide forehead, thin blonde eyebrows, pointed chin.",
            [("hairline", "wide forehead"), ("eyebrows", "thin blonde eyebrows"), ("face_shape", "pointed chin")],
        ),
        (
            "The elderly shopkeeper remembered the nose most vividly — bulbous at the tip, with enlarged pores.",
            [("nose", "bulbous at the tip")],
        ),
        (
            "A series of small tattoos across the knuckles and a long scar bisecting the right eyebrow.",
            [("distinguishing_marks", "long scar bisecting the right eyebrow")],
        ),
    ]
    for text, spans in raw:
        i = sid()
        anns = [_ann(i, text, cat, span) for cat, span in spans]
        samples.append(_sample(i, text, anns))

    return samples


# ---------------------------------------------------------------------------
# Offset integrity verification pass
# ---------------------------------------------------------------------------

def verify_offsets(samples: List[Dict]) -> List[str]:
    errors = []
    for s in samples:
        text = s["witness_text"]
        for ann in s["annotations"]:
            if ann["certainty"] == "unknown":
                continue
            start = ann["start_char"]
            end = ann["end_char"]
            span = ann["source_span"]
            actual = text[start:end]
            if actual != span:
                errors.append(
                    f"[OFFSET MISMATCH] {s['sample_id']} | cat={ann['category']} | "
                    f"expected={span!r} | got={actual!r} | [{start}:{end}]"
                )
    return errors


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "datasets", "nlp")
    out_path = os.path.join(out_dir, "out_of_frame_eval.jsonl")
    root_out = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "datasets", "nlp"
    )

    samples = build_samples()

    print(f"Generated {len(samples)} out-of-frame evaluation samples.")

    # Verify all character offsets
    errors = verify_offsets(samples)
    if errors:
        print("\n[ERROR] Offset verification failures detected:")
        for e in errors:
            print(" ", e)
        sys.exit(1)
    else:
        print("Offset verification: ALL PASSED")

    # Count annotations
    total_anns = sum(
        len([a for a in s["annotations"] if a["certainty"] != "unknown"])
        for s in samples
    )
    print(f"Total explicit/uncertain annotations: {total_anns}")

    # Save to phantom-ai-ml/datasets/nlp/
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Saved to: {out_path}")

    # Also save to project root datasets/nlp/ (symlink-compatible copy)
    abs_root_out = os.path.abspath(root_out)
    if os.path.isdir(abs_root_out):
        root_out_path = os.path.join(abs_root_out, "out_of_frame_eval.jsonl")
        with open(root_out_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Also saved to: {root_out_path}")
    else:
        print(f"Root dataset dir not found, skipping root copy: {abs_root_out}")

    print(f"\nSummary: {len(samples)} samples, {total_anns} annotated spans")
    print("Corpus is NOT to be used for training or validation. Evaluation only.")


if __name__ == "__main__":
    main()
