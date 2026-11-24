import json
import os
import random
import re
from typing import List, Dict, Any, Tuple
from schemas.attributes import AttributeCategory, EvidenceState
from schemas.dataset import (
    DatasetSample,
    AttributeAnnotation,
    SampleMetadata,
    ForensicDataset,
)
from data.validator import DatasetValidator
from data.dataset_loader import DatasetLoader


# Rich vocabulary per category with realistic forensic phrasing
VOCABULARY = {
    AttributeCategory.FACE_SHAPE: [
        ("oval face", "oval"),
        ("round face", "round"),
        ("square jaw", "square"),
        ("sharp jawline", "sharp"),
        ("chiseled chin", "chiseled"),
        ("angular face", "angular"),
        ("heart-shaped face", "heart-shaped"),
        ("oblong face", "oblong"),
        ("pronounced cleft chin", "cleft chin"),
        ("prominent cheekbones", "prominent cheekbones"),
        ("wide flat face", "wide flat"),
        ("narrow pointed chin", "narrow pointed"),
        ("heavy square jaw", "heavy square"),
        ("broad face", "broad"),
    ],
    AttributeCategory.HAIR: [
        ("short dark curly hair", "short dark curly"),
        ("straight blonde hair", "straight blonde"),
        ("shoulder-length wavy hair", "shoulder-length wavy"),
        ("buzzcut", "buzzcut"),
        ("bald head", "bald"),
        ("jet black hair", "jet black"),
        ("thick brown hair", "thick brown"),
        ("shaved head", "shaved"),
        ("messy dark hair", "messy dark"),
        ("curly red hair", "curly red"),
        ("slicked-back black hair", "slicked-back black"),
        ("thinning grey hair", "thinning grey"),
        ("tight black afro", "afro"),
        ("long straight brown hair", "long straight brown"),
        ("cropped blond hair", "cropped blond"),
        ("wavy dark brown hair", "wavy dark brown"),
    ],
    AttributeCategory.HAIRLINE: [
        ("receding hairline", "receding"),
        ("widow's peak", "widow's peak"),
        ("deep receding temples", "deep receding"),
        ("straight low hairline", "straight low"),
        ("high forehead hairline", "high forehead"),
        ("prominent widow's peak", "prominent widow's peak"),
        ("uneven hairline", "uneven"),
        ("noticeably high hairline", "noticeably high"),
    ],
    AttributeCategory.EYES: [
        ("dark brown eyes", "dark brown"),
        ("piercing blue eyes", "piercing blue"),
        ("hazel eyes", "hazel"),
        ("deep-set green eyes", "deep-set green"),
        ("narrow squinting eyes", "narrow squinting"),
        ("light blue eyes", "light blue"),
        ("wide dark eyes", "wide dark"),
        ("almond-shaped brown eyes", "almond-shaped brown"),
        ("pale grey eyes", "pale grey"),
        ("intense black eyes", "intense black"),
        ("hooded green eyes", "hooded green"),
    ],
    AttributeCategory.EYEBROWS: [
        ("thick bushy eyebrows", "thick bushy"),
        ("thin plucked eyebrows", "thin plucked"),
        ("arched eyebrows", "arched"),
        ("faint unibrow", "unibrow"),
        ("straight dark eyebrows", "straight dark"),
        ("prominent brow ridge", "prominent brow"),
        ("sparse light eyebrows", "sparse light"),
        ("heavy arched eyebrows", "heavy arched"),
        ("bushy black eyebrows", "bushy black"),
    ],
    AttributeCategory.NOSE: [
        ("crooked nose", "crooked"),
        ("large hooked nose", "large hooked"),
        ("wide flat nose", "wide flat"),
        ("sharp pointed nose", "sharp pointed"),
        ("aquiline nose", "aquiline"),
        ("broken nose", "broken"),
        ("small button nose", "button"),
        ("broad nostrils", "broad"),
        ("straight narrow nose", "straight narrow"),
        ("prominent bridge nose", "prominent bridge"),
    ],
    AttributeCategory.LIPS: [
        ("thin lips", "thin"),
        ("full lips", "full"),
        ("dry chapped lips", "dry chapped"),
        ("wide thin mouth", "wide thin"),
        ("pronounced cupid's bow", "pronounced cupid's bow"),
        ("thick lower lip", "thick lower lip"),
        ("pouty lips", "pouty"),
        ("narrow pale lips", "narrow pale"),
    ],
    AttributeCategory.FACIAL_HAIR: [
        ("trimmed goatee", "trimmed goatee"),
        ("full thick beard", "full thick beard"),
        ("bushy mustache", "bushy mustache"),
        ("short stubble", "short stubble"),
        ("five o'clock shadow", "five o'clock shadow"),
        ("clean-shaven", "clean-shaven"),
        ("thin pencil mustache", "thin pencil mustache"),
        ("patchy brown beard", "patchy beard"),
        ("neatly trimmed beard", "neatly trimmed beard"),
        ("sideburns down to jaw", "sideburns"),
        ("heavy dark stubble", "heavy dark stubble"),
    ],
    AttributeCategory.DISTINGUISHING_MARKS: [
        ("jagged scar across cheek", "jagged scar"),
        ("dark mole above lip", "dark mole"),
        ("faint birthmark on neck", "birthmark"),
        ("small teardrop tattoo under eye", "teardrop tattoo"),
        ("prominent eyebrow scar", "eyebrow scar"),
        ("burn mark on jawline", "burn mark"),
        ("noticeable facial mole", "facial mole"),
        ("scar through right eyebrow", "scar through eyebrow"),
    ],
    AttributeCategory.OTHER_OBSERVABLE: [
        ("protruding ears", "protruding ears"),
        ("wire-rim glasses", "wire-rim glasses"),
        ("silver hoop earring", "silver hoop earring"),
        ("deep forehead wrinkles", "deep forehead wrinkles"),
        ("gold tooth in front", "gold tooth"),
        ("cauliflower ear", "cauliflower ear"),
        ("black-framed glasses", "black-framed glasses"),
        ("small gold stud earring", "gold stud earring"),
    ],
}


class SentenceBuilder:
    """
    Constructs natural text while tracking character offsets mathematically.
    """
    def __init__(self):
        self.text = ""
        self.spans = []  # List of (category, value, span_text, start_char, end_char, certainty, cues)

    def add_text(self, text: str):
        self.text += text

    def add_span(self, category: AttributeCategory, value: str, span_text: str, certainty: EvidenceState, cues: List[str] = None):
        start = len(self.text)
        self.text += span_text
        end = len(self.text)
        assert self.text[start:end] == span_text
        self.spans.append((category, value, span_text, start, end, certainty, cues or []))


def generate_expanded_dataset(target_count: int = 600, seed: int = 42) -> ForensicDataset:
    rng = random.Random(seed)
    samples: List[DatasetSample] = []
    seen_texts = set()

    meta = SampleMetadata(
        dataset_version="2.0.0",
        source="Phantom AI Synthetic Forensic Witness Expansion Suite",
        license="CC-BY-4.0-Synthetic",
        is_synthetic=True,
        notes="Synthetic training and benchmark samples for custom BiLSTM-CRF sequence labeling.",
    )

    all_categories = list(AttributeCategory)

    def sample_item(cat):
        return rng.choice(VOCABULARY[cat])

    sample_idx = 1

    # Frame 1: Narrative multi-attribute description (3-4 features)
    openers = [
        "The witness reported that the suspect had ",
        "According to the caller, the person was seen with ",
        "The individual was described as having ",
        "I clearly noticed that the subject possessed ",
        "The complainant observed a person who had ",
        "During the incident, the perpetrator displayed ",
        "The report indicates the suspect had ",
        "The bystander described someone with ",
    ]
    connectors_1 = [", alongside ", ", accompanied by ", ", together with ", ", and also "]
    connectors_2 = [", with ", ", and ", ", sporting ", ", plus "]

    for opener in openers:
        for _ in range(16):
            if len(samples) >= target_count:
                break
            b = SentenceBuilder()
            b.add_text(opener)

            c1, c2, c3 = rng.sample(all_categories, 3)
            s1, v1 = sample_item(c1)
            b.add_span(c1, v1, s1, EvidenceState.EXPLICIT)
            b.add_text(rng.choice(connectors_1))

            s2, v2 = sample_item(c2)
            b.add_span(c2, v2, s2, EvidenceState.EXPLICIT)
            b.add_text(rng.choice(connectors_2))

            s3, v3 = sample_item(c3)
            b.add_span(c3, v3, s3, EvidenceState.EXPLICIT)
            b.add_text(".")

            text = b.text.strip()
            if text not in seen_texts:
                seen_texts.add(text)
                samples.append(_create_sample(f"SYNTH-EXP-{sample_idx:04d}", text, b.spans, all_categories, meta))
                sample_idx += 1

    # Frame 2: Police dispatch / fragmented incident log
    police_intros = [
        "Suspect description: ",
        "Officer field notes: ",
        "BOLO profile: ",
        "Dispatch summary: ",
        "Witness statement fragment: ",
        "Case file excerpt: ",
    ]
    for intro in police_intros:
        for _ in range(16):
            if len(samples) >= target_count:
                break
            b = SentenceBuilder()
            b.add_text(intro)

            c1, c2, c3 = rng.sample(all_categories, 3)
            s1, v1 = sample_item(c1)
            b.add_span(c1, v1, s1.capitalize(), EvidenceState.EXPLICIT)
            b.add_text(". ")

            s2, v2 = sample_item(c2)
            b.add_span(c2, v2, s2.capitalize(), EvidenceState.EXPLICIT)
            b.add_text(". ")

            s3, v3 = sample_item(c3)
            b.add_span(c3, v3, s3.capitalize(), EvidenceState.EXPLICIT)
            b.add_text(".")

            text = b.text.strip()
            if text not in seen_texts:
                seen_texts.add(text)
                samples.append(_create_sample(f"SYNTH-EXP-{sample_idx:04d}", text, b.spans, all_categories, meta))
                sample_idx += 1

    # Frame 3: Uncertain / Hedging testimony (2-3 features, 1-2 uncertain)
    hedging_openers = [
        ("I think he maybe had ", ["i think", "maybe"]),
        ("It appeared to be someone with ", ["appeared to be"]),
        ("I am not completely certain, but I saw ", ["not completely certain"]),
        ("The witness stated it seemed like ", ["seemed like"]),
        ("If I recall correctly, there was ", ["if i recall"]),
        ("Could be someone having ", ["could be"]),
        ("Perhaps the suspect had ", ["perhaps"]),
        ("Hard to tell for sure, but possibly ", ["hard to tell", "possibly"]),
    ]
    for h_open, cues in hedging_openers:
        for _ in range(14):
            if len(samples) >= target_count:
                break
            b = SentenceBuilder()
            b.add_text(h_open)

            c1, c2 = rng.sample(all_categories, 2)
            s1, v1 = sample_item(c1)
            b.add_span(c1, v1, s1, EvidenceState.UNCERTAIN, cues)

            b.add_text(", though they also definitely noticed ")
            s2, v2 = sample_item(c2)
            b.add_span(c2, v2, s2, EvidenceState.EXPLICIT)
            b.add_text(".")

            text = b.text.strip()
            if text not in seen_texts:
                seen_texts.add(text)
                samples.append(_create_sample(f"SYNTH-EXP-{sample_idx:04d}", text, b.spans, all_categories, meta))
                sample_idx += 1

    # Frame 4: Negations & Absence of features (e.g. clean-shaven, no marks)
    negation_templates = [
        ("The person was completely clean-shaven with no facial hair, but had ", AttributeCategory.FACIAL_HAIR, "clean-shaven", "clean-shaven"),
        ("The subject had no facial hair whatsoever, displaying ", AttributeCategory.FACIAL_HAIR, "no facial hair", "no facial hair"),
        ("No beard or mustache was visible, but the suspect had ", AttributeCategory.FACIAL_HAIR, "no beard or mustache", "no beard or mustache"),
    ]
    for neg_intro, neg_cat, neg_span, neg_val in negation_templates:
        for _ in range(14):
            if len(samples) >= target_count:
                break
            b = SentenceBuilder()
            b.add_text(neg_intro)

            other_cats = [c for c in all_categories if c != neg_cat]
            c1, c2 = rng.sample(other_cats, 2)

            s1, v1 = sample_item(c1)
            b.add_span(c1, v1, s1, EvidenceState.EXPLICIT)
            b.add_text(" and ")

            s2, v2 = sample_item(c2)
            b.add_span(c2, v2, s2, EvidenceState.EXPLICIT)
            b.add_text(".")

            text = b.text.strip()
            if text not in seen_texts:
                seen_texts.add(text)
                samples.append(_create_sample(f"SYNTH-EXP-{sample_idx:04d}", text, b.spans, all_categories, meta))
                sample_idx += 1

    # Frame 5: Contradictory statements (conflicting descriptions)
    contra_pairs = [
        (AttributeCategory.HAIR, "bald head", "bald", "shoulder-length wavy hair", "shoulder-length wavy"),
        (AttributeCategory.FACIAL_HAIR, "clean-shaven", "clean-shaven", "full thick beard", "full thick beard"),
        (AttributeCategory.EYES, "dark brown eyes", "dark brown", "piercing blue eyes", "piercing blue"),
        (AttributeCategory.HAIR, "buzzcut", "buzzcut", "straight blonde hair", "straight blonde"),
        (AttributeCategory.FACE_SHAPE, "round face", "round", "sharp jawline", "sharp"),
    ]
    contra_frames = [
        ("The first statement claimed {span1}, whereas the second statement insisted on {span2}.", "{span1}", "{span2}"),
        ("Witness A reported {span1}, but Witness B claimed {span2}.", "{span1}", "{span2}"),
        ("Initial notes indicated {span1}, though later interview mentioned {span2}.", "{span1}", "{span2}"),
    ]
    for frame, placeholder1, placeholder2 in contra_frames:
        for cat, span1, val1, span2, val2 in contra_pairs:
            if len(samples) >= target_count:
                break
            b = SentenceBuilder()
            prefix = frame.split(placeholder1)[0]
            middle = frame.split(placeholder1)[1].split(placeholder2)[0]
            suffix = frame.split(placeholder2)[1]

            b.add_text(prefix)
            b.add_span(cat, val1, span1, EvidenceState.CONTRADICTORY)
            b.add_text(middle)
            b.add_span(cat, val2, span2, EvidenceState.CONTRADICTORY)
            b.add_text(suffix)

            text = b.text.strip()
            if text not in seen_texts:
                seen_texts.add(text)
                samples.append(_create_sample(f"SYNTH-EXP-{sample_idx:04d}", text, b.spans, all_categories, meta))
                sample_idx += 1

    # Frame 6: Single-attribute sparse statements (all other features UNKNOWN)
    sparse_openers = [
        "The sole noticeable feature was ",
        "I could only get a clear look at his ",
        "The only distinguishing element recalled was ",
        "All the witness remembered was ",
        "Lighting was poor, so the observer only registered ",
        "From that distance, the only visible trait was ",
    ]
    for opener in sparse_openers:
        for cat in all_categories:
            if len(samples) >= target_count:
                break
            b = SentenceBuilder()
            b.add_text(opener)
            s, v = sample_item(cat)
            b.add_span(cat, v, s, EvidenceState.EXPLICIT)
            b.add_text(".")

            text = b.text.strip()
            if text not in seen_texts:
                seen_texts.add(text)
                samples.append(_create_sample(f"SYNTH-EXP-{sample_idx:04d}", text, b.spans, all_categories, meta))
                sample_idx += 1

    # Frame 7: Dense descriptive paragraphs (4-5 features across multiple sentences)
    dense_intros = [
        "The suspect stood near the cashier. ",
        "I got a direct front view of the perpetrator. ",
        "The driver exited the vehicle immediately after the collision. ",
        "The security guard approached the individual in the lobby. ",
        "Surveillance footage showed the person in clear profile. ",
    ]
    for intro in dense_intros:
        for _ in range(16):
            if len(samples) >= target_count:
                break
            b = SentenceBuilder()
            b.add_text(intro)

            c1, c2, c3, c4 = rng.sample(all_categories, 4)
            s1, v1 = sample_item(c1)
            b.add_text("He had ")
            b.add_span(c1, v1, s1, EvidenceState.EXPLICIT)
            b.add_text(" and ")

            s2, v2 = sample_item(c2)
            b.add_span(c2, v2, s2, EvidenceState.EXPLICIT)
            b.add_text(". In addition, there was ")

            s3, v3 = sample_item(c3)
            b.add_span(c3, v3, s3, EvidenceState.EXPLICIT)
            b.add_text(", as well as ")

            s4, v4 = sample_item(c4)
            b.add_span(c4, v4, s4, EvidenceState.EXPLICIT)
            b.add_text(".")

            text = b.text.strip()
            if text not in seen_texts:
                seen_texts.add(text)
                samples.append(_create_sample(f"SYNTH-EXP-{sample_idx:04d}", text, b.spans, all_categories, meta))
                sample_idx += 1

    # Frame 8: Conversational witness utterances (hedged, spoken, disjointed)
    spoken_openers = [
        "Look, he had this ",
        "Well, I distinctly remember a ",
        "Honestly, what stood out was a ",
        "To be honest, I mostly saw ",
        "Yeah, the guy definitely had ",
    ]
    spoken_connectors = [
        ", and maybe some kind of ",
        ", but wait, he also had ",
        ", plus I'm pretty sure he had ",
        ", along with ",
    ]
    for s_open in spoken_openers:
        for _ in range(16):
            if len(samples) >= target_count:
                break
            b = SentenceBuilder()
            b.add_text(s_open)

            c1, c2 = rng.sample(all_categories, 2)
            s1, v1 = sample_item(c1)
            b.add_span(c1, v1, s1, EvidenceState.EXPLICIT)

            b.add_text(rng.choice(spoken_connectors))
            s2, v2 = sample_item(c2)
            b.add_span(c2, v2, s2, EvidenceState.UNCERTAIN, ["pretty sure" if "pretty sure" in b.text else "maybe"])
            b.add_text(".")

            text = b.text.strip()
            if text not in seen_texts:
                seen_texts.add(text)
                samples.append(_create_sample(f"SYNTH-EXP-{sample_idx:04d}", text, b.spans, all_categories, meta))
                sample_idx += 1

    return ForensicDataset(
        name="Phantom AI Synthetic Forensic Witness Expansion Benchmark",
        version="2.0.0",
        license="CC-BY-4.0-Synthetic",
        source="Phantom AI Synthetic Witness Expansion Suite",
        is_synthetic=True,
        samples=samples,
    )


def _create_sample(sample_id: str, text: str, spans: List[Tuple], all_categories: List[AttributeCategory], meta: SampleMetadata) -> DatasetSample:
    anns: List[AttributeAnnotation] = []
    covered_cats = set()

    for idx, (cat, val, span_text, start, end, cert, cues) in enumerate(spans, start=1):
        assert text[start:end] == span_text, (
            f"Offset mismatch in {sample_id} ({cat}): Text[{start}:{end}] is '{text[start:end]}', expected '{span_text}'"
        )
        anns.append(
            AttributeAnnotation(
                annotation_id=f"{sample_id}-ANN-{idx}",
                category=cat,
                attribute_value=val,
                source_span=span_text,
                start_char=start,
                end_char=end,
                certainty=cert,
                annotator_id="SYNTHETIC-EXPANSION-PIPELINE",
                uncertainty_cues=cues,
                notes=f"Synthetic {cert.value} annotation generated for custom BiLSTM-CRF training",
            )
        )
        covered_cats.add(cat)

    # All unmentioned categories are preserved as explicit UNKNOWN
    unmentioned = [c for c in all_categories if c not in covered_cats]
    for u_idx, unk_cat in enumerate(unmentioned, start=1):
        anns.append(
            AttributeAnnotation(
                annotation_id=f"{sample_id}-UNK-{u_idx}",
                category=unk_cat,
                attribute_value=None,
                source_span=None,
                start_char=None,
                end_char=None,
                certainty=EvidenceState.UNKNOWN,
                annotator_id="SYNTHETIC-EXPANSION-PIPELINE",
                notes="Feature was not observed or mentioned - strictly preserved as UNKNOWN",
            )
        )

    return DatasetSample(
        sample_id=sample_id,
        witness_text=text,
        annotations=anns,
        metadata=meta,
    )


if __name__ == "__main__":
    dataset = generate_expanded_dataset(target_count=550, seed=42)
    loader = DatasetLoader()
    json_path = os.path.join("datasets", "expanded_witness_dataset.json")
    jsonl_path = os.path.join("datasets", "expanded_witness_dataset.jsonl")

    loader.save_json(dataset, json_path)
    loader.save_jsonl(dataset.samples, jsonl_path)

    # Also save to phantom-ai-ml/datasets/
    ml_json_path = os.path.join("phantom-ai-ml", "datasets", "expanded_witness_dataset.json")
    ml_jsonl_path = os.path.join("phantom-ai-ml", "datasets", "expanded_witness_dataset.jsonl")
    loader.save_json(dataset, ml_json_path)
    loader.save_jsonl(dataset.samples, ml_jsonl_path)

    validator = DatasetValidator()
    summary = validator.validate_dataset(dataset.samples)
    print(f"Generated {len(dataset.samples)} expanded synthetic samples.")
    print(f"Validation Valid: {summary.is_valid}, Errors: {len(summary.errors)}, Warnings: {len(summary.warnings)}")
    print(f"Category counts: {summary.statistics['category_distribution']}")
    print(f"Certainty counts: {summary.statistics['certainty_distribution']}")

