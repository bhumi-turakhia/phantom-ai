from typing import List
from schemas.attributes import AttributeCategory, EvidenceState
from schemas.dataset import (
    DatasetSample,
    AttributeAnnotation,
    SampleMetadata,
    ForensicDataset,
)


def get_synthetic_dataset() -> ForensicDataset:
    """
    Builds a benchmark collection of 24 diverse synthetic witness description samples.
    All samples are explicitly labeled as synthetic with strict character offset fidelity.
    """
    common_metadata = SampleMetadata(
        dataset_version="1.0.0",
        source="Phantom AI Synthetic Witness Benchmarking Suite",
        license="CC-BY-4.0-Synthetic",
        is_synthetic=True,
        notes="Illustrative synthetic samples generated strictly for pipeline testing. Not real witness data.",
    )

    raw_samples = [
        # SYNTH-001: Explicit multi-attribute statement
        {
            "id": "SYNTH-001",
            "text": "The suspect had an oval face, short dark curly hair, and brown eyes.",
            "annotations": [
                ("ANN-001-1", AttributeCategory.FACE_SHAPE, "oval", "oval face", 19, 28, EvidenceState.EXPLICIT, []),
                ("ANN-001-2", AttributeCategory.HAIR, "short dark curly", "short dark curly hair", 30, 51, EvidenceState.EXPLICIT, []),
                ("ANN-001-3", AttributeCategory.EYES, "brown", "brown eyes", 57, 67, EvidenceState.EXPLICIT, []),
            ],
            "unknowns": [
                AttributeCategory.HAIRLINE,
                AttributeCategory.EYEBROWS,
                AttributeCategory.NOSE,
                AttributeCategory.LIPS,
                AttributeCategory.FACIAL_HAIR,
                AttributeCategory.DISTINGUISHING_MARKS,
                AttributeCategory.OTHER_OBSERVABLE,
            ],
        },
        # SYNTH-002: Uncertain description with hedging cues
        {
            "id": "SYNTH-002",
            "text": "I think he maybe had a trimmed beard, but I am not certain about his eye color.",
            "annotations": [
                ("ANN-002-1", AttributeCategory.FACIAL_HAIR, "trimmed beard", "trimmed beard", 23, 36, EvidenceState.UNCERTAIN, ["i think", "maybe"]),
                ("ANN-002-2", AttributeCategory.EYES, "unspecified", "eye color", 69, 78, EvidenceState.UNCERTAIN, ["not certain"]),
            ],
            "unknowns": [
                AttributeCategory.FACE_SHAPE,
                AttributeCategory.HAIR,
                AttributeCategory.HAIRLINE,
                AttributeCategory.EYEBROWS,
                AttributeCategory.NOSE,
                AttributeCategory.LIPS,
                AttributeCategory.DISTINGUISHING_MARKS,
                AttributeCategory.OTHER_OBSERVABLE,
            ],
        },
        # SYNTH-003: Contradictory witness statements
        {
            "id": "SYNTH-003",
            "text": "The witness initially reported a completely bald head, but later claimed shoulder-length wavy hair was visible.",
            "annotations": [
                ("ANN-003-1", AttributeCategory.HAIR, "bald", "bald head", 44, 53, EvidenceState.CONTRADICTORY, []),
                ("ANN-003-2", AttributeCategory.HAIR, "shoulder-length wavy", "shoulder-length wavy hair", 73, 98, EvidenceState.CONTRADICTORY, []),
            ],
            "unknowns": [
                AttributeCategory.FACE_SHAPE,
                AttributeCategory.HAIRLINE,
                AttributeCategory.EYES,
                AttributeCategory.EYEBROWS,
                AttributeCategory.NOSE,
                AttributeCategory.LIPS,
                AttributeCategory.FACIAL_HAIR,
                AttributeCategory.DISTINGUISHING_MARKS,
                AttributeCategory.OTHER_OBSERVABLE,
            ],
        },
        # SYNTH-004: Distinguishing mark in passive voice
        {
            "id": "SYNTH-004",
            "text": "A prominent jagged scar was observed across his left cheek.",
            "annotations": [
                ("ANN-004-1", AttributeCategory.DISTINGUISHING_MARKS, "jagged scar", "prominent jagged scar", 2, 23, EvidenceState.EXPLICIT, []),
            ],
            "unknowns": [
                AttributeCategory.FACE_SHAPE,
                AttributeCategory.HAIR,
                AttributeCategory.HAIRLINE,
                AttributeCategory.EYES,
                AttributeCategory.EYEBROWS,
                AttributeCategory.NOSE,
                AttributeCategory.LIPS,
                AttributeCategory.FACIAL_HAIR,
                AttributeCategory.OTHER_OBSERVABLE,
            ],
        },
        # SYNTH-005: Hairline and eyebrows
        {
            "id": "SYNTH-005",
            "text": "The individual possessed a receding hairline with thick arched eyebrows.",
            "annotations": [
                ("ANN-005-1", AttributeCategory.HAIRLINE, "receding", "receding hairline", 27, 44, EvidenceState.EXPLICIT, []),
                ("ANN-005-2", AttributeCategory.EYEBROWS, "thick arched", "thick arched eyebrows", 50, 71, EvidenceState.EXPLICIT, []),
            ],
            "unknowns": [
                AttributeCategory.FACE_SHAPE,
                AttributeCategory.HAIR,
                AttributeCategory.EYES,
                AttributeCategory.NOSE,
                AttributeCategory.LIPS,
                AttributeCategory.FACIAL_HAIR,
                AttributeCategory.DISTINGUISHING_MARKS,
                AttributeCategory.OTHER_OBSERVABLE,
            ],
        },
        # SYNTH-006: Nose and lips
        {
            "id": "SYNTH-006",
            "text": "His nose was distinctly crooked, and he had unusually thin lips.",
            "annotations": [
                ("ANN-006-1", AttributeCategory.NOSE, "crooked", "crooked", 24, 31, EvidenceState.EXPLICIT, []),
                ("ANN-006-2", AttributeCategory.LIPS, "thin", "thin lips", 54, 63, EvidenceState.EXPLICIT, []),
            ],
            "unknowns": [
                AttributeCategory.FACE_SHAPE,
                AttributeCategory.HAIR,
                AttributeCategory.HAIRLINE,
                AttributeCategory.EYES,
                AttributeCategory.EYEBROWS,
                AttributeCategory.FACIAL_HAIR,
                AttributeCategory.DISTINGUISHING_MARKS,
                AttributeCategory.OTHER_OBSERVABLE,
            ],
        },
        # SYNTH-007: Other observable facial features
        {
            "id": "SYNTH-007",
            "text": "She had protruding ears with a silver hoop earring in the right ear.",
            "annotations": [
                ("ANN-007-1", AttributeCategory.OTHER_OBSERVABLE, "protruding ears", "protruding ears", 8, 23, EvidenceState.EXPLICIT, []),
                ("ANN-007-2", AttributeCategory.OTHER_OBSERVABLE, "silver hoop earring", "silver hoop earring", 31, 50, EvidenceState.EXPLICIT, []),
            ],
            "unknowns": [
                AttributeCategory.FACE_SHAPE,
                AttributeCategory.HAIR,
                AttributeCategory.HAIRLINE,
                AttributeCategory.EYES,
                AttributeCategory.EYEBROWS,
                AttributeCategory.NOSE,
                AttributeCategory.LIPS,
                AttributeCategory.FACIAL_HAIR,
                AttributeCategory.DISTINGUISHING_MARKS,
            ],
        },
        # SYNTH-008: Uncertain face shape and explicit facial hair
        {
            "id": "SYNTH-008",
            "text": "The face appeared somewhat angular, and he definitely had a bushy mustache.",
            "annotations": [
                ("ANN-008-1", AttributeCategory.FACE_SHAPE, "angular", "angular", 27, 34, EvidenceState.UNCERTAIN, ["appeared somewhat"]),
                ("ANN-008-2", AttributeCategory.FACIAL_HAIR, "bushy mustache", "bushy mustache", 60, 74, EvidenceState.EXPLICIT, []),
            ],
            "unknowns": [
                AttributeCategory.HAIR,
                AttributeCategory.HAIRLINE,
                AttributeCategory.EYES,
                AttributeCategory.EYEBROWS,
                AttributeCategory.NOSE,
                AttributeCategory.LIPS,
                AttributeCategory.DISTINGUISHING_MARKS,
                AttributeCategory.OTHER_OBSERVABLE,
            ],
        },
        # SYNTH-009: Negative statement (clean-shaven)
        {
            "id": "SYNTH-009",
            "text": "The suspect was completely clean-shaven with no facial hair whatsoever.",
            "annotations": [
                ("ANN-009-1", AttributeCategory.FACIAL_HAIR, "clean-shaven", "clean-shaven", 27, 39, EvidenceState.EXPLICIT, []),
            ],
            "unknowns": [
                AttributeCategory.FACE_SHAPE,
                AttributeCategory.HAIR,
                AttributeCategory.HAIRLINE,
                AttributeCategory.EYES,
                AttributeCategory.EYEBROWS,
                AttributeCategory.NOSE,
                AttributeCategory.LIPS,
                AttributeCategory.DISTINGUISHING_MARKS,
                AttributeCategory.OTHER_OBSERVABLE,
            ],
        },
        # SYNTH-010: Fragmented testimony cadence
        {
            "id": "SYNTH-010",
            "text": "Tall subject. Broad hooked nose. Hazel eyes, very piercing.",
            "annotations": [
                ("ANN-010-1", AttributeCategory.NOSE, "broad hooked", "Broad hooked nose", 14, 31, EvidenceState.EXPLICIT, []),
                ("ANN-010-2", AttributeCategory.EYES, "hazel", "Hazel eyes", 33, 43, EvidenceState.EXPLICIT, []),
            ],
            "unknowns": [
                AttributeCategory.FACE_SHAPE,
                AttributeCategory.HAIR,
                AttributeCategory.HAIRLINE,
                AttributeCategory.EYEBROWS,
                AttributeCategory.LIPS,
                AttributeCategory.FACIAL_HAIR,
                AttributeCategory.DISTINGUISHING_MARKS,
                AttributeCategory.OTHER_OBSERVABLE,
            ],
        },
        # SYNTH-011: Contradictory facial hair assertion
        {
            "id": "SYNTH-011",
            "text": "First the witness stated clean shaven, then mentioned a heavy stubble beard.",
            "annotations": [
                ("ANN-011-1", AttributeCategory.FACIAL_HAIR, "clean shaven", "clean shaven", 25, 37, EvidenceState.CONTRADICTORY, []),
                ("ANN-011-2", AttributeCategory.FACIAL_HAIR, "heavy stubble beard", "heavy stubble beard", 56, 75, EvidenceState.CONTRADICTORY, []),
            ],
            "unknowns": [
                AttributeCategory.FACE_SHAPE,
                AttributeCategory.HAIR,
                AttributeCategory.HAIRLINE,
                AttributeCategory.EYES,
                AttributeCategory.EYEBROWS,
                AttributeCategory.NOSE,
                AttributeCategory.LIPS,
                AttributeCategory.DISTINGUISHING_MARKS,
                AttributeCategory.OTHER_OBSERVABLE,
            ],
        },
        # SYNTH-012: Multiple uncertain attributes
        {
            "id": "SYNTH-012",
            "text": "Could be blond hair, perhaps blue eyes, but the lighting was extremely dim.",
            "annotations": [
                ("ANN-012-1", AttributeCategory.HAIR, "blond", "blond hair", 9, 19, EvidenceState.UNCERTAIN, ["could be"]),
                ("ANN-012-2", AttributeCategory.EYES, "blue", "blue eyes", 29, 38, EvidenceState.UNCERTAIN, ["perhaps"]),
            ],
            "unknowns": [
                AttributeCategory.FACE_SHAPE,
                AttributeCategory.HAIRLINE,
                AttributeCategory.EYEBROWS,
                AttributeCategory.NOSE,
                AttributeCategory.LIPS,
                AttributeCategory.FACIAL_HAIR,
                AttributeCategory.DISTINGUISHING_MARKS,
                AttributeCategory.OTHER_OBSERVABLE,
            ],
        },
        # SYNTH-013: Square jaw and widow's peak
        {
            "id": "SYNTH-013",
            "text": "He had a sharp square jaw and an obvious widow's peak hairline.",
            "annotations": [
                ("ANN-013-1", AttributeCategory.FACE_SHAPE, "square", "sharp square jaw", 9, 25, EvidenceState.EXPLICIT, []),
                ("ANN-013-2", AttributeCategory.HAIRLINE, "widow's peak", "widow's peak hairline", 41, 62, EvidenceState.EXPLICIT, []),
            ],
            "unknowns": [
                AttributeCategory.HAIR,
                AttributeCategory.EYES,
                AttributeCategory.EYEBROWS,
                AttributeCategory.NOSE,
                AttributeCategory.LIPS,
                AttributeCategory.FACIAL_HAIR,
                AttributeCategory.DISTINGUISHING_MARKS,
                AttributeCategory.OTHER_OBSERVABLE,
            ],
        },
        # SYNTH-014: Distinguishing marks
        {
            "id": "SYNTH-014",
            "text": "There was a dark mole above the upper lip and a faint birthmark on the neck.",
            "annotations": [
                ("ANN-014-1", AttributeCategory.DISTINGUISHING_MARKS, "dark mole", "dark mole", 12, 21, EvidenceState.EXPLICIT, []),
                ("ANN-014-2", AttributeCategory.DISTINGUISHING_MARKS, "faint birthmark", "faint birthmark", 48, 63, EvidenceState.EXPLICIT, []),
            ],
            "unknowns": [
                AttributeCategory.FACE_SHAPE,
                AttributeCategory.HAIR,
                AttributeCategory.HAIRLINE,
                AttributeCategory.EYES,
                AttributeCategory.EYEBROWS,
                AttributeCategory.NOSE,
                AttributeCategory.LIPS,
                AttributeCategory.FACIAL_HAIR,
                AttributeCategory.OTHER_OBSERVABLE,
            ],
        },
        # SYNTH-015: Dense multi-attribute description
        {
            "id": "SYNTH-015",
            "text": "Subject had a round face, long straight blonde hair, full lips, and a faint unibrow.",
            "annotations": [
                ("ANN-015-1", AttributeCategory.FACE_SHAPE, "round", "round face", 14, 24, EvidenceState.EXPLICIT, []),
                ("ANN-015-2", AttributeCategory.HAIR, "long straight blonde", "long straight blonde hair", 26, 51, EvidenceState.EXPLICIT, []),
                ("ANN-015-3", AttributeCategory.LIPS, "full", "full lips", 53, 62, EvidenceState.EXPLICIT, []),
                ("ANN-015-4", AttributeCategory.EYEBROWS, "unibrow", "faint unibrow", 70, 83, EvidenceState.EXPLICIT, []),
            ],
            "unknowns": [
                AttributeCategory.HAIRLINE,
                AttributeCategory.EYES,
                AttributeCategory.NOSE,
                AttributeCategory.FACIAL_HAIR,
                AttributeCategory.DISTINGUISHING_MARKS,
                AttributeCategory.OTHER_OBSERVABLE,
            ],
        },
        # SYNTH-016: Sparse sample (single feature)
        {
            "id": "SYNTH-016",
            "text": "The only feature I noticed was a large hooked nose.",
            "annotations": [
                ("ANN-016-1", AttributeCategory.NOSE, "large hooked", "large hooked nose", 33, 50, EvidenceState.EXPLICIT, []),
            ],
            "unknowns": [
                AttributeCategory.FACE_SHAPE,
                AttributeCategory.HAIR,
                AttributeCategory.HAIRLINE,
                AttributeCategory.EYES,
                AttributeCategory.EYEBROWS,
                AttributeCategory.LIPS,
                AttributeCategory.FACIAL_HAIR,
                AttributeCategory.DISTINGUISHING_MARKS,
                AttributeCategory.OTHER_OBSERVABLE,
            ],
        },
        # SYNTH-017: Uncertainty in multiple attributes
        {
            "id": "SYNTH-017",
            "text": "His eyebrows appeared to be bushy, and the face shape seemed like a heart-shaped outline.",
            "annotations": [
                ("ANN-017-1", AttributeCategory.EYEBROWS, "bushy", "bushy", 28, 33, EvidenceState.UNCERTAIN, ["appeared to be"]),
                ("ANN-017-2", AttributeCategory.FACE_SHAPE, "heart-shaped", "heart-shaped", 68, 80, EvidenceState.UNCERTAIN, ["seemed like"]),
            ],
            "unknowns": [
                AttributeCategory.HAIR,
                AttributeCategory.HAIRLINE,
                AttributeCategory.EYES,
                AttributeCategory.NOSE,
                AttributeCategory.LIPS,
                AttributeCategory.FACIAL_HAIR,
                AttributeCategory.DISTINGUISHING_MARKS,
                AttributeCategory.OTHER_OBSERVABLE,
            ],
        },
        # SYNTH-018: Conflicting eye color testimony
        {
            "id": "SYNTH-018",
            "text": "Witness A claimed bright blue eyes while Witness B insisted on dark brown eyes.",
            "annotations": [
                ("ANN-018-1", AttributeCategory.EYES, "bright blue", "bright blue eyes", 18, 34, EvidenceState.CONTRADICTORY, []),
                ("ANN-018-2", AttributeCategory.EYES, "dark brown", "dark brown eyes", 63, 78, EvidenceState.CONTRADICTORY, []),
            ],
            "unknowns": [
                AttributeCategory.FACE_SHAPE,
                AttributeCategory.HAIR,
                AttributeCategory.HAIRLINE,
                AttributeCategory.EYEBROWS,
                AttributeCategory.NOSE,
                AttributeCategory.LIPS,
                AttributeCategory.FACIAL_HAIR,
                AttributeCategory.DISTINGUISHING_MARKS,
                AttributeCategory.OTHER_OBSERVABLE,
            ],
        },
        # SYNTH-019: Forehead wrinkles and cleft chin
        {
            "id": "SYNTH-019",
            "text": "He had deep forehead wrinkles and a pronounced cleft chin.",
            "annotations": [
                ("ANN-019-1", AttributeCategory.OTHER_OBSERVABLE, "deep forehead wrinkles", "deep forehead wrinkles", 7, 29, EvidenceState.EXPLICIT, []),
                ("ANN-019-2", AttributeCategory.FACE_SHAPE, "cleft chin", "pronounced cleft chin", 36, 57, EvidenceState.EXPLICIT, []),
            ],
            "unknowns": [
                AttributeCategory.HAIR,
                AttributeCategory.HAIRLINE,
                AttributeCategory.EYES,
                AttributeCategory.EYEBROWS,
                AttributeCategory.NOSE,
                AttributeCategory.LIPS,
                AttributeCategory.FACIAL_HAIR,
                AttributeCategory.DISTINGUISHING_MARKS,
            ],
        },
        # SYNTH-020: Goatee and straight low hairline
        {
            "id": "SYNTH-020",
            "text": "The person had a neatly trimmed goatee and a straight low hairline.",
            "annotations": [
                ("ANN-020-1", AttributeCategory.FACIAL_HAIR, "neatly trimmed goatee", "neatly trimmed goatee", 17, 38, EvidenceState.EXPLICIT, []),
                ("ANN-020-2", AttributeCategory.HAIRLINE, "straight low", "straight low hairline", 45, 66, EvidenceState.EXPLICIT, []),
            ],
            "unknowns": [
                AttributeCategory.FACE_SHAPE,
                AttributeCategory.HAIR,
                AttributeCategory.EYES,
                AttributeCategory.EYEBROWS,
                AttributeCategory.NOSE,
                AttributeCategory.LIPS,
                AttributeCategory.DISTINGUISHING_MARKS,
                AttributeCategory.OTHER_OBSERVABLE,
            ],
        },
        # SYNTH-021: Buzzcut and pencil mustache
        {
            "id": "SYNTH-021",
            "text": "He wore a buzzcut and had a thin pencil mustache.",
            "annotations": [
                ("ANN-021-1", AttributeCategory.HAIR, "buzzcut", "buzzcut", 10, 17, EvidenceState.EXPLICIT, []),
                ("ANN-021-2", AttributeCategory.FACIAL_HAIR, "thin pencil mustache", "thin pencil mustache", 28, 48, EvidenceState.EXPLICIT, []),
            ],
            "unknowns": [
                AttributeCategory.FACE_SHAPE,
                AttributeCategory.HAIRLINE,
                AttributeCategory.EYES,
                AttributeCategory.EYEBROWS,
                AttributeCategory.NOSE,
                AttributeCategory.LIPS,
                AttributeCategory.DISTINGUISHING_MARKS,
                AttributeCategory.OTHER_OBSERVABLE,
            ],
        },
        # SYNTH-022: Plucked eyebrows and deep-set green eyes
        {
            "id": "SYNTH-022",
            "text": "She had thin plucked eyebrows framing deep-set green eyes.",
            "annotations": [
                ("ANN-022-1", AttributeCategory.EYEBROWS, "thin plucked", "thin plucked eyebrows", 8, 29, EvidenceState.EXPLICIT, []),
                ("ANN-022-2", AttributeCategory.EYES, "deep-set green", "deep-set green eyes", 38, 57, EvidenceState.EXPLICIT, []),
            ],
            "unknowns": [
                AttributeCategory.FACE_SHAPE,
                AttributeCategory.HAIR,
                AttributeCategory.HAIRLINE,
                AttributeCategory.NOSE,
                AttributeCategory.LIPS,
                AttributeCategory.FACIAL_HAIR,
                AttributeCategory.DISTINGUISHING_MARKS,
                AttributeCategory.OTHER_OBSERVABLE,
            ],
        },
        # SYNTH-023: Teardrop tattoo
        {
            "id": "SYNTH-023",
            "text": "A small teardrop tattoo was clearly visible under the right eye.",
            "annotations": [
                ("ANN-023-1", AttributeCategory.DISTINGUISHING_MARKS, "teardrop tattoo", "small teardrop tattoo", 2, 23, EvidenceState.EXPLICIT, []),
            ],
            "unknowns": [
                AttributeCategory.FACE_SHAPE,
                AttributeCategory.HAIR,
                AttributeCategory.HAIRLINE,
                AttributeCategory.EYES,
                AttributeCategory.EYEBROWS,
                AttributeCategory.NOSE,
                AttributeCategory.LIPS,
                AttributeCategory.FACIAL_HAIR,
                AttributeCategory.OTHER_OBSERVABLE,
            ],
        },
        # SYNTH-024: Chapped lips and wide flat nose
        {
            "id": "SYNTH-024",
            "text": "The individual had dry chapped lips and a wide flat nose.",
            "annotations": [
                ("ANN-024-1", AttributeCategory.LIPS, "dry chapped", "dry chapped lips", 19, 35, EvidenceState.EXPLICIT, []),
                ("ANN-024-2", AttributeCategory.NOSE, "wide flat", "wide flat nose", 42, 56, EvidenceState.EXPLICIT, []),
            ],
            "unknowns": [
                AttributeCategory.FACE_SHAPE,
                AttributeCategory.HAIR,
                AttributeCategory.HAIRLINE,
                AttributeCategory.EYES,
                AttributeCategory.EYEBROWS,
                AttributeCategory.FACIAL_HAIR,
                AttributeCategory.DISTINGUISHING_MARKS,
                AttributeCategory.OTHER_OBSERVABLE,
            ],
        },
    ]

    samples: List[DatasetSample] = []
    for raw in raw_samples:
        anns: List[AttributeAnnotation] = []
        for ann_id, cat, val, span, start, end, cert, cues in raw["annotations"]:
            # Sanity verification at creation time
            actual_span = raw["text"][start:end]
            assert actual_span == span, (
                f"Offset error in {raw['id']} ({ann_id})! Text[{start}:{end}] is '{actual_span}', expected '{span}'"
            )
            anns.append(
                AttributeAnnotation(
                    annotation_id=ann_id,
                    category=cat,
                    attribute_value=val,
                    source_span=span,
                    start_char=start,
                    end_char=end,
                    certainty=cert,
                    annotator_id="FORENSIC-ANNOTATOR-DEV-01",
                    uncertainty_cues=cues,
                    notes=f"Synthetic {cert.value} annotation for pipeline testing",
                )
            )

        # Append explicit UNKNOWN annotations for unmentioned observable features
        for u_idx, unk_cat in enumerate(raw["unknowns"], start=1):
            anns.append(
                AttributeAnnotation(
                    annotation_id=f"{raw['id']}-UNK-{u_idx}",
                    category=unk_cat,
                    attribute_value=None,
                    source_span=None,
                    start_char=None,
                    end_char=None,
                    certainty=EvidenceState.UNKNOWN,
                    annotator_id="FORENSIC-ANNOTATOR-DEV-01",
                    notes="Feature unmentioned in testimony - strictly preserved as UNKNOWN",
                )
            )

        samples.append(
            DatasetSample(
                sample_id=raw["id"],
                witness_text=raw["text"],
                annotations=anns,
                metadata=common_metadata,
            )
        )

    return ForensicDataset(
        name="Phantom AI Synthetic Forensic Witness Benchmark",
        version="1.0.0",
        license="CC-BY-4.0-Synthetic",
        source="Phantom AI Synthetic Witness Benchmarking Suite",
        is_synthetic=True,
        samples=samples,
    )


if __name__ == "__main__":
    import os
    from data.dataset_loader import DatasetLoader

    dataset = get_synthetic_dataset()
    loader = DatasetLoader()
    output_path = os.path.join("datasets", "synthetic_witness_dataset.json")
    loader.save_json(dataset, output_path)
    print(f"Successfully generated and validated {len(dataset.samples)} synthetic samples in {output_path}")
