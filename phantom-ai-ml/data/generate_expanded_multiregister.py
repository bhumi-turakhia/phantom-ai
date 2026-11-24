"""
Expanded Multi-Register Synthetic Dataset Generator for Phantom AI 3.0 (Phase 3C.2).
Generates ~1,600-1,800 diverse, evidence-aware synthetic witness descriptions across >=10 linguistic registers:
1. Ordinary witness narratives
2. Informal conversational recall (hesitations, "um", "like", "you know")
3. Police interview transcripts ("Detective:", "Witness:")
4. Courtroom direct/cross-examination ("Q:", "A:")
5. 911 dispatch calls ("Dispatcher:", "Caller:")
6. Short fragmented notes / police blotters
7. Long descriptive narratives with rich scene context
8. Multi-attribute complex sentences (4-6 attributes)
9. Negated observations ("no beard", "clean-shaven", "without scars")
10. Uncertain and contradictory observations ("maybe", "perhaps", conflicting witness reports)

Enforces:
- Exact character offsets: text[start:end] == span_text
- Evidence certainty: explicit, uncertain, unknown, contradictory, negated
- Non-demographic attributes only
- Zero leakage into frozen evaluation sets (test.jsonl and out_of_frame_eval.jsonl)
- is_synthetic: True on all samples and annotations
"""

import json
import os
import random
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional

from schemas.attributes import AttributeCategory, EvidenceState
from schemas.dataset import (
    DatasetSample,
    AttributeAnnotation,
    SampleMetadata,
    ForensicDataset,
)
from data.validator import DatasetValidator

# Extended Vocabulary for all 10 categories
VOCAB = {
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
        ("gaunt hollow cheeks", "gaunt hollow"),
        ("soft rounded face", "soft rounded"),
        ("diamond-shaped face", "diamond-shaped"),
        ("boxy jaw", "boxy"),
        ("strong defined chin", "strong defined"),
        ("tapered jawline", "tapered"),
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
        ("dreadlocks", "dreadlocks"),
        ("spiky brown hair", "spiky brown"),
        ("unruly silver hair", "unruly silver"),
        ("bleached blonde hair", "bleached blonde"),
        ("coarse wavy hair", "coarse wavy"),
        ("slicked side-parted hair", "slicked side-parted"),
        ("shaggy brunette hair", "shaggy brunette"),
        ("tied-back ponytail", "ponytail"),
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
        ("mature squared hairline", "mature squared"),
        ("m-shaped hairline", "m-shaped"),
        ("sharp clean hairline", "sharp clean"),
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
        ("watery amber eyes", "watery amber"),
        ("heavy-lidded brown eyes", "heavy-lidded brown"),
        ("droopy hazel eyes", "droopy hazel"),
        ("striking icy blue eyes", "icy blue"),
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
        ("neatly groomed eyebrows", "neatly groomed"),
        ("furrowed dark eyebrows", "furrowed dark"),
        ("faint blonde eyebrows", "faint blonde"),
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
        ("bulbous nose tip", "bulbous"),
        ("flattened nasal bridge", "flattened nasal bridge"),
        ("narrow upturned nose", "narrow upturned"),
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
        ("asymmetric upper lip", "asymmetric upper lip"),
        ("firm set lips", "firm set"),
        ("plump rounded lips", "plump rounded"),
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
        ("handlebar mustache", "handlebar mustache"),
        ("scruffy jawline beard", "scruffy jawline beard"),
        ("mutton-chop sideburns", "mutton-chop sideburns"),
        ("sparse chin goatee", "sparse chin goatee"),
        ("thick bristly mustache", "thick bristly mustache"),
        ("neat horseshoe mustache", "horseshoe mustache"),
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
        ("vertical surgical scar on chin", "surgical scar"),
        ("cluster of dark freckles", "dark freckles"),
        ("cross tattoo near temple", "cross tattoo"),
        ("faded acne scarring on cheeks", "acne scarring"),
        ("prominent mole on left cheek", "mole on cheek"),
        ("deep scar across nasal bridge", "scar on bridge"),
        ("notched left eyebrow", "notched eyebrow"),
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
        ("crow's feet around eyes", "crow's feet"),
        ("thick tortoiseshell spectacles", "tortoiseshell spectacles"),
        ("diamond stud in right nostril", "diamond nose stud"),
        ("prominent laugh lines", "laugh lines"),
        ("pierced left eyebrow ring", "eyebrow ring"),
        ("chipped front tooth", "chipped tooth"),
    ],
}

# Negation phrases that specify absence of a facial attribute
NEGATIONS = [
    (AttributeCategory.FACIAL_HAIR, "no beard or mustache", "no beard or mustache"),
    (AttributeCategory.FACIAL_HAIR, "completely clean-shaven", "clean-shaven"),
    (AttributeCategory.FACIAL_HAIR, "no facial hair whatsoever", "no facial hair"),
    (AttributeCategory.FACIAL_HAIR, "did not have a beard", "did not have a beard"),
    (AttributeCategory.FACIAL_HAIR, "no mustache or stubble", "no mustache or stubble"),
    (AttributeCategory.FACIAL_HAIR, "smooth-shaven chin", "smooth-shaven chin"),
    (AttributeCategory.DISTINGUISHING_MARKS, "no visible scars or tattoos", "no scars or tattoos"),
    (AttributeCategory.DISTINGUISHING_MARKS, "no marks on his face", "no marks"),
    (AttributeCategory.DISTINGUISHING_MARKS, "without any scars", "without scars"),
    (AttributeCategory.HAIR, "not bald at all", "not bald"),
    (AttributeCategory.OTHER_OBSERVABLE, "no glasses or earrings", "no glasses or earrings"),
    (AttributeCategory.OTHER_OBSERVABLE, "did not wear glasses", "did not wear glasses"),
]

UNCERTAIN_CUES = [
    "I think",
    "seemed to have",
    "looked like",
    "possibly",
    "maybe",
    "might have had",
    "appeared to be",
    "as far as I could tell",
    "if I recall correctly",
    "give or take",
]


class SpanTracker:
    def __init__(self):
        self.text = ""
        self.annotations: List[AttributeAnnotation] = []
        self.ann_counter = 0

    def add_text(self, text: str):
        self.text += text

    def add_span(
        self,
        category: AttributeCategory,
        value: str,
        span_text: str,
        certainty: EvidenceState = EvidenceState.EXPLICIT,
        cues: Optional[List[str]] = None,
        notes: str = "Expanded synthetic multi-register corpus"
    ):
        start = len(self.text)
        self.text += span_text
        end = len(self.text)
        assert self.text[start:end] == span_text, f"Offset error: '{self.text[start:end]}' != '{span_text}'"

        self.ann_counter += 1
        ann_id = f"ANN-{uuid.uuid4().hex[:8]}"
        self.annotations.append(
            AttributeAnnotation(
                annotation_id=ann_id,
                category=category,
                attribute_value=value if certainty != EvidenceState.UNKNOWN else None,
                source_span=span_text if certainty != EvidenceState.UNKNOWN else None,
                start_char=start if certainty != EvidenceState.UNKNOWN else None,
                end_char=end if certainty != EvidenceState.UNKNOWN else None,
                certainty=certainty,
                annotator_id="MULTI-REGISTER-SYNTH-EXPANSION",
                annotation_timestamp=datetime.now(timezone.utc).isoformat(),
                uncertainty_cues=cues or [],
                notes=notes,
            )
        )

    def add_unknowns_for_missing(self):
        present_categories = {a.category for a in self.annotations}
        for cat in AttributeCategory:
            if cat not in present_categories:
                self.ann_counter += 1
                self.annotations.append(
                    AttributeAnnotation(
                        annotation_id=f"UNK-{uuid.uuid4().hex[:8]}",
                        category=cat,
                        attribute_value=None,
                        source_span=None,
                        start_char=None,
                        end_char=None,
                        certainty=EvidenceState.UNKNOWN,
                        annotator_id="MULTI-REGISTER-SYNTH-EXPANSION",
                        annotation_timestamp=datetime.now(timezone.utc).isoformat(),
                        uncertainty_cues=[],
                        notes="Unobserved attribute explicitly marked as UNKNOWN",
                    )
                )


def pick_attribute(category: AttributeCategory, rng: random.Random) -> Tuple[str, str]:
    choices = VOCAB[category]
    return rng.choice(choices)


# ---------------------------------------------------------------------------
# Register Generators
# ---------------------------------------------------------------------------

def gen_register_1_ordinary_narrative(rng: random.Random) -> SpanTracker:
    st = SpanTracker()
    intros = [
        "I was standing at the corner when I caught a clear view of the person.",
        "As the individual walked past the shop window, I observed the face closely.",
        "I was sitting in my parked car and saw the subject approach the doorway.",
        "While waiting at the bus stop, I noticed someone behaving suspiciously.",
        "I happened to look directly at the subject as they crossed the street.",
        "The person was standing right near the cash register when I entered.",
        "I got a good look at the suspect under the bright streetlamp.",
        "When the door opened, the individual looked up and I saw their face.",
    ]
    st.add_text(rng.choice(intros) + " ")

    cats = rng.sample(list(AttributeCategory), k=rng.randint(2, 4))
    
    # First attribute
    span1, val1 = pick_attribute(cats[0], rng)
    connectors1 = ["He possessed ", "The subject had ", "They featured ", "I clearly saw ", "He had "]
    st.add_text(rng.choice(connectors1))
    st.add_span(cats[0], val1, span1, EvidenceState.EXPLICIT)

    # Second attribute
    span2, val2 = pick_attribute(cats[1], rng)
    connectors2 = [", along with ", ", combined with ", " accompanied by ", " and also "]
    st.add_text(rng.choice(connectors2))
    st.add_span(cats[1], val2, span2, EvidenceState.EXPLICIT)

    if len(cats) >= 3:
        span3, val3 = pick_attribute(cats[2], rng)
        connectors3 = [". In addition, there was ", ". Furthermore, I noted ", ". The person also had "]
        st.add_text(rng.choice(connectors3))
        st.add_span(cats[2], val3, span3, EvidenceState.EXPLICIT)
        if len(cats) >= 4:
            span4, val4 = pick_attribute(cats[3], rng)
            st.add_text(" and ")
            st.add_span(cats[3], val4, span4, EvidenceState.EXPLICIT)

    st.add_text(".")
    st.add_unknowns_for_missing()
    return st


def gen_register_2_informal_conversation(rng: random.Random) -> SpanTracker:
    st = SpanTracker()
    openers = [
        "Yeah, so like, I was right there and saw the guy.",
        "Well, um, it happened super fast, but you know,",
        "Okay, let me think for a second... um, right,",
        "Honestly, the main thing I remember was the face, like,",
        "I mean, he came right up to the counter, you know?",
        "So basically, I looked up from my phone and saw him,",
        "Look, I only saw him for a few seconds, but honestly,",
    ]
    st.add_text(rng.choice(openers) + " ")

    cats = rng.sample(list(AttributeCategory), k=rng.randint(2, 3))
    
    # Hedged or conversational attribute 1
    span1, val1 = pick_attribute(cats[0], rng)
    filler1 = rng.choice(["he had this ", "he totally had ", "like he had ", "there was this "])
    st.add_text(filler1)
    st.add_span(cats[0], val1, span1, EvidenceState.EXPLICIT)

    span2, val2 = pick_attribute(cats[1], rng)
    filler2 = rng.choice([", and then like ", ", plus you could see ", ", and oh yeah, ", ", with this "])
    st.add_text(filler2)
    st.add_span(cats[1], val2, span2, EvidenceState.EXPLICIT)

    if len(cats) >= 3:
        span3, val3 = pick_attribute(cats[2], rng)
        st.add_text(rng.choice([". And I think he also had ", ". Also, pretty sure there was "]))
        st.add_span(cats[2], val3, span3, EvidenceState.UNCERTAIN, cues=["pretty sure", "think"])

    closers = [
        ", if that makes sense.",
        ", that's pretty much what stuck out.",
        ", you know what I mean?",
        ", I'm almost positive about that.",
        ".",
    ]
    st.add_text(rng.choice(closers))
    st.add_unknowns_for_missing()
    return st


def gen_register_3_police_interview(rng: random.Random) -> SpanTracker:
    st = SpanTracker()
    prompts = [
        ("Detective: Can you describe the suspect's facial features?", "Witness: Yes, absolutely."),
        ("Officer: What details of the face did you focus on?", "Witness: Well, primarily"),
        ("Investigator: Did anything specific about the suspect stand out?", "Witness: What I noted was"),
        ("Detective: Please recount what the individual looked like.", "Witness: The suspect had"),
        ("Officer: Take your time. What did you notice about his appearance?", "Witness: I remember"),
    ]
    det, wit = rng.choice(prompts)
    st.add_text(f"{det} {wit} ")

    cats = rng.sample(list(AttributeCategory), k=rng.randint(2, 4))
    span1, val1 = pick_attribute(cats[0], rng)
    st.add_text("the person exhibited ")
    st.add_span(cats[0], val1, span1, EvidenceState.EXPLICIT)

    span2, val2 = pick_attribute(cats[1], rng)
    st.add_text(", and clearly had ")
    st.add_span(cats[1], val2, span2, EvidenceState.EXPLICIT)

    if len(cats) >= 3:
        span3, val3 = pick_attribute(cats[2], rng)
        st.add_text(". The officer asked about upper facial marks, and I noted ")
        st.add_span(cats[2], val3, span3, EvidenceState.EXPLICIT)

    if len(cats) >= 4:
        span4, val4 = pick_attribute(cats[3], rng)
        st.add_text(", along with ")
        st.add_span(cats[3], val4, span4, EvidenceState.EXPLICIT)

    st.add_text(".")
    st.add_unknowns_for_missing()
    return st


def gen_register_4_courtroom_qa(rng: random.Random) -> SpanTracker:
    st = SpanTracker()
    intros = [
        ("Q: Directing your attention to the perpetrator, what facial features did you observe?", "A: Under oath, I testified that the defendant displayed"),
        ("Q: Did you have an unobstructed view of the individual's face?", "A: Yes, I observed"),
        ("Q: Could you describe the physical characteristics of the assailant?", "A: The individual possessed"),
        ("Q: Looking at the person, what distinguishable features were visible?", "A: I clearly identified"),
    ]
    q, a = rng.choice(intros)
    st.add_text(f"{q} {a} ")

    cats = rng.sample(list(AttributeCategory), k=rng.randint(2, 3))
    span1, val1 = pick_attribute(cats[0], rng)
    st.add_span(cats[0], val1, span1, EvidenceState.EXPLICIT)

    span2, val2 = pick_attribute(cats[1], rng)
    st.add_text(", as well as ")
    st.add_span(cats[1], val2, span2, EvidenceState.EXPLICIT)

    if len(cats) >= 3:
        span3, val3 = pick_attribute(cats[2], rng)
        st.add_text(". Cross-examination confirmed ")
        st.add_span(cats[2], val3, span3, EvidenceState.EXPLICIT)

    st.add_text(".")
    st.add_unknowns_for_missing()
    return st


def gen_register_5_dispatch_call(rng: random.Random) -> SpanTracker:
    st = SpanTracker()
    headers = [
        "Dispatcher: 911 dispatch, what is your emergency and description? Caller: Suspect fleeing south.",
        "Dispatch: Unit 4, broadcast suspect description. Unit 4: Be on the lookout.",
        "Operator: Can you give a physical description for incoming units? Caller: Got a visual here.",
        "Dispatch Log: BOLO issued for adult male subject.",
    ]
    st.add_text(rng.choice(headers) + " Subject displays ")

    cats = rng.sample(list(AttributeCategory), k=rng.randint(2, 3))
    span1, val1 = pick_attribute(cats[0], rng)
    st.add_span(cats[0], val1, span1, EvidenceState.EXPLICIT)

    span2, val2 = pick_attribute(cats[1], rng)
    st.add_text(", ")
    st.add_span(cats[1], val2, span2, EvidenceState.EXPLICIT)

    if len(cats) >= 3:
        span3, val3 = pick_attribute(cats[2], rng)
        st.add_text(", and ")
        st.add_span(cats[2], val3, span3, EvidenceState.EXPLICIT)

    closers = [
        ". Units advised to proceed with caution.",
        ". Last seen heading towards the transit center.",
        ". Responding patrol en route.",
        ". Description corroborated by secondary caller.",
    ]
    st.add_text(rng.choice(closers))
    st.add_unknowns_for_missing()
    return st


def gen_register_6_short_fragmented(rng: random.Random) -> SpanTracker:
    st = SpanTracker()
    cats = rng.sample(list(AttributeCategory), k=rng.randint(2, 4))
    
    parts = []
    for cat in cats:
        span, val = pick_attribute(cat, rng)
        parts.append((cat, val, span))

    # Pattern: "Attribute 1. Attribute 2. Attribute 3."
    for idx, (cat, val, span) in enumerate(parts):
        # capitalize first letter if at sentence start
        span_capped = span[0].upper() + span[1:] if idx == 0 or st.text.endswith(". ") else span
        st.add_span(cat, val, span_capped, EvidenceState.EXPLICIT)
        if idx < len(parts) - 1:
            punct = rng.choice([". ", "; ", ", "])
            st.add_text(punct)
        else:
            st.add_text(".")

    st.add_unknowns_for_missing()
    return st


def gen_register_7_long_narrative(rng: random.Random) -> SpanTracker:
    st = SpanTracker()
    backgrounds = [
        "The incident occurred around twilight in the alley behind the convenience store. While the ambient illumination was somewhat dim, the security floodlight illuminated the suspect's facial profile with remarkable clarity.",
        "According to the store manager who had direct face-to-face interaction during the transaction, the individual stood approximately three feet away across the counter for nearly two full minutes.",
        "The delivery courier witnessed the encounter while completing a drop-off at the loading dock, observing the perpetrator pause beneath the outdoor lamp post before turning swiftly toward the exit.",
        "Surveillance review and witness interview records indicate that the complainant maintained eye contact with the unidentified individual throughout the confrontation in the lobby.",
    ]
    st.add_text(rng.choice(backgrounds) + " In the detailed testimony, the witness recounted seeing ")

    cats = rng.sample(list(AttributeCategory), k=rng.randint(3, 5))
    span1, val1 = pick_attribute(cats[0], rng)
    st.add_span(cats[0], val1, span1, EvidenceState.EXPLICIT)

    span2, val2 = pick_attribute(cats[1], rng)
    st.add_text(", which was accented by ")
    st.add_span(cats[1], val2, span2, EvidenceState.EXPLICIT)

    span3, val3 = pick_attribute(cats[2], rng)
    st.add_text(". The witness also distinctly recalled observing ")
    st.add_span(cats[2], val3, span3, EvidenceState.EXPLICIT)

    if len(cats) >= 4:
        span4, val4 = pick_attribute(cats[3], rng)
        st.add_text(", complemented by ")
        st.add_span(cats[3], val4, span4, EvidenceState.EXPLICIT)

    if len(cats) >= 5:
        span5, val5 = pick_attribute(cats[4], rng)
        st.add_text(", and finally ")
        st.add_span(cats[4], val5, span5, EvidenceState.EXPLICIT)

    st.add_text(". These identifying marks were subsequently documented in the formal report.")
    st.add_unknowns_for_missing()
    return st


def gen_register_8_multiple_attributes(rng: random.Random) -> SpanTracker:
    st = SpanTracker()
    starters = [
        "The composite interview yielded a comprehensive facial profile comprising ",
        "Upon thorough examination, the subject was identified by a configuration of ",
        "The witness provided an extensive accounting of the suspect, who presented ",
        "Several distinct physical features defined the suspect's appearance, namely ",
        "The individual exhibited a pronounced combination of facial attributes including ",
    ]
    st.add_text(rng.choice(starters))

    # 4 to 6 attributes coordinated
    k = rng.randint(4, 6)
    cats = rng.sample(list(AttributeCategory), k=k)

    for i, cat in enumerate(cats):
        span, val = pick_attribute(cat, rng)
        st.add_span(cat, val, span, EvidenceState.EXPLICIT)
        if i == len(cats) - 2:
            st.add_text(", and ")
        elif i < len(cats) - 2:
            st.add_text(", ")
        else:
            st.add_text(".")

    st.add_unknowns_for_missing()
    return st


def gen_register_9_negated_observations(rng: random.Random) -> SpanTracker:
    """
    CRITICAL: Generates descriptions with explicit negations.
    Negated spans are annotated with EvidenceState.NEGATED so they DO NOT receive positive BIO tags!
    """
    st = SpanTracker()
    
    # Select 1 or 2 negations
    neg_cat, neg_span, neg_val = rng.choice(NEGATIONS)
    
    # And 1 to 3 positive affirmed attributes
    available_cats = [c for c in AttributeCategory if c != neg_cat]
    pos_cats = rng.sample(available_cats, k=rng.randint(1, 3))

    patterns = [
        "The suspect had {NEG}, but definitely had {POS1} and {POS2}.",
        "Noticeably, the individual had {NEG}; however, I clearly observed {POS1}.",
        "The witness confirmed {NEG}. The subject did, however, display {POS1} along with {POS2}.",
        "Although the person had {NEG}, they were easily recognised by their {POS1}.",
        "Officer note: Subject had {NEG}. Distinguishing features included {POS1} and {POS2}.",
        "The complainant stressed {NEG}, while noting prominent {POS1}.",
    ]
    pat = rng.choice(patterns)

    # We build the sentence step by step to track offsets precisely
    if pat.startswith("The suspect had "):
        st.add_text("The suspect had ")
        st.add_span(neg_cat, neg_val, neg_span, EvidenceState.NEGATED, notes="Explicitly negated facial attribute")
        st.add_text(", but definitely had ")
        span1, val1 = pick_attribute(pos_cats[0], rng)
        st.add_span(pos_cats[0], val1, span1, EvidenceState.EXPLICIT)
        if len(pos_cats) > 1:
            st.add_text(" and ")
            span2, val2 = pick_attribute(pos_cats[1], rng)
            st.add_span(pos_cats[1], val2, span2, EvidenceState.EXPLICIT)
        st.add_text(".")
    elif pat.startswith("Noticeably, the individual had "):
        st.add_text("Noticeably, the individual had ")
        st.add_span(neg_cat, neg_val, neg_span, EvidenceState.NEGATED, notes="Explicitly negated facial attribute")
        st.add_text("; however, I clearly observed ")
        span1, val1 = pick_attribute(pos_cats[0], rng)
        st.add_span(pos_cats[0], val1, span1, EvidenceState.EXPLICIT)
        st.add_text(".")
    elif pat.startswith("The witness confirmed "):
        st.add_text("The witness confirmed ")
        st.add_span(neg_cat, neg_val, neg_span, EvidenceState.NEGATED, notes="Explicitly negated facial attribute")
        st.add_text(". The subject did, however, display ")
        span1, val1 = pick_attribute(pos_cats[0], rng)
        st.add_span(pos_cats[0], val1, span1, EvidenceState.EXPLICIT)
        if len(pos_cats) > 1:
            st.add_text(" along with ")
            span2, val2 = pick_attribute(pos_cats[1], rng)
            st.add_span(pos_cats[1], val2, span2, EvidenceState.EXPLICIT)
        st.add_text(".")
    elif pat.startswith("Although the person had "):
        st.add_text("Although the person had ")
        st.add_span(neg_cat, neg_val, neg_span, EvidenceState.NEGATED, notes="Explicitly negated facial attribute")
        st.add_text(", they were easily recognised by their ")
        span1, val1 = pick_attribute(pos_cats[0], rng)
        st.add_span(pos_cats[0], val1, span1, EvidenceState.EXPLICIT)
        st.add_text(".")
    elif pat.startswith("Officer note: Subject had "):
        st.add_text("Officer note: Subject had ")
        st.add_span(neg_cat, neg_val, neg_span, EvidenceState.NEGATED, notes="Explicitly negated facial attribute")
        st.add_text(". Distinguishing features included ")
        span1, val1 = pick_attribute(pos_cats[0], rng)
        st.add_span(pos_cats[0], val1, span1, EvidenceState.EXPLICIT)
        if len(pos_cats) > 1:
            st.add_text(" and ")
            span2, val2 = pick_attribute(pos_cats[1], rng)
            st.add_span(pos_cats[1], val2, span2, EvidenceState.EXPLICIT)
        st.add_text(".")
    else:
        st.add_text("The complainant stressed ")
        st.add_span(neg_cat, neg_val, neg_span, EvidenceState.NEGATED, notes="Explicitly negated facial attribute")
        st.add_text(", while noting prominent ")
        span1, val1 = pick_attribute(pos_cats[0], rng)
        st.add_span(pos_cats[0], val1, span1, EvidenceState.EXPLICIT)
        st.add_text(".")

    st.add_unknowns_for_missing()
    return st


def gen_register_10_uncertain_contradictory(rng: random.Random) -> SpanTracker:
    st = SpanTracker()
    is_contradictory = rng.random() < 0.35

    if is_contradictory:
        # Two witnesses disagree on an attribute
        cat = rng.choice(list(AttributeCategory))
        choices = VOCAB[cat]
        if len(choices) >= 2:
            (span1, val1), (span2, val2) = rng.sample(choices, 2)
        else:
            span1, val1 = choices[0]
            span2, val2 = choices[0]

        other_cat = rng.choice([c for c in AttributeCategory if c != cat])
        pos_span, pos_val = pick_attribute(other_cat, rng)

        patterns = [
            ("Witness A claimed the suspect had ", ", whereas Witness B insisted on ", "; both agreed on "),
            ("Initial dispatch noted ", ", but the secondary witness reported ", ". Consensus remained on "),
            ("The clerk remembered ", ", while the security guard described ", ", though both recalled "),
        ]
        p1, p2, p3 = rng.choice(patterns)

        st.add_text(p1)
        st.add_span(cat, val1, span1, EvidenceState.CONTRADICTORY, notes="Conflicting witness testimony")
        st.add_text(p2)
        st.add_span(cat, val2, span2, EvidenceState.CONTRADICTORY, notes="Conflicting witness testimony")
        st.add_text(p3)
        st.add_span(other_cat, pos_val, pos_span, EvidenceState.EXPLICIT)
        st.add_text(".")
    else:
        # Hedged / uncertain observation
        cat = rng.choice(list(AttributeCategory))
        cue = rng.choice(UNCERTAIN_CUES)
        span, val = pick_attribute(cat, rng)

        other_cat = rng.choice([c for c in AttributeCategory if c != cat])
        pos_span, pos_val = pick_attribute(other_cat, rng)

        st.add_text(f"The witness stated that {cue} ")
        st.add_span(cat, val, span, EvidenceState.UNCERTAIN, cues=[cue])
        st.add_text(", but was entirely certain about the ")
        st.add_span(other_cat, pos_val, pos_span, EvidenceState.EXPLICIT)
        st.add_text(".")

    st.add_unknowns_for_missing()
    return st


REGISTERS = [
    ("ordinary_narrative", gen_register_1_ordinary_narrative),
    ("informal_conversation", gen_register_2_informal_conversation),
    ("police_interview", gen_register_3_police_interview),
    ("courtroom_qa", gen_register_4_courtroom_qa),
    ("dispatch_call", gen_register_5_dispatch_call),
    ("short_fragmented", gen_register_6_short_fragmented),
    ("long_narrative", gen_register_7_long_narrative),
    ("multiple_attributes", gen_register_8_multiple_attributes),
    ("negated_observations", gen_register_9_negated_observations),
    ("uncertain_contradictory", gen_register_10_uncertain_contradictory),
]


def generate_expanded_dataset(
    target_count: int = 1650,
    seed: int = 42,
    existing_eval_paths: List[str] = None
) -> List[DatasetSample]:
    """
    Generates target_count new samples distributed evenly across all 10 registers.
    Verifies character offsets, verifies absence of test leakage, and assigns sample IDs.
    """
    rng = random.Random(seed)

    # Load frozen evaluation texts to verify zero leakage
    eval_texts = set()
    if existing_eval_paths:
        for path in existing_eval_paths:
            if os.path.exists(path):
                with open(path) as f:
                    for line in f:
                        if line.strip():
                            eval_texts.add(json.loads(line)["witness_text"].strip().lower())
    print(f"Loaded {len(eval_texts)} frozen evaluation texts for strict anti-leakage verification.")

    samples_per_reg = (target_count // len(REGISTERS)) + 1
    generated_samples: List[DatasetSample] = []
    seen_texts = set()

    sample_idx = 1
    for reg_name, reg_gen in REGISTERS:
        reg_count = 0
        attempts = 0
        while reg_count < samples_per_reg and attempts < samples_per_reg * 10:
            attempts += 1
            st = reg_gen(rng)
            text = st.text.strip()

            # Leakage & Duplicate check
            norm_text = text.lower()
            if norm_text in eval_texts:
                print(f"LEAKAGE PREVENTED: Generated text matched frozen eval set! Skipping.")
                continue
            if norm_text in seen_texts:
                continue

            # Verify offsets internally
            sample_id = f"SYNTH-PH3C2-{sample_idx:04d}"
            metadata = SampleMetadata(
                dataset_version="2.0.0",
                source=f"Phantom AI Phase 3C.2 Multi-Register Synthetic Corpus ({reg_name})",
                license="CC-BY-4.0-Synthetic",
                is_synthetic=True,
                notes=f"Linguistic register: {reg_name}",
            )
            sample = DatasetSample(
                sample_id=sample_id,
                witness_text=text,
                annotations=st.annotations,
                metadata=metadata,
            )

            errs = sample.verify_offsets()
            if errs:
                raise ValueError(f"Offset verification failed for sample {sample_id}: {errs}")

            seen_texts.add(norm_text)
            generated_samples.append(sample)
            sample_idx += 1
            reg_count += 1

        print(f"Register '{reg_name:<24}': generated {reg_count} valid samples.")

    # Trim to target count
    rng.shuffle(generated_samples)
    final_samples = generated_samples[:target_count]
    print(f"Total new multi-register samples generated: {len(final_samples)}")
    return final_samples


if __name__ == "__main__":
    eval_paths = [
        "phantom-ai-ml/datasets/nlp/test.jsonl",
        "phantom-ai-ml/datasets/nlp/out_of_frame_eval.jsonl",
    ]
    samples = generate_expanded_dataset(target_count=1650, seed=42, existing_eval_paths=eval_paths)
    print(f"Successfully generated {len(samples)} samples.")

