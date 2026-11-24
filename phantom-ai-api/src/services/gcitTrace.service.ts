/**
 * GCIT Traceability and Explainability Service
 * Computes deterministic, auditable mapping from extracted/approved evidence attributes
 * to the 20-dimensional Generator-Conditioning Input Tensor (GCIT).
 */

export interface IGCITTraceItem {
  attributeName: string;
  category: string;
  effectiveValue: string | null;
  originalValue: string | null;
  evidenceState: 'explicit' | 'uncertain' | 'negated' | 'contradictory' | 'unknown';
  originalEvidenceState: 'explicit' | 'uncertain' | 'negated' | 'contradictory' | 'unknown';
  sourceSpan: string;
  startChar: number | null;
  endChar: number | null;
  surroundingContext: string;
  extractedBy: string;
  confidence: number | null;
  isCorrected: boolean;
  correction?: {
    correctedValue: string | null;
    correctedEvidenceState: string;
    reason: string;
    timestamp?: Date | string;
    correctedBy?: string;
    rejected?: boolean;
  };
  reviewStatus?: string;
  gcitStatus: {
    isEligible: boolean;
    gcitDimension: string | null;
    gcitIndex: number | null;
    conditionValue: number;
    maskValue: number;
    polarity: 'normal' | 'inverted' | 'none';
    isIncluded: boolean;
    inclusionReason: string;
  };
}

export const GCIT_20_DIMENSIONS: Array<{
  name: string;
  index: number;
  category: string;
  polarity: 'normal' | 'inverted';
  description: string;
}> = [
  { name: 'hair_black', index: 0, category: 'hair', polarity: 'normal', description: 'Black hair color' },
  { name: 'hair_brown', index: 1, category: 'hair', polarity: 'normal', description: 'Brown hair color' },
  { name: 'hair_blond', index: 2, category: 'hair', polarity: 'normal', description: 'Blond / blonde hair color' },
  { name: 'hair_gray', index: 3, category: 'hair', polarity: 'normal', description: 'Gray / white hair color' },
  { name: 'hair_bald', index: 4, category: 'hair', polarity: 'normal', description: 'Baldness / absence of scalp hair' },
  { name: 'hair_bangs', index: 5, category: 'hair', polarity: 'normal', description: 'Frontal bangs / fringe' },
  { name: 'hair_straight', index: 6, category: 'hair', polarity: 'normal', description: 'Straight hair texture' },
  { name: 'hair_wavy', index: 7, category: 'hair', polarity: 'normal', description: 'Wavy / curly hair texture' },
  { name: 'hairline_receding', index: 8, category: 'hairline', polarity: 'normal', description: 'Receding frontal hairline' },
  { name: 'facial_mustache', index: 9, category: 'facial_hair', polarity: 'normal', description: 'Upper lip mustache' },
  { name: 'facial_goatee', index: 10, category: 'facial_hair', polarity: 'normal', description: 'Chin goatee' },
  { name: 'facial_sideburns', index: 11, category: 'facial_hair', polarity: 'normal', description: 'Prominent sideburns' },
  { name: 'facial_stubble', index: 12, category: 'facial_hair', polarity: 'normal', description: '5 o\'clock shadow / stubble' },
  { name: 'facial_no_beard', index: 13, category: 'facial_hair', polarity: 'inverted', description: 'Absence of beard (+1: clean-shaven / no beard; -1: beard present)' },
  { name: 'nose_big', index: 14, category: 'nose', polarity: 'normal', description: 'Large / broad nose' },
  { name: 'nose_pointy', index: 15, category: 'nose', polarity: 'normal', description: 'Pointed / sharp nose tip' },
  { name: 'lips_big', index: 16, category: 'lips', polarity: 'normal', description: 'Thick / full lips' },
  { name: 'face_oval', index: 17, category: 'face_shape', polarity: 'normal', description: 'Oval face shape' },
  { name: 'face_cheekbones', index: 18, category: 'face_shape', polarity: 'normal', description: 'High / prominent cheekbones' },
  { name: 'glasses', index: 19, category: 'other_observable', polarity: 'normal', description: 'Eyeglasses / spectacles' },
];

/**
 * Extracts a coherent sentence or contextual window around the given character span.
 */
export const extractSurroundingSentence = (
  statement: string,
  start: number | null,
  end: number | null
): string => {
  if (!statement || typeof start !== 'number' || typeof end !== 'number') {
    return statement ? statement.trim() : '';
  }

  const safeStart = Math.max(0, Math.min(start, statement.length));
  const safeEnd = Math.max(safeStart, Math.min(end, statement.length));

  // Search backward for sentence boundary (. ! ? \n)
  let sentenceStart = statement.lastIndexOf('.', safeStart);
  const qStart = statement.lastIndexOf('?', safeStart);
  const eStart = statement.lastIndexOf('!', safeStart);
  const nStart = statement.lastIndexOf('\n', safeStart);

  sentenceStart = Math.max(sentenceStart, qStart, eStart, nStart);
  sentenceStart = sentenceStart === -1 ? 0 : sentenceStart + 1;

  // Search forward for sentence boundary
  let sentenceEnd = statement.indexOf('.', safeEnd);
  const qEnd = statement.indexOf('?', safeEnd);
  const eEnd = statement.indexOf('!', safeEnd);
  const nEnd = statement.indexOf('\n', safeEnd);

  const ends = [sentenceEnd, qEnd, eEnd, nEnd].filter((i) => i !== -1);
  sentenceEnd = ends.length > 0 ? Math.min(...ends) + 1 : statement.length;

  const result = statement.substring(sentenceStart, sentenceEnd).trim();
  return result || statement.trim();
};

/**
 * Formats a category slug into a readable label.
 */
export const formatAttributeCategory = (category: string): string => {
  const map: Record<string, string> = {
    hair: 'Hair Style & Color',
    hair_color: 'Hair Color',
    hair_style: 'Hair Texture',
    hairline: 'Hairline',
    facial_hair: 'Facial Hair',
    nose: 'Nose Geometry',
    lips: 'Lip Shape',
    face_shape: 'Face Shape',
    eyes: 'Eye Characteristics',
    eye_color: 'Eye Color',
    eyebrows: 'Eyebrows',
    distinguishing_marks: 'Distinguishing Marks',
    other_observable: 'Accessories & Observables',
    gender: 'Gender Perception',
    age: 'Estimated Age',
    skin_tone: 'Complexion & Skin Tone',
    build: 'Physical Build',
  };
  return map[category.toLowerCase()] || category.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
};

/**
 * Analyzes an attribute and resolves its exact GCIT mapping, condition value, mask, and explanation.
 */
export const resolveGCITTraceForAttribute = (
  category: string,
  effectiveValue: string | null,
  evidenceState: 'explicit' | 'uncertain' | 'negated' | 'contradictory' | 'unknown'
): IGCITTraceItem['gcitStatus'] => {
  const normCat = category.toLowerCase().trim();
  const val = (effectiveValue || '').toLowerCase().trim();

  // 1. Contradictory evidence: Blocks generation
  if (evidenceState === 'contradictory') {
    return {
      isEligible: false,
      gcitDimension: null,
      gcitIndex: null,
      conditionValue: 0.0,
      maskValue: 0.0,
      polarity: 'none',
      isIncluded: false,
      inclusionReason: 'Contradictory testimony detected. Blocks generation until explicitly arbitrated and corrected by investigator.',
    };
  }

  // 2. Unknown or unmentioned evidence: Masked to 0
  if (evidenceState === 'unknown' || !effectiveValue) {
    return {
      isEligible: false,
      gcitDimension: null,
      gcitIndex: null,
      conditionValue: 0.0,
      maskValue: 0.0,
      polarity: 'none',
      isIncluded: false,
      inclusionReason: 'Not observed or unmentioned in witness statement. Masked to 0.0 (unconditioned). Feature is drawn from empirical prior.',
    };
  }

  // 3. Uncertain / Hedged evidence: Explicitly masked to 0 to prevent cognitive anchoring
  if (evidenceState === 'uncertain') {
    return {
      isEligible: false,
      gcitDimension: null,
      gcitIndex: null,
      conditionValue: 0.0,
      maskValue: 0.0,
      polarity: 'none',
      isIncluded: false,
      inclusionReason: 'Witness expressed uncertainty / epistemic hedge ("maybe", "possibly", "not sure"). Masked to 0.0 to prevent cognitive bias.',
    };
  }

  // 4. Category-specific mapping logic (matches Python gcit_translator.py exactly)

  // A. Facial Hair
  if (normCat === 'facial_hair' || normCat === 'beard' || normCat === 'mustache') {
    if (evidenceState === 'negated') {
      if (val.includes('mustache') || val.includes('no mustache')) {
        return {
          isEligible: true,
          gcitDimension: 'facial_mustache',
          gcitIndex: 9,
          conditionValue: -1.0,
          maskValue: 1.0,
          polarity: 'normal',
          isIncluded: true,
          inclusionReason: 'Witness explicitly negated mustache. Conditioned with -1.0 on normal CelebA dimension "facial_mustache" to suppress mustache.',
        };
      }
      // "no beard", "clean-shaven", "bare face" -> Inverted No_Beard dimension (+1 enforces absence!)
      return {
        isEligible: true,
        gcitDimension: 'facial_no_beard',
        gcitIndex: 13,
        conditionValue: 1.0,
        maskValue: 1.0,
        polarity: 'inverted',
        isIncluded: true,
        inclusionReason: 'Witness negated facial hair (clean-shaven / no beard). Conditioned with +1.0 on inverted CelebA dimension "facial_no_beard" to enforce beard absence.',
      };
    }

    if (evidenceState === 'explicit') {
      if (val.includes('clean') || val.includes('shaven') || val.includes('no beard') || val.includes('bare')) {
        return {
          isEligible: true,
          gcitDimension: 'facial_no_beard',
          gcitIndex: 13,
          conditionValue: 1.0,
          maskValue: 1.0,
          polarity: 'inverted',
          isIncluded: true,
          inclusionReason: 'Witness explicitly affirmed clean-shaven (absence of beard). Conditioned with +1.0 on inverted CelebA dimension "facial_no_beard" to enforce beard absence.',
        };
      }
      if (val.includes('mustache')) {
        return {
          isEligible: true,
          gcitDimension: 'facial_mustache',
          gcitIndex: 9,
          conditionValue: 1.0,
          maskValue: 1.0,
          polarity: 'normal',
          isIncluded: true,
          inclusionReason: 'Witness affirmed mustache. Conditioned with +1.0 on normal dimension "facial_mustache".',
        };
      }
      if (val.includes('goatee')) {
        return {
          isEligible: true,
          gcitDimension: 'facial_goatee',
          gcitIndex: 10,
          conditionValue: 1.0,
          maskValue: 1.0,
          polarity: 'normal',
          isIncluded: true,
          inclusionReason: 'Witness affirmed goatee. Conditioned with +1.0 on normal dimension "facial_goatee".',
        };
      }
      if (val.includes('sideburn')) {
        return {
          isEligible: true,
          gcitDimension: 'facial_sideburns',
          gcitIndex: 11,
          conditionValue: 1.0,
          maskValue: 1.0,
          polarity: 'normal',
          isIncluded: true,
          inclusionReason: 'Witness affirmed prominent sideburns. Conditioned with +1.0 on normal dimension "facial_sideburns".',
        };
      }
      if (val.includes('stubble') || val.includes('shadow')) {
        return {
          isEligible: true,
          gcitDimension: 'facial_stubble',
          gcitIndex: 12,
          conditionValue: 1.0,
          maskValue: 1.0,
          polarity: 'normal',
          isIncluded: true,
          inclusionReason: 'Witness affirmed 5 o\'clock shadow / stubble. Conditioned with +1.0 on normal dimension "facial_stubble".',
        };
      }
      if (val.includes('beard') && !val.includes('no beard')) {
        return {
          isEligible: true,
          gcitDimension: 'facial_no_beard',
          gcitIndex: 13,
          conditionValue: -1.0,
          maskValue: 1.0,
          polarity: 'inverted',
          isIncluded: true,
          inclusionReason: 'Witness affirmed beard. Conditioned with -1.0 on inverted dimension "facial_no_beard" (advisory: full beard styles may vary).',
        };
      }
    }
  }

  // B. Hair
  if (normCat === 'hair' || normCat === 'hair_color' || normCat === 'hair_style') {
    if (evidenceState === 'negated') {
      if (val.includes('bald')) {
        return {
          isEligible: true,
          gcitDimension: 'hair_bald',
          gcitIndex: 4,
          conditionValue: -1.0,
          maskValue: 1.0,
          polarity: 'normal',
          isIncluded: true,
          inclusionReason: 'Witness explicitly negated baldness. Conditioned with -1.0 on normal dimension "hair_bald" to suppress baldness.',
        };
      }
    }

    if (evidenceState === 'explicit') {
      if (val.includes('bald')) {
        return {
          isEligible: true,
          gcitDimension: 'hair_bald',
          gcitIndex: 4,
          conditionValue: 1.0,
          maskValue: 1.0,
          polarity: 'normal',
          isIncluded: true,
          inclusionReason: 'Witness affirmed baldness. Conditioned with +1.0 on normal dimension "hair_bald".',
        };
      }
      if (val.includes('black')) {
        return {
          isEligible: true,
          gcitDimension: 'hair_black',
          gcitIndex: 0,
          conditionValue: 1.0,
          maskValue: 1.0,
          polarity: 'normal',
          isIncluded: true,
          inclusionReason: 'Witness affirmed black hair. Conditioned with +1.0 on normal dimension "hair_black".',
        };
      }
      if (val.includes('brown')) {
        return {
          isEligible: true,
          gcitDimension: 'hair_brown',
          gcitIndex: 1,
          conditionValue: 1.0,
          maskValue: 1.0,
          polarity: 'normal',
          isIncluded: true,
          inclusionReason: 'Witness affirmed brown hair. Conditioned with +1.0 on normal dimension "hair_brown".',
        };
      }
      if (val.includes('blond') || val.includes('blonde')) {
        return {
          isEligible: true,
          gcitDimension: 'hair_blond',
          gcitIndex: 2,
          conditionValue: 1.0,
          maskValue: 1.0,
          polarity: 'normal',
          isIncluded: true,
          inclusionReason: 'Witness affirmed blond hair. Conditioned with +1.0 on normal dimension "hair_blond".',
        };
      }
      if (val.includes('gray') || val.includes('grey') || val.includes('white')) {
        return {
          isEligible: true,
          gcitDimension: 'hair_gray',
          gcitIndex: 3,
          conditionValue: 1.0,
          maskValue: 1.0,
          polarity: 'normal',
          isIncluded: true,
          inclusionReason: 'Witness affirmed gray/white hair. Conditioned with +1.0 on normal dimension "hair_gray".',
        };
      }
      if (val.includes('straight')) {
        return {
          isEligible: true,
          gcitDimension: 'hair_straight',
          gcitIndex: 6,
          conditionValue: 1.0,
          maskValue: 1.0,
          polarity: 'normal',
          isIncluded: true,
          inclusionReason: 'Witness affirmed straight hair. Conditioned with +1.0 on normal dimension "hair_straight".',
        };
      }
      if (val.includes('wavy') || val.includes('curly')) {
        return {
          isEligible: true,
          gcitDimension: 'hair_wavy',
          gcitIndex: 7,
          conditionValue: 1.0,
          maskValue: 1.0,
          polarity: 'normal',
          isIncluded: true,
          inclusionReason: 'Witness affirmed wavy hair. Conditioned with +1.0 on normal dimension "hair_wavy".',
        };
      }
      if (val.includes('bang') || val.includes('fringe')) {
        return {
          isEligible: true,
          gcitDimension: 'hair_bangs',
          gcitIndex: 5,
          conditionValue: 1.0,
          maskValue: 1.0,
          polarity: 'normal',
          isIncluded: true,
          inclusionReason: 'Witness affirmed bangs / fringe. Conditioned with +1.0 on normal dimension "hair_bangs".',
        };
      }
    }
  }

  // C. Hairline
  if (normCat === 'hairline') {
    if (evidenceState === 'explicit' && val.includes('receding')) {
      return {
        isEligible: true,
        gcitDimension: 'hairline_receding',
        gcitIndex: 8,
        conditionValue: 1.0,
        maskValue: 1.0,
        polarity: 'normal',
        isIncluded: true,
        inclusionReason: 'Witness affirmed receding hairline. Conditioned with +1.0 on normal dimension "hairline_receding".',
      };
    }
  }

  // D. Nose
  if (normCat === 'nose' || normCat === 'nose_shape') {
    if (evidenceState === 'explicit') {
      if (val.includes('big') || val.includes('large') || val.includes('broad')) {
        return {
          isEligible: true,
          gcitDimension: 'nose_big',
          gcitIndex: 14,
          conditionValue: 1.0,
          maskValue: 1.0,
          polarity: 'normal',
          isIncluded: true,
          inclusionReason: 'Witness affirmed large/broad nose. Conditioned with +1.0 on normal dimension "nose_big".',
        };
      }
      if (val.includes('pointy') || val.includes('sharp') || val.includes('pointed')) {
        return {
          isEligible: true,
          gcitDimension: 'nose_pointy',
          gcitIndex: 15,
          conditionValue: 1.0,
          maskValue: 1.0,
          polarity: 'normal',
          isIncluded: true,
          inclusionReason: 'Witness affirmed pointy/sharp nose. Conditioned with +1.0 on normal dimension "nose_pointy".',
        };
      }
    }
    if (evidenceState === 'negated') {
      if (val.includes('big') || val.includes('large')) {
        return {
          isEligible: true,
          gcitDimension: 'nose_big',
          gcitIndex: 14,
          conditionValue: -1.0,
          maskValue: 1.0,
          polarity: 'normal',
          isIncluded: true,
          inclusionReason: 'Witness negated large nose. Conditioned with -1.0 on normal dimension "nose_big" to suppress large nose.',
        };
      }
    }
  }

  // E. Lips
  if (normCat === 'lips' || normCat === 'lip_shape') {
    if (evidenceState === 'explicit' && (val.includes('big') || val.includes('thick') || val.includes('full'))) {
      return {
        isEligible: true,
        gcitDimension: 'lips_big',
        gcitIndex: 16,
        conditionValue: 1.0,
        maskValue: 1.0,
        polarity: 'normal',
        isIncluded: true,
        inclusionReason: 'Witness affirmed thick/full lips. Conditioned with +1.0 on normal dimension "lips_big".',
      };
    }
    if (evidenceState === 'negated' && (val.includes('big') || val.includes('thick') || val.includes('full'))) {
      return {
        isEligible: true,
        gcitDimension: 'lips_big',
        gcitIndex: 16,
        conditionValue: -1.0,
        maskValue: 1.0,
        polarity: 'normal',
        isIncluded: true,
        inclusionReason: 'Witness negated thick lips. Conditioned with -1.0 on normal dimension "lips_big" to suppress full lips.',
      };
    }
  }

  // F. Face Shape
  if (normCat === 'face_shape') {
    if (evidenceState === 'explicit') {
      if (val.includes('oval')) {
        return {
          isEligible: true,
          gcitDimension: 'face_oval',
          gcitIndex: 17,
          conditionValue: 1.0,
          maskValue: 1.0,
          polarity: 'normal',
          isIncluded: true,
          inclusionReason: 'Witness affirmed oval face shape. Conditioned with +1.0 on normal dimension "face_oval".',
        };
      }
      if (val.includes('cheekbone') || val.includes('high cheekbone')) {
        return {
          isEligible: true,
          gcitDimension: 'face_cheekbones',
          gcitIndex: 18,
          conditionValue: 1.0,
          maskValue: 1.0,
          polarity: 'normal',
          isIncluded: true,
          inclusionReason: 'Witness affirmed high cheekbones. Conditioned with +1.0 on normal dimension "face_cheekbones".',
        };
      }
    }
  }

  // G. Accessories / Eyeglasses
  if (normCat === 'other_observable' || normCat === 'accessories') {
    if (val.includes('glass')) {
      if (evidenceState === 'explicit') {
        return {
          isEligible: true,
          gcitDimension: 'glasses',
          gcitIndex: 19,
          conditionValue: 1.0,
          maskValue: 1.0,
          polarity: 'normal',
          isIncluded: true,
          inclusionReason: 'Witness affirmed eyeglasses. Conditioned with +1.0 on normal dimension "glasses".',
        };
      }
      if (evidenceState === 'negated') {
        return {
          isEligible: true,
          gcitDimension: 'glasses',
          gcitIndex: 19,
          conditionValue: -1.0,
          maskValue: 1.0,
          polarity: 'normal',
          isIncluded: true,
          inclusionReason: 'Witness negated eyeglasses. Conditioned with -1.0 on normal dimension "glasses" to actively suppress glasses.',
        };
      }
    }
    return {
      isEligible: false,
      gcitDimension: null,
      gcitIndex: null,
      conditionValue: 0.0,
      maskValue: 0.0,
      polarity: 'none',
      isIncluded: false,
      inclusionReason: 'Non-facial observable (clothing/scene). Intentionally isolated from facial synthesis conditioning.',
    };
  }

  // H. Unsupported Categories (Eyes, Distinguishing Marks, etc.)
  if (normCat === 'eyes' || normCat === 'eye_color') {
    return {
      isEligible: false,
      gcitDimension: null,
      gcitIndex: null,
      conditionValue: 0.0,
      maskValue: 0.0,
      polarity: 'none',
      isIncluded: false,
      inclusionReason: 'Eye color is unannotated in CelebA generator condition space. Iris pigmentation is drawn from empirical model prior.',
    };
  }

  if (normCat === 'distinguishing_marks') {
    return {
      isEligible: false,
      gcitDimension: null,
      gcitIndex: null,
      conditionValue: 0.0,
      maskValue: 0.0,
      polarity: 'none',
      isIncluded: false,
      inclusionReason: 'Scars, tattoos, and birthmarks are unannotated in CelebA generator condition space. Feature is unconditioned.',
    };
  }

  // Fallback for unsupported / unrecognized feature
  return {
    isEligible: false,
    gcitDimension: null,
    gcitIndex: null,
    conditionValue: 0.0,
    maskValue: 0.0,
    polarity: 'none',
    isIncluded: false,
    inclusionReason: `Descriptor "${val}" under category "${category}" does not map to any of the 20 supported GCIT dimensions. Feature remains unconditioned.`,
  };
};

/**
 * Builds the complete 20-dimensional GCIT condition vector and binary mask from trace items.
 */
export const buildGCITTensorsFromTrace = (
  traceItems: IGCITTraceItem[]
): { cVector: number[]; mVector: number[] } => {
  const cVector = new Array(20).fill(0.0);
  const mVector = new Array(20).fill(0.0);

  for (const item of traceItems) {
    if (item.gcitStatus.isIncluded && typeof item.gcitStatus.gcitIndex === 'number') {
      const idx = item.gcitStatus.gcitIndex;
      if (idx >= 0 && idx < 20) {
        cVector[idx] = item.gcitStatus.conditionValue;
        mVector[idx] = item.gcitStatus.maskValue;
      }
    }
  }

  return { cVector, mVector };
};
