import api from "./api";

export const createWitnessDescription = async (data: {
  case: string;
  rawStatement: string;
}) => {
  const response = await api.post("/witness-descriptions", data);
  return response.data;
};

export const analyzeWitness = async (witnessDescriptionId: string) => {
  const response = await api.post(`/ai-analysis/${witnessDescriptionId}`);
  return response.data;
};

export const addWitnessCorrection = async (
  witnessDescriptionId: string,
  correctionData: {
    attributeCategory: string;
    previousValue?: string | null;
    correctedValue: string | null;
    previousEvidenceState?: string;
    reason: string;
    rejected?: boolean;
    correctedEvidenceState?: string;
  }
) => {
  const response = await api.post(
    `/witness-descriptions/${witnessDescriptionId}/corrections`,
    correctionData
  );
  return response.data;
};

export const approveWitnessAnalysis = async (
  witnessDescriptionId: string,
  status: "approved" | "rejected" = "approved"
) => {
  const response = await api.post(
    `/witness-descriptions/${witnessDescriptionId}/approve`,
    { status }
  );
  return response.data;
};

export const generatePrompt = async (witnessDescriptionId: string) => {
  const response = await api.post(`/prompt-generation/${witnessDescriptionId}`);
  return response.data;
};

export const generateImage = async (jobId: string) => {
  const response = await api.post(`/image-generation/${jobId}/generate`);
  return response.data;
};

export const getImageJob = async (jobId: string) => {
  const response = await api.get(`/image-generation/job/${jobId}`);
  return response.data;
};

export const getWitnessDescriptionsByCase = async (caseId: string) => {
  const response = await api.get(`/witness-descriptions/case/${caseId}`);
  return response.data;
};

export interface GenerateCompositePayload {
  witnessDescriptionId: string;
  seed?: number;
  numVariations?: number;
  notes?: string;
}

export const generateCaseComposite = async (caseId: string, payload: GenerateCompositePayload) => {
  const response = await api.post(`/cases/${caseId}/composites/generate`, payload);
  return response.data;
};

export const getCaseComposites = async (caseId: string) => {
  const response = await api.get(`/cases/${caseId}/composites`);
  return response.data;
};

export const getCaseCompositeById = async (caseId: string, compositeId: string) => {
  const response = await api.get(`/cases/${caseId}/composites/${compositeId}`);
  return response.data;
};

export const selectCompositeVariation = async (
  caseId: string,
  compositeId: string,
  variationIndex: number,
  status: 'draft' | 'under_review' | 'approved' | 'superseded' = 'approved',
  notes?: string
) => {
  const response = await api.patch(`/cases/${caseId}/composites/${compositeId}/select`, {
    variationIndex,
    status,
    notes,
  });
  return response.data;
};

export interface LatentDirectionItem {
  attribute_key: string;
  display_name?: string;
  description: string;
  validation_metrics: {
    intended_change: number;
    consistency: number;
    worst_side_effect: {
      attribute: string;
      change: number;
    };
    selectivity_ratio: number;
  };
}

export interface LatentDirectionsInfo {
  num_supported_directions: number;
  num_rejected_directions: number;
  supported_directions: LatentDirectionItem[];
  unsupported_directions: Array<{
    attribute_key: string;
    rejection_reason: string;
    metrics?: Record<string, any>;
  }>;
  unsupported_facial_features_notice: {
    smiling_expression: string;
    continuous_hair_length: string;
    arbitrary_3d_geometry: string;
  };
}

export interface EditCompositePayload {
  selectedSeed: number;
  continuousAdjustments?: Record<string, number>;
  latentTraversals?: Record<string, number>;
  notes?: string;
}

export const getLatentDirections = async (caseId: string) => {
  const response = await api.get(`/cases/${caseId}/composites/latent-directions`);
  return response.data;
};

export const editCaseComposite = async (
  caseId: string,
  compositeId: string,
  payload: EditCompositePayload
) => {
  const response = await api.post(`/cases/${caseId}/composites/${compositeId}/edit`, payload);
  return response.data;
};

export interface EvidenceTraceItem {
  attributeName: string;
  category: string;
  categoryDisplayName: string;
  extractedValue: string | null;
  currentValue: string | null;
  originalWitnessStatement: string;
  exactCharacterOffsets: {
    start: number | null;
    end: number | null;
    matchedSubstring: string | null;
  };
  surroundingContextSentence: string | null;
  extractedEvidenceState: string;
  currentEvidenceState: string;
  nlpProvenance: {
    modelName: string;
    architecture: string;
    checkpointSha256: string;
    extractionConfidence: number | null;
    confidenceNote: string;
  };
  investigatorCorrection: {
    isCorrected: boolean;
    correctedBy: string | null;
    correctedAt: string | null;
    originalValue: string | null;
    originalEvidenceState: string | null;
    rationale: string | null;
  };
  reviewStatus: string;
  gcitTranslation: {
    isSupported: boolean;
    isEligible: boolean;
    dimensionIndex: number | null;
    dimensionName: string | null;
    polarity: "direct" | "inverted" | "none";
    conditionValue: number;
    maskValue: number;
    inclusionStatus: "conditioned" | "masked_out" | "unsupported_attribute" | "contradictory_blocked";
    inclusionReason: string;
  };
  forensicDisclosure: {
    isBiometricIdentification: boolean;
    generativePriorNotice: string;
  };
}

export interface CompositeTraceResponse {
  compositeId: string;
  caseId: string;
  witnessDescriptionId: string;
  version: number;
  modelVersion: string;
  checkpointSha256: string;
  createdAt: string;
  gcitVector: number[];
  gcitMask: number[];
  imageHashes: Array<{ variationIndex: number; sha256: string; filePath?: string }>;
  evidenceTrace: EvidenceTraceItem[];
  disclaimer: string;
}

export interface WitnessDescriptionTraceResponse {
  witnessDescriptionId: string;
  caseId: string;
  rawStatement: string;
  reviewStatus: string;
  reviewedBy: any;
  reviewedAt: string;
  evidenceTrace: EvidenceTraceItem[];
  gcitPreview: {
    gcitVector: number[];
    gcitMask: number[];
    activeDimensions: Array<{
      index: number;
      name: string;
      value: number;
      mask: number;
      polarity: string;
    }>;
  };
  disclaimer: string;
}

export const getCompositeTrace = async (caseId: string, compositeId: string) => {
  const response = await api.get(`/cases/${caseId}/composites/${compositeId}/trace`);
  return response.data;
};

export const getWitnessDescriptionTrace = async (witnessDescriptionId: string) => {
  const response = await api.get(`/witness-descriptions/${witnessDescriptionId}/trace`);
  return response.data;
};

// ============================================================================
// Phase 7.1: Multi-Variation Comparison & Attribute Difference Analysis Types
// ============================================================================

export interface ComparedVariationItem {
  variationKey: string;
  variationId: string | null;
  compositeId: string;
  caseId: string;
  witnessDescriptionId: string | null;
  version: number;
  compositeStatus: "draft" | "under_review" | "approved" | "superseded";
  variationIndex: number;
  variationRole: "standard" | "semantic_edited" | "semantic_baseline";
  seed: number;
  storageKey: string;
  imageUrl: string;
  dataUrl: string | null;
  mimeType: string;
  width: number;
  height: number;
  fileSizeBytes: number;
  sha256: string | null;
  isSelected: boolean;
  selectedAt: string | null;
  selectedBy: {
    _id?: string;
    name?: string;
    email?: string;
    role?: string;
    investigatorId?: string;
  } | string | null;
  selectionNotes: string | null;
  createdAt: string;
  createdBy: any;
  reviewedBy: any;
  reviewedAt: string | null;
  editingType: "initial" | "conditioning_traversal" | "latent_traversal" | "hybrid_traversal";
  parentCompositeId: string | null;
  traversalParameters: {
    continuousAdjustments: Record<string, number>;
    latentTraversals: Array<{ attribute: string; alpha: number }>;
  };
  generatorMetadata: Record<string, any>;
  gcitVector: number[];
  gcitMask: number[];
  conditionedCount: number;
  maskedCount: number;
}

export interface GCITDimensionPerVariation {
  variationKey: string;
  compositeId: string;
  version: number;
  variationIndex: number;
  conditionValue: number;
  maskValue: number;
  status: "conditioned" | "masked_uncertain" | "masked_unknown" | "contradictory";
  recordedAttributeValue: string | null;
  evidenceState: "explicit" | "uncertain" | "negated" | "contradictory" | "unknown";
  isContinuouslyAdjusted: boolean;
  latentTraversalAlpha: number | null;
  explanation: string;
}

export interface GCITDimensionComparisonRow {
  index: number;
  dimensionName: string;
  category: string;
  categoryDisplayName: string;
  description: string;
  polarity: "normal" | "inverted";
  hasConditionDifference: boolean;
  hasMaskDifference: boolean;
  hasLatentDifference: boolean;
  hasStateDifference: boolean;
  hasAnyDifference: boolean;
  differenceSummary: string;
  perVariation: GCITDimensionPerVariation[];
}

export interface RecordedAttributePerVariation {
  variationKey: string;
  compositeId: string;
  version: number;
  variationIndex: number;
  recordedValue: string | null;
  originalValue: string | null;
  evidenceState: "explicit" | "uncertain" | "negated" | "contradictory" | "unknown";
  classification:
    | "conditioned"
    | "masked_uncertain"
    | "masked_unmentioned"
    | "unsupported_attribute"
    | "contradictory_blocked";
  isSupportedInGCIT: boolean;
  gcitDimension: string | null;
  gcitIndex: number | null;
  conditionValue: number;
  maskValue: number;
  isCorrected: boolean;
  rationale: string;
}

export interface RecordedAttributeComparisonRow {
  category: string;
  categoryDisplayName: string;
  isSupportedInGCIT: boolean;
  hasDifference: boolean;
  perVariation: RecordedAttributePerVariation[];
}

export interface SelectionHistoryEntry {
  _id?: string;
  compositeId: string;
  version: number;
  variationIndex: number;
  variationId?: string;
  seed: number;
  previousVariationIndex: number | null;
  selectedBy: {
    _id?: string;
    name?: string;
    email?: string;
    role?: string;
    investigatorId?: string;
  } | string;
  selectedAt: string;
  status: string;
  notes?: string;
}

export interface VariationComparisonResponse {
  caseId: string;
  comparedAt: string;
  variations: ComparedVariationItem[];
  dimensionComparison: GCITDimensionComparisonRow[];
  recordedAttributesComparison: RecordedAttributeComparisonRow[];
  selectionHistory: SelectionHistoryEntry[];
  summary: {
    totalVariationsCompared: number;
    sameCompositeSession: boolean;
    compositeIds: string[];
    differingGCITDimensionsCount: number;
    differingGCITDimensions: string[];
    differingRecordedAttributesCount: number;
    seedsCompared: number[];
    seedDifferenceNote: string;
  };
  disclosures: {
    unconditionedPriorDisclosure: string;
    noPixelInferenceDisclosure: string;
    nonBiometricSelectionDisclosure: string;
  };
}

export const compareSingleCompositeVariations = async (
  caseId: string,
  compositeId: string,
  variationIndices?: number[]
) => {
  const query =
    variationIndices && variationIndices.length > 0
      ? `?variations=${variationIndices.join(",")}`
      : "";
  const response = await api.get(`/cases/${caseId}/composites/${compositeId}/compare${query}`);
  return response.data;
};

export const compareCaseCompositeVariations = async (
  caseId: string,
  payload: {
    compositeId?: string;
    variationIndices?: number[];
    targets?: Array<{ compositeId: string; variationIndex: number }>;
  }
) => {
  const response = await api.post(`/cases/${caseId}/composites/compare`, payload);
  return response.data;
};



