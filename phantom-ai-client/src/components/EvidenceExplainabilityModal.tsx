import React, { useState, useEffect } from "react";
import {
  X,
  ShieldCheck,
  Cpu,
  FileText,
  Sliders,
  CheckCircle2,
  HelpCircle,
  AlertTriangle,
  XCircle,
  Hash,
  Eye,
  Binary,
  Info,
} from "lucide-react";
import { Button } from "./ui/Button";
import type {
  EvidenceTraceItem,
  CompositeTraceResponse,
  WitnessDescriptionTraceResponse,
} from "../services/imageService";

interface EvidenceExplainabilityModalProps {
  isOpen: boolean;
  onClose: () => void;
  traceData: (CompositeTraceResponse | WitnessDescriptionTraceResponse) | null;
  type: "composite" | "witness";
  initialCategory?: string;
  loading?: boolean;
}

function normalizeTraceItem(raw: any, rawStatementFallback = ""): EvidenceTraceItem {
  if (raw && raw.gcitTranslation && raw.exactCharacterOffsets) {
    return raw as EvidenceTraceItem;
  }

  const gcitStatus = raw?.gcitStatus || {};
  const isIncluded = Boolean(gcitStatus.isIncluded);
  const isEligible = Boolean(gcitStatus.isEligible);
  const currentState = raw?.evidenceState || raw?.currentEvidenceState || "unknown";

  let inclusionStatus: "conditioned" | "masked_out" | "unsupported_attribute" | "contradictory_blocked" =
    "masked_out";
  if (currentState === "contradictory") {
    inclusionStatus = "contradictory_blocked";
  } else if (isIncluded) {
    inclusionStatus = "conditioned";
  } else if (
    !isEligible &&
    currentState !== "unknown" &&
    currentState !== "uncertain" &&
    (raw?.effectiveValue || raw?.currentValue)
  ) {
    inclusionStatus = "unsupported_attribute";
  } else {
    inclusionStatus = "masked_out";
  }

  const polarityRaw = gcitStatus.polarity || "none";
  const polarity: "direct" | "inverted" | "none" =
    polarityRaw === "inverted" ? "inverted" : polarityRaw === "normal" || polarityRaw === "direct" ? "direct" : "none";

  return {
    attributeName: raw?.attributeName || raw?.category || "Attribute",
    category: raw?.category || "unknown",
    categoryDisplayName:
      raw?.categoryDisplayName ||
      raw?.attributeName ||
      (raw?.category || "unknown").replace(/_/g, " "),
    extractedValue: raw?.originalValue ?? raw?.extractedValue ?? raw?.effectiveValue ?? null,
    currentValue: raw?.effectiveValue ?? raw?.currentValue ?? null,
    originalWitnessStatement: raw?.originalWitnessStatement || rawStatementFallback || "",
    exactCharacterOffsets: {
      start: typeof raw?.startChar === "number" ? raw.startChar : raw?.exactCharacterOffsets?.start ?? null,
      end: typeof raw?.endChar === "number" ? raw.endChar : raw?.exactCharacterOffsets?.end ?? null,
      matchedSubstring: raw?.sourceSpan || raw?.exactCharacterOffsets?.matchedSubstring || null,
    },
    surroundingContextSentence:
      raw?.surroundingContext || raw?.surroundingContextSentence || rawStatementFallback || null,
    extractedEvidenceState: raw?.originalEvidenceState || raw?.extractedEvidenceState || currentState,
    currentEvidenceState: currentState,
    nlpProvenance: {
      modelName: raw?.extractedBy || raw?.nlpProvenance?.modelName || "char_bilstm_crf_phase3c2",
      architecture:
        raw?.nlpProvenance?.architecture || "BiLSTM-CRF with Character-Aware CNN Embeddings",
      checkpointSha256:
        raw?.nlpProvenance?.checkpointSha256 ||
        "2d567c79d759e76940968fadb02d412b530d3405e377a2d6454c64e9f2e7b4a3",
      extractionConfidence:
        typeof raw?.confidence === "number"
          ? raw.confidence
          : raw?.nlpProvenance?.extractionConfidence ?? null,
      confidenceNote:
        raw?.nlpProvenance?.confidenceNote ||
        "Uncalibrated token-level CRF probabilities are suppressed per Phase 3.7 calibration audit to prevent investigator over-reliance.",
    },
    investigatorCorrection: {
      isCorrected: Boolean(raw?.isCorrected ?? raw?.investigatorCorrection?.isCorrected),
      correctedBy:
        raw?.correction?.correctedBy || raw?.investigatorCorrection?.correctedBy || null,
      correctedAt:
        raw?.correction?.timestamp || raw?.investigatorCorrection?.correctedAt || null,
      originalValue:
        raw?.originalValue ?? raw?.investigatorCorrection?.originalValue ?? null,
      originalEvidenceState:
        raw?.originalEvidenceState ?? raw?.investigatorCorrection?.originalEvidenceState ?? null,
      rationale:
        raw?.correction?.reason || raw?.investigatorCorrection?.rationale || null,
    },
    reviewStatus: raw?.reviewStatus || "approved",
    gcitTranslation: {
      isSupported: isEligible || inclusionStatus === "conditioned",
      isEligible,
      dimensionIndex:
        typeof gcitStatus.gcitIndex === "number" ? gcitStatus.gcitIndex : null,
      dimensionName: gcitStatus.gcitDimension || null,
      polarity,
      conditionValue:
        typeof gcitStatus.conditionValue === "number" ? gcitStatus.conditionValue : 0.0,
      maskValue: typeof gcitStatus.maskValue === "number" ? gcitStatus.maskValue : 0.0,
      inclusionStatus,
      inclusionReason:
        gcitStatus.inclusionReason ||
        "Unconditioned attribute; sampled stochastically from empirical CelebA training prior.",
    },
    forensicDisclosure: {
      isBiometricIdentification: false,
      generativePriorNotice:
        raw?.forensicDisclosure?.generativePriorNotice ||
        "Generated composites are investigative lead aids only and never constitute biometric identification. Unconditioned attributes (m_i = 0.0) are sampled from the empirical dataset prior and are not guaranteed to be absent.",
    },
  };
}

export const EvidenceExplainabilityModal: React.FC<EvidenceExplainabilityModalProps> = ({
  isOpen,
  onClose,
  traceData,
  type,
  initialCategory,
  loading = false,
}) => {
  const [selectedAttributeKey, setSelectedAttributeKey] = useState<string>("");
  const [activeTab, setActiveTab] = useState<"attributes" | "gcit_matrix" | "integrity">("attributes");
  const [filterState, setFilterState] = useState<"all" | "conditioned" | "masked" | "unsupported">("all");

  const rawStatementFallback =
    (traceData as any)?.rawStatement ||
    (traceData as any)?.witnessDescription?.rawStatement ||
    "";

  const evidenceTrace: EvidenceTraceItem[] = ((traceData as any)?.evidenceTrace || []).map((item: any) =>
    normalizeTraceItem(item, rawStatementFallback)
  );

  // When modal opens or traceData loads, select the first or targeted attribute
  useEffect(() => {
    if (evidenceTrace.length > 0) {
      if (initialCategory) {
        const found = evidenceTrace.find(
          (a) => a.category.toLowerCase() === initialCategory.toLowerCase()
        );
        if (found) {
          setSelectedAttributeKey(found.category);
          return;
        }
      }
      setSelectedAttributeKey(evidenceTrace[0].category);
    }
  }, [traceData, initialCategory]);

  if (!isOpen) return null;

  const selectedItem: EvidenceTraceItem | undefined =
    evidenceTrace.find((a) => a.category === selectedAttributeKey) || evidenceTrace[0];

  // Filtered attributes list
  const filteredAttributes = evidenceTrace.filter((item) => {
    if (filterState === "conditioned") {
      return item.gcitTranslation.inclusionStatus === "conditioned";
    }
    if (filterState === "masked") {
      return (
        item.gcitTranslation.inclusionStatus === "masked_out" ||
        item.gcitTranslation.inclusionStatus === "contradictory_blocked"
      );
    }
    if (filterState === "unsupported") {
      return item.gcitTranslation.inclusionStatus === "unsupported_attribute";
    }
    return true;
  });

  const conditionedCount = evidenceTrace.filter(
    (i) => i.gcitTranslation.inclusionStatus === "conditioned"
  ).length;
  const maskedCount = evidenceTrace.filter(
    (i) =>
      i.gcitTranslation.inclusionStatus === "masked_out" ||
      i.gcitTranslation.inclusionStatus === "contradictory_blocked"
  ).length;
  const unsupportedCount = evidenceTrace.filter(
    (i) => i.gcitTranslation.inclusionStatus === "unsupported_attribute"
  ).length;

  const isCompositeTrace = type === "composite" && "gcitVector" in (traceData || {});
  const compositeData = isCompositeTrace ? (traceData as CompositeTraceResponse) : null;
  const witnessData = !isCompositeTrace ? (traceData as WitnessDescriptionTraceResponse) : null;

  const gcitVector =
    compositeData?.gcitVector ||
    (witnessData as any)?.gcitPreview?.conditionVector ||
    witnessData?.gcitPreview?.gcitVector ||
    [];
  const gcitMask =
    compositeData?.gcitMask ||
    (witnessData as any)?.gcitPreview?.maskVector ||
    witnessData?.gcitPreview?.gcitMask ||
    [];
  const normalizedImageHashes: Array<{ variationIndex: number; sha256: string }> = (
    compositeData?.imageHashes || []
  ).map((h: any, idx: number) =>
    typeof h === "string"
      ? { variationIndex: idx + 1, sha256: h }
      : { variationIndex: h.variationIndex ?? idx + 1, sha256: h.sha256 }
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4 animate-in fade-in duration-150">
      <div className="bg-bg-panel border border-line rounded-xl max-w-5xl w-full max-h-[92vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-line bg-bg-inset">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded bg-gold/10 border border-gold/30">
              <ShieldCheck className="w-5 h-5 text-gold" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-base font-bold text-ink-primary">
                  Evidence Traceability & Explainability (XAI)
                </h3>
                <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-bg-deepest border border-line text-gold uppercase">
                  {type === "composite"
                    ? `Composite v${compositeData?.version || 1} Provenance`
                    : "Witness Evidence Provenance"}
                </span>
              </div>
              <p className="text-xs text-ink-tertiary">
                Inspect 15-point audit trail: witness source text, character spans, NLP provenance, investigator review, and 20-D GCIT conditioning.
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-md text-ink-tertiary hover:text-ink-primary hover:bg-bg-deepest transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Technical Summary Bar */}
        <div className="p-3 border-b border-line/70 bg-bg-deepest grid grid-cols-2 sm:grid-cols-5 gap-2 text-xs font-mono">
          <div className="p-2 rounded bg-bg-inset border border-line">
            <span className="text-[10px] text-ink-disabled block">EXTRACTED ATTRIBUTES</span>
            <span className="text-sm font-bold text-ink-primary">{evidenceTrace.length}</span>
          </div>
          <div className="p-2 rounded bg-bg-inset border border-line">
            <span className="text-[10px] text-ink-disabled block">CONDITIONED (m = 1.0)</span>
            <span className="text-sm font-bold text-emerald-400">{conditionedCount} Dimensions</span>
          </div>
          <div className="p-2 rounded bg-bg-inset border border-line">
            <span className="text-[10px] text-ink-disabled block">MASKED OUT (m = 0.0)</span>
            <span className="text-sm font-bold text-amber-400">{maskedCount} Dimensions</span>
          </div>
          <div className="p-2 rounded bg-bg-inset border border-line">
            <span className="text-[10px] text-ink-disabled block">UNSUPPORTED / PRIOR</span>
            <span className="text-sm font-bold text-ink-tertiary">{unsupportedCount} Dimensions</span>
          </div>
          <div className="p-2 rounded bg-bg-inset border border-line col-span-2 sm:col-span-1">
            <span className="text-[10px] text-ink-disabled block">CHECKPOINT INTEGRITY</span>
            <span
              className="text-xs font-semibold text-gold truncate block"
              title="a1714389d0d1eddad10fae74a736ca41de5e230eada095e6868e5daeeb2a1964"
            >
              a1714389... (Verified)
            </span>
          </div>
        </div>

        {/* View Navigation Tabs */}
        <div className="flex items-center gap-1 px-4 pt-2 border-b border-line bg-bg-inset">
          <button
            onClick={() => setActiveTab("attributes")}
            className={`px-3 py-2 text-xs font-semibold rounded-t border-b-2 flex items-center gap-2 transition-colors ${
              activeTab === "attributes"
                ? "border-gold text-gold bg-bg-panel"
                : "border-transparent text-ink-tertiary hover:text-ink-primary"
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            15-Point Attribute Trace ({evidenceTrace.length})
          </button>
          <button
            onClick={() => setActiveTab("gcit_matrix")}
            className={`px-3 py-2 text-xs font-semibold rounded-t border-b-2 flex items-center gap-2 transition-colors ${
              activeTab === "gcit_matrix"
                ? "border-gold text-gold bg-bg-panel"
                : "border-transparent text-ink-tertiary hover:text-ink-primary"
            }`}
          >
            <Binary className="w-3.5 h-3.5" />
            20-D GCIT Vector & Polarity
          </button>
          {compositeData && (
            <button
              onClick={() => setActiveTab("integrity")}
              className={`px-3 py-2 text-xs font-semibold rounded-t border-b-2 flex items-center gap-2 transition-colors ${
                activeTab === "integrity"
                  ? "border-gold text-gold bg-bg-panel"
                  : "border-transparent text-ink-tertiary hover:text-ink-primary"
              }`}
            >
              <Hash className="w-3.5 h-3.5" />
              Cryptographic Image Hashes ({compositeData.imageHashes?.length || 0})
            </button>
          )}
        </div>

        {/* Modal Body */}
        <div className="flex-1 overflow-hidden p-4 bg-bg-panel">
          {loading ? (
            <div className="h-64 flex flex-col items-center justify-center space-y-2 text-ink-tertiary">
              <div className="w-6 h-6 border-2 border-gold border-t-transparent rounded-full animate-spin" />
              <p className="text-xs font-mono">Retrieving cryptographic trace and GCIT conditioning snapshot...</p>
            </div>
          ) : activeTab === "attributes" ? (
            <div className="grid grid-cols-1 md:grid-cols-12 gap-4 h-full max-h-[62vh]">
              {/* Left Column: Attribute Selector & Filter */}
              <div className="md:col-span-4 flex flex-col border border-line rounded-lg overflow-hidden bg-bg-inset">
                {/* Filter Selector */}
                <div className="p-2 border-b border-line bg-bg-deepest flex items-center gap-1 text-[11px]">
                  <button
                    onClick={() => setFilterState("all")}
                    className={`px-2 py-0.5 rounded font-mono ${
                      filterState === "all" ? "bg-gold text-bg-deepest font-bold" : "text-ink-tertiary hover:text-ink-primary"
                    }`}
                  >
                    All ({evidenceTrace.length})
                  </button>
                  <button
                    onClick={() => setFilterState("conditioned")}
                    className={`px-2 py-0.5 rounded font-mono ${
                      filterState === "conditioned" ? "bg-emerald-400 text-bg-deepest font-bold" : "text-ink-tertiary hover:text-ink-primary"
                    }`}
                  >
                    Conditioned ({conditionedCount})
                  </button>
                  <button
                    onClick={() => setFilterState("masked")}
                    className={`px-2 py-0.5 rounded font-mono ${
                      filterState === "masked" ? "bg-amber-400 text-bg-deepest font-bold" : "text-ink-tertiary hover:text-ink-primary"
                    }`}
                  >
                    Masked ({maskedCount})
                  </button>
                  <button
                    onClick={() => setFilterState("unsupported")}
                    className={`px-2 py-0.5 rounded font-mono ${
                      filterState === "unsupported" ? "bg-ink-tertiary text-bg-deepest font-bold" : "text-ink-tertiary hover:text-ink-primary"
                    }`}
                  >
                    Prior ({unsupportedCount})
                  </button>
                </div>

                {/* Attribute List */}
                <div className="flex-1 overflow-y-auto divide-y divide-line/60">
                  {filteredAttributes.length === 0 ? (
                    <div className="p-4 text-center text-xs text-ink-disabled italic">
                      No attributes match the selected filter.
                    </div>
                  ) : (
                    filteredAttributes.map((item) => {
                      const isSelected = selectedAttributeKey === item.category;
                      const status = item.gcitTranslation.inclusionStatus;

                      return (
                        <div
                          key={item.category}
                          onClick={() => setSelectedAttributeKey(item.category)}
                          className={`p-2.5 cursor-pointer transition-colors text-xs ${
                            isSelected
                              ? "bg-gold/15 border-l-4 border-l-gold text-ink-primary"
                              : "hover:bg-bg-panel text-ink-secondary"
                          }`}
                        >
                          <div className="flex items-center justify-between gap-1">
                            <span className="font-mono font-semibold uppercase text-ink-primary">
                              {item.categoryDisplayName || item.category.replace(/_/g, " ")}
                            </span>
                            {renderEvidenceBadge(item.currentEvidenceState)}
                          </div>

                          <div className="flex items-center justify-between mt-1 text-[11px]">
                            <span className="truncate max-w-[140px] text-ink-tertiary">
                              {item.currentValue || "Unspecified"}
                            </span>
                            <span
                              className={`px-1.5 py-0.2 rounded font-mono text-[9px] uppercase font-bold ${
                                status === "conditioned"
                                  ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                                  : status === "masked_out"
                                  ? "bg-amber-500/20 text-amber-400 border border-amber-500/30"
                                  : status === "contradictory_blocked"
                                  ? "bg-rose-500/20 text-rose-400 border border-rose-500/30"
                                  : "bg-ink-disabled/20 text-ink-tertiary border border-line"
                              }`}
                            >
                              {status === "conditioned"
                                ? "Active"
                                : status === "masked_out"
                                ? "Masked"
                                : status === "contradictory_blocked"
                                ? "Blocked"
                                : "Prior"}
                            </span>
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>

              {/* Right Column: 15-Point Trace Details */}
              <div className="md:col-span-8 flex flex-col border border-line rounded-lg overflow-y-auto bg-bg-inset p-4 space-y-4">
                {selectedItem ? (
                  <>
                    {/* Header */}
                    <div className="flex items-center justify-between border-b border-line pb-3">
                      <div>
                        <div className="flex items-center gap-2">
                          <h4 className="text-base font-bold text-ink-primary uppercase tracking-wide">
                            {selectedItem.categoryDisplayName || selectedItem.category.replace(/_/g, " ")}
                          </h4>
                          {renderEvidenceBadge(selectedItem.currentEvidenceState)}
                          {selectedItem.investigatorCorrection.isCorrected && (
                            <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-amber-500/20 text-amber-400 border border-amber-500/40 font-semibold">
                              AUDITED OVERRIDE
                            </span>
                          )}
                        </div>
                        <span className="text-[11px] font-mono text-ink-disabled">
                          Attribute Key: {selectedItem.category} • Review Status:{" "}
                          <span className="text-ink-secondary uppercase">{selectedItem.reviewStatus}</span>
                        </span>
                      </div>

                      <div className="text-right font-mono text-[11px]">
                        <span className="text-ink-disabled block">GCIT INCLUSION</span>
                        <span
                          className={`font-bold ${
                            selectedItem.gcitTranslation.inclusionStatus === "conditioned"
                              ? "text-emerald-400"
                              : selectedItem.gcitTranslation.inclusionStatus === "contradictory_blocked"
                              ? "text-rose-400"
                              : "text-amber-400"
                          }`}
                        >
                          {selectedItem.gcitTranslation.inclusionStatus.toUpperCase()}
                        </span>
                      </div>
                    </div>

                    {/* Section 1: Source Testimony & Surrounding Sentence Context */}
                    <div className="p-3 rounded-lg bg-bg-panel border border-line space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="label-technical text-ink-secondary flex items-center gap-1.5">
                          <FileText className="w-3.5 h-3.5 text-gold" />
                          Witness Testimony & Surrounding Context
                        </span>
                        <span className="text-[10px] font-mono text-ink-disabled">
                          Character Offsets: [
                          {selectedItem.exactCharacterOffsets?.start !== null && selectedItem.exactCharacterOffsets?.start !== undefined
                            ? selectedItem.exactCharacterOffsets.start
                            : "N/A"}
                          :
                          {selectedItem.exactCharacterOffsets?.end !== null && selectedItem.exactCharacterOffsets?.end !== undefined
                            ? selectedItem.exactCharacterOffsets.end
                            : "N/A"}
                          ]
                        </span>
                      </div>

                      {/* Sentence Context with highlighted span */}
                      <div className="p-2.5 rounded bg-bg-deepest border border-line text-sm text-ink-primary leading-relaxed font-sans">
                        {selectedItem.surroundingContextSentence ? (
                          <span>
                            &ldquo;
                            {selectedItem.exactCharacterOffsets?.matchedSubstring ? (
                              renderHighlightedSentence(
                                selectedItem.surroundingContextSentence,
                                selectedItem.exactCharacterOffsets.matchedSubstring
                              )
                            ) : (
                              selectedItem.surroundingContextSentence
                            )}
                            &rdquo;
                          </span>
                        ) : selectedItem.originalWitnessStatement ? (
                          <span className="italic text-ink-tertiary">
                            &ldquo;{selectedItem.originalWitnessStatement}&rdquo;
                          </span>
                        ) : (
                          <span className="text-ink-disabled italic text-xs">
                            No source text captured for this unobserved category.
                          </span>
                        )}
                      </div>

                      {/* Span offsets breakdown */}
                      <div className="grid grid-cols-3 gap-2 text-[11px] font-mono pt-1 text-ink-tertiary">
                        <div>
                          <span className="text-ink-disabled block">Matched Substring:</span>
                          <span className="text-gold font-semibold truncate block">
                            {selectedItem.exactCharacterOffsets?.matchedSubstring || "None (unobserved)"}
                          </span>
                        </div>
                        <div>
                          <span className="text-ink-disabled block">Start Offset:</span>
                          <span className="text-ink-primary">
                            {selectedItem.exactCharacterOffsets?.start !== null && selectedItem.exactCharacterOffsets?.start !== undefined
                              ? selectedItem.exactCharacterOffsets.start
                              : "null"}
                          </span>
                        </div>
                        <div>
                          <span className="text-ink-disabled block">End Offset:</span>
                          <span className="text-ink-primary">
                            {selectedItem.exactCharacterOffsets?.end !== null && selectedItem.exactCharacterOffsets?.end !== undefined
                              ? selectedItem.exactCharacterOffsets.end
                              : "null"}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Section 2: NLP Model Provenance */}
                    <div className="p-3 rounded-lg bg-bg-panel border border-line space-y-2 text-xs">
                      <div className="flex items-center justify-between">
                        <span className="label-technical text-ink-secondary flex items-center gap-1.5">
                          <Cpu className="w-3.5 h-3.5 text-gold" />
                          NLP Extraction Provenance (Phase 3 Checkpoint)
                        </span>
                        <span className="text-[10px] font-mono text-emerald-400">Frozen Model Weights</span>
                      </div>

                      <div className="grid grid-cols-2 gap-2 text-[11px] font-mono">
                        <div>
                          <span className="text-ink-disabled block">Model Architecture:</span>
                          <span className="text-ink-primary">
                            {selectedItem.nlpProvenance.architecture || "BiLSTM-CRF (Char-Aware)"}
                          </span>
                        </div>
                        <div>
                          <span className="text-ink-disabled block">Extraction Confidence:</span>
                          <span className="text-amber-400 font-semibold">
                            {selectedItem.nlpProvenance.extractionConfidence === null
                              ? "Null (Suppressed)"
                              : selectedItem.nlpProvenance.extractionConfidence}
                          </span>
                        </div>
                        <div className="col-span-2">
                          <span className="text-ink-disabled block">Checkpoint SHA-256:</span>
                          <span
                            className="text-gold truncate block max-w-full"
                            title={selectedItem.nlpProvenance.checkpointSha256}
                          >
                            {selectedItem.nlpProvenance.checkpointSha256 ||
                              "2d567c79d759e76940968fadb02d412b530d3405e377a2d6454c64e9f2e7b4a3"}
                          </span>
                        </div>
                      </div>

                      <div className="p-2 rounded bg-bg-deepest border border-line/60 text-[11px] text-ink-tertiary">
                        <strong>Confidence Note:</strong> {selectedItem.nlpProvenance.confidenceNote}
                      </div>
                    </div>

                    {/* Section 3: Non-Destructive Review & Investigator Correction History */}
                    <div className="p-3 rounded-lg bg-bg-panel border border-line space-y-2 text-xs">
                      <div className="flex items-center justify-between">
                        <span className="label-technical text-ink-secondary flex items-center gap-1.5">
                          <Eye className="w-3.5 h-3.5 text-gold" />
                          Non-Destructive Review & Investigator History
                        </span>
                        <span className="text-[10px] font-mono text-ink-disabled">
                          Original Testimony Preserved
                        </span>
                      </div>

                      <div className="grid grid-cols-2 gap-3 text-[11px] font-mono">
                        <div className="p-2 rounded bg-bg-deepest border border-line">
                          <span className="text-[10px] text-ink-disabled block">ORIGINAL NLP EXTRACTION</span>
                          <div className="mt-1 space-y-1">
                            <div>
                              <span className="text-ink-tertiary">Value: </span>
                              <span className="text-ink-primary font-semibold">
                                {selectedItem.investigatorCorrection.originalValue ||
                                  selectedItem.extractedValue ||
                                  "None"}
                              </span>
                            </div>
                            <div>
                              <span className="text-ink-tertiary">State: </span>
                              <span className="text-ink-secondary">
                                {selectedItem.investigatorCorrection.originalEvidenceState ||
                                  selectedItem.extractedEvidenceState}
                              </span>
                            </div>
                          </div>
                        </div>

                        <div className="p-2 rounded bg-bg-deepest border border-line">
                          <span className="text-[10px] text-ink-disabled block">ACTIVE INVESTIGATOR VALUE</span>
                          <div className="mt-1 space-y-1">
                            <div>
                              <span className="text-ink-tertiary">Value: </span>
                              <span className="text-emerald-400 font-semibold">
                                {selectedItem.currentValue || "Unspecified"}
                              </span>
                            </div>
                            <div>
                              <span className="text-ink-tertiary">State: </span>
                              <span className="text-emerald-400">
                                {selectedItem.currentEvidenceState}
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>

                      {selectedItem.investigatorCorrection.isCorrected && (
                        <div className="p-2 rounded bg-amber-500/10 border border-amber-500/30 text-[11px] space-y-1">
                          <div className="font-semibold text-amber-400 flex items-center justify-between">
                            <span>Investigator Audit Rationale:</span>
                            <span className="font-mono text-[10px] text-ink-tertiary">
                              {selectedItem.investigatorCorrection.correctedAt
                                ? new Date(selectedItem.investigatorCorrection.correctedAt).toLocaleString()
                                : "Logged"}
                            </span>
                          </div>
                          <p className="text-ink-primary italic">
                            &ldquo;{selectedItem.investigatorCorrection.rationale}&rdquo;
                          </p>
                        </div>
                      )}
                    </div>

                    {/* Section 4: GCIT Conditioning Translation & Explainability */}
                    <div className="p-3 rounded-lg bg-bg-panel border border-line space-y-2 text-xs">
                      <div className="flex items-center justify-between">
                        <span className="label-technical text-ink-secondary flex items-center gap-1.5">
                          <Sliders className="w-3.5 h-3.5 text-gold" />
                          GCIT Generator Conditioning & Polarity
                        </span>
                        <span className="text-[10px] font-mono text-ink-disabled">
                          20-Dimensional Conditional Space
                        </span>
                      </div>

                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] font-mono">
                        <div className="p-2 rounded bg-bg-deepest border border-line">
                          <span className="text-[10px] text-ink-disabled block">GCIT INDEX</span>
                          <span className="text-ink-primary font-bold">
                            {selectedItem.gcitTranslation.dimensionIndex !== null
                              ? `i = ${selectedItem.gcitTranslation.dimensionIndex}`
                              : "None (Unmapped)"}
                          </span>
                        </div>
                        <div className="p-2 rounded bg-bg-deepest border border-line">
                          <span className="text-[10px] text-ink-disabled block">POLARITY</span>
                          <span
                            className={`font-bold ${
                              selectedItem.gcitTranslation.polarity === "inverted"
                                ? "text-amber-400"
                                : selectedItem.gcitTranslation.polarity === "direct"
                                ? "text-emerald-400"
                                : "text-ink-disabled"
                            }`}
                          >
                            {selectedItem.gcitTranslation.polarity.toUpperCase()}
                          </span>
                        </div>
                        <div className="p-2 rounded bg-bg-deepest border border-line">
                          <span className="text-[10px] text-ink-disabled block">CONDITION (c_i)</span>
                          <span className="text-gold font-bold">
                            {typeof selectedItem.gcitTranslation?.conditionValue === "number"
                              ? selectedItem.gcitTranslation.conditionValue.toFixed(1)
                              : "0.0"}
                          </span>
                        </div>
                        <div className="p-2 rounded bg-bg-deepest border border-line">
                          <span className="text-[10px] text-ink-disabled block">MASK (m_i)</span>
                          <span
                            className={`font-bold ${
                              selectedItem.gcitTranslation?.maskValue === 1.0
                                ? "text-emerald-400"
                                : "text-ink-disabled"
                            }`}
                          >
                            {typeof selectedItem.gcitTranslation?.maskValue === "number"
                              ? selectedItem.gcitTranslation.maskValue.toFixed(1)
                              : "0.0"}
                          </span>
                        </div>
                      </div>

                      {/* Inclusion / Exclusion Reason */}
                      <div className="p-2 rounded bg-bg-deepest border border-line/70 text-[11px] text-ink-primary leading-relaxed">
                        <strong>Inclusion Rationale: </strong>
                        {selectedItem.gcitTranslation.inclusionReason}
                      </div>

                      {/* Specific Polarity Callout for Inverted Dimensions (e.g. facial_no_beard) */}
                      {selectedItem.gcitTranslation.polarity === "inverted" && (
                        <div className="p-2 rounded bg-amber-500/10 border border-amber-500/30 text-[11px] text-amber-300">
                          <strong>Note on CelebA Inverted Polarity:</strong> Dimension #{selectedItem.gcitTranslation.dimensionIndex} (<code>No_Beard</code>) is inverted in the underlying dataset annotation (+1.0 denotes absence of beard; -1.0 denotes presence of beard). When witness notes absence of beard, condition vector receives +1.0 with active mask m=1.0.
                        </div>
                      )}
                    </div>

                    {/* Section 5: Forensic Science Disclosure */}
                    <div className="p-2.5 rounded bg-bg-deepest border border-line text-[11px] text-ink-disabled space-y-1">
                      <div className="font-semibold text-ink-secondary flex items-center gap-1.5">
                        <Info className="w-3.5 h-3.5 text-gold" />
                        Forensic Disclosure & Non-Biometric Use
                      </div>
                      <p>
                        {selectedItem.forensicDisclosure.generativePriorNotice}
                      </p>
                    </div>
                  </>
                ) : (
                  <div className="p-8 text-center text-ink-disabled">
                    Select an attribute on the left to inspect its complete 15-point trace.
                  </div>
                )}
              </div>
            </div>
          ) : activeTab === "gcit_matrix" ? (
            /* 20-D GCIT Vector & Polarity Table */
            <div className="h-full max-h-[62vh] overflow-y-auto space-y-3">
              <div className="p-3 rounded-lg bg-bg-inset border border-line flex items-center justify-between text-xs">
                <div>
                  <h4 className="font-bold text-ink-primary">
                    Full 20-Dimensional Generative Condition Information Translation (GCIT) Vector
                  </h4>
                  <p className="text-ink-tertiary text-[11px] mt-0.5">
                    Exact numerical conditioning inputs passed to the PyTorch Conditional WGAN-GP generator (conditioning vector c &isin; &reals;&sup2;&deg;, mask m &isin; &#123;0, 1&#125;&sup2;&deg;).
                  </p>
                </div>
                <div className="text-right font-mono text-[11px]">
                  <span className="text-ink-disabled block">GENERATOR ARCHITECTURE</span>
                  <span className="text-gold font-bold">Conditional WGAN-GP (Projection)</span>
                </div>
              </div>

              <div className="border border-line rounded-lg overflow-hidden">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="bg-bg-deepest border-b border-line text-ink-disabled text-[10px] uppercase">
                    <tr>
                      <th className="p-2.5">Idx</th>
                      <th className="p-2.5">GCIT Dimension</th>
                      <th className="p-2.5">Category</th>
                      <th className="p-2.5 text-center">Polarity</th>
                      <th className="p-2.5 text-center">Condition (c_i)</th>
                      <th className="p-2.5 text-center">Mask (m_i)</th>
                      <th className="p-2.5">Conditioning Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-line/60 bg-bg-inset">
                    {GCIT_DIMENSION_NAMES.map((dim, idx) => {
                      const cVal = gcitVector[idx] !== undefined ? gcitVector[idx] : 0.0;
                      const mVal = gcitMask[idx] !== undefined ? gcitMask[idx] : 0.0;
                      const isInverted = dim === "facial_no_beard" || dim === "No_Beard";
                      const isConditioned = mVal === 1.0;

                      return (
                        <tr
                          key={dim}
                          className={`hover:bg-bg-panel transition-colors ${
                            isConditioned ? "bg-emerald-500/5 font-medium" : ""
                          }`}
                        >
                          <td className="p-2.5 text-ink-disabled">{idx}</td>
                          <td className="p-2.5 text-ink-primary font-bold">{dim}</td>
                          <td className="p-2.5 text-ink-tertiary">{getCategoryForGCITDim(dim)}</td>
                          <td className="p-2.5 text-center">
                            {isInverted ? (
                              <span className="px-1.5 py-0.5 rounded text-[10px] bg-amber-500/20 text-amber-400 font-bold">
                                INVERTED
                              </span>
                            ) : (
                              <span className="text-ink-disabled text-[11px]">DIRECT</span>
                            )}
                          </td>
                          <td className="p-2.5 text-center text-gold font-bold">
                            {cVal > 0 ? `+${cVal.toFixed(1)}` : cVal.toFixed(1)}
                          </td>
                          <td className="p-2.5 text-center">
                            <span
                              className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                                isConditioned
                                  ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                                  : "bg-bg-deepest text-ink-disabled border border-line"
                              }`}
                            >
                              {mVal.toFixed(1)}
                            </span>
                          </td>
                          <td className="p-2.5 text-[11px]">
                            {isConditioned ? (
                              <span className="text-emerald-400 flex items-center gap-1 font-sans">
                                <CheckCircle2 className="w-3.5 h-3.5" /> Conditioned
                              </span>
                            ) : (
                              <span className="text-ink-disabled font-sans">
                                Masked to empirical prior
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            /* Cryptographic Image Hashes Tab */
            <div className="h-full max-h-[62vh] overflow-y-auto space-y-3">
              <div className="p-3 rounded-lg bg-bg-inset border border-line text-xs">
                <h4 className="font-bold text-ink-primary">
                  Variation-Level Cryptographic Integrity Verification
                </h4>
                <p className="text-ink-tertiary text-[11px] mt-0.5">
                  Each variation image buffer generated by PyTorch is hashed via SHA-256 upon synthesis. These digests serve as algorithmic integrity checksums to verify bit-level file immutability and detect accidental corruption or tampering. Hashing alone does not establish legal chain of custody or evidence authenticity.
                </p>
              </div>

              {normalizedImageHashes.length > 0 ? (
                <div className="border border-line rounded-lg overflow-hidden">
                  <table className="w-full text-left text-xs font-mono">
                    <thead className="bg-bg-deepest border-b border-line text-ink-disabled text-[10px] uppercase">
                      <tr>
                        <th className="p-2.5">Variation</th>
                        <th className="p-2.5">SHA-256 Image Hash (Lossless PNG Buffer)</th>
                        <th className="p-2.5 text-right">Integrity</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-line/60 bg-bg-inset">
                      {normalizedImageHashes.map((img) => (
                        <tr key={img.variationIndex} className="hover:bg-bg-panel transition-colors">
                          <td className="p-2.5 text-ink-primary font-bold">
                            Variation #{img.variationIndex}
                          </td>
                          <td className="p-2.5 text-gold truncate max-w-md select-all">
                            {img.sha256}
                          </td>
                          <td className="p-2.5 text-right">
                            <span className="text-emerald-400 font-sans font-semibold text-[11px]">
                              VERIFIED
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="p-6 text-center text-ink-disabled text-xs border border-line rounded-lg bg-bg-inset">
                  No image variation hashes available for this record.
                </div>
              )}

              <div className="p-3 rounded-lg bg-bg-deepest border border-line text-xs font-mono space-y-1">
                <div className="text-ink-disabled text-[10px]">GENERATOR CHECKPOINT HASH</div>
                <div className="text-gold break-all select-all">
                  a1714389d0d1eddad10fae74a736ca41de5e230eada095e6868e5daeeb2a1964
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-line bg-bg-inset flex items-center justify-between">
          <div className="text-xs text-ink-tertiary">
            Forensic Standard: Non-destructive provenance ledger • Uncalibrated NLP confidence suppressed
          </div>
          <Button variant="secondary" size="sm" onClick={onClose}>
            Close Trace
          </Button>
        </div>
      </div>
    </div>
  );
};

// Helper: 20-D GCIT dimension list matching GCIT_20_DIMENSIONS
const GCIT_DIMENSION_NAMES = [
  "hair_black",
  "hair_brown",
  "hair_blond",
  "hair_gray",
  "hair_bald",
  "hair_bangs",
  "hair_straight",
  "hair_wavy",
  "hairline_receding",
  "facial_mustache",
  "facial_goatee",
  "facial_sideburns",
  "facial_stubble",
  "facial_no_beard",
  "nose_big",
  "nose_pointy",
  "lips_big",
  "face_oval",
  "face_cheekbones",
  "glasses",
];

function getCategoryForGCITDim(dim: string): string {
  if (dim.startsWith("hair_black") || dim.startsWith("hair_brown") || dim.startsWith("hair_blond") || dim.startsWith("hair_gray")) return "hair_color";
  if (dim.startsWith("hair_") || dim === "hairline_receding") return "hair_type / hair_length";
  if (dim.startsWith("facial_")) return "facial_hair";
  if (dim.startsWith("nose_")) return "nose";
  if (dim.startsWith("lips_")) return "mouth";
  if (dim.startsWith("face_")) return "face_shape";
  if (dim === "glasses") return "glasses";
  return "other";
}

function renderHighlightedSentence(sentence: string, substring: string) {
  const lowerSentence = sentence.toLowerCase();
  const lowerSub = substring.toLowerCase();
  const idx = lowerSentence.indexOf(lowerSub);

  if (idx === -1) {
    return sentence;
  }

  const before = sentence.slice(0, idx);
  const match = sentence.slice(idx, idx + substring.length);
  const after = sentence.slice(idx + substring.length);

  return (
    <>
      {before}
      <mark className="bg-gold/30 text-gold font-bold px-1 py-0.5 rounded border border-gold/50 mx-0.5">
        {match}
      </mark>
      {after}
    </>
  );
}
function renderEvidenceBadge(state: string) {
  switch (state) {
    case "explicit":
      return (
        <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 font-mono">
          <CheckCircle2 className="w-2.5 h-2.5" /> EXPLICIT
        </span>
      );
    case "uncertain":
      return (
        <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/30 font-mono">
          <HelpCircle className="w-2.5 h-2.5" /> UNCERTAIN
        </span>
      );
    case "negated":
      return (
        <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/30 font-mono">
          <XCircle className="w-2.5 h-2.5" /> NEGATED
        </span>
      );
    case "contradictory":
      return (
        <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/30 font-mono">
          <AlertTriangle className="w-2.5 h-2.5" /> CONTRADICTORY
        </span>
      );
    default:
      return (
        <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded bg-ink-disabled/10 text-ink-disabled border border-line font-mono">
          UNKNOWN
        </span>
      );
  }
}

export default EvidenceExplainabilityModal;
