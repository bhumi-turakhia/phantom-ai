import React, { useState, useEffect, useMemo } from "react";
import {
  X,
  Columns,
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  Ban,
  ShieldCheck,
  Sparkles,
  Sliders,
  History,
  ZoomIn,
  ZoomOut,
  Grid,
  Hash,
  FileSearch,
  RefreshCw,
  Check,
  Clock,
  UserCheck,
  Info,
  Layers,
  Filter,
} from "lucide-react";
import { Button } from "./ui/Button";
import {
  compareCaseCompositeVariations,
  selectCompositeVariation,
} from "../services/imageService";
import type {
  VariationComparisonResponse,
  ComparedVariationItem,
  GCITDimensionComparisonRow,
  RecordedAttributeComparisonRow,
} from "../services/imageService";

interface MultiVariationComparisonModalProps {
  isOpen: boolean;
  onClose: () => void;
  caseId: string;
  compositesList: any[];
  initialCompositeId?: string;
  initialTargets?: Array<{ compositeId: string; variationIndex: number }>;
  onVariationSelected?: (updatedComposite: any) => void;
  onOpenExplainability?: (compositeId: string, category?: string) => void;
}

function getUserDisplayName(userObj: any): string {
  if (!userObj) return "Investigator";
  if (typeof userObj === "string") return userObj;
  return (
    userObj.username ||
    userObj.name ||
    userObj.email ||
    userObj.investigatorId ||
    "Investigator"
  );
}

function getUserRoleLabel(userObj: any): string {
  if (!userObj || typeof userObj === "string") return "";
  return userObj.role ? `(${userObj.role})` : "";
}

export const MultiVariationComparisonModal: React.FC<MultiVariationComparisonModalProps> = ({
  isOpen,
  onClose,
  caseId,
  compositesList,
  initialCompositeId,
  initialTargets,
  onVariationSelected,
  onOpenExplainability,
}) => {
  const [selectedTargets, setSelectedTargets] = useState<
    Array<{ compositeId: string; variationIndex: number }>
  >([]);
  const [comparisonData, setComparisonData] = useState<VariationComparisonResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const [activeTab, setActiveTab] = useState<
    "gcit_diff" | "recorded_attributes" | "selection_audit"
  >("gcit_diff");
  const [dimFilter, setDimFilter] = useState<
    "all" | "differing" | "conditioned" | "masked"
  >("all");

  // Zoom / Enlarge Inspector State (non-destructive client-side inspection only)
  const [zoomScale, setZoomScale] = useState<number>(3); // 3x = 192px, 4x = 256px
  const [showPixelGrid, setShowPixelGrid] = useState<boolean>(false);
  const [enlargedVariation, setEnlargedVariation] = useState<ComparedVariationItem | null>(null);

  // Preferred Selection State
  const [selectingKey, setSelectingKey] = useState<string | null>(null);
  const [selectionNotesMap, setSelectionNotesMap] = useState<Record<string, string>>({});
  const [selectionSuccessMsg, setSelectionSuccessMsg] = useState<string | null>(null);

  // Initialize targets when modal opens
  useEffect(() => {
    if (!isOpen) return;
    setSelectionSuccessMsg(null);
    setEnlargedVariation(null);

    if (initialTargets && initialTargets.length >= 2) {
      setSelectedTargets(initialTargets.slice(0, 6));
      return;
    }

    const targetComp =
      compositesList.find((c) => c._id === initialCompositeId) || compositesList[0];

    if (targetComp && Array.isArray(targetComp.variations) && targetComp.variations.length >= 2) {
      setSelectedTargets(
        targetComp.variations.slice(0, 4).map((v: any) => ({
          compositeId: targetComp._id,
          variationIndex: v.variationIndex,
        }))
      );
    } else if (compositesList.length >= 2) {
      const crossTargets: Array<{ compositeId: string; variationIndex: number }> = [];
      for (const comp of compositesList.slice(0, 4)) {
        const primaryIdx = comp.selectedVariationIndex || comp.selectedVariation || 1;
        crossTargets.push({ compositeId: comp._id, variationIndex: primaryIdx });
      }
      setSelectedTargets(crossTargets);
    } else if (targetComp && Array.isArray(targetComp.variations) && targetComp.variations.length === 1) {
      setSelectedTargets([
        {
          compositeId: targetComp._id,
          variationIndex: targetComp.variations[0].variationIndex || 1,
        },
      ]);
    } else {
      setSelectedTargets([]);
    }
  }, [isOpen, initialCompositeId, initialTargets, compositesList]);

  // Fetch comparison analysis whenever selectedTargets changes
  const fetchComparison = async (
    targetsToCompare: Array<{ compositeId: string; variationIndex: number }>
  ) => {
    if (!caseId || targetsToCompare.length === 0) {
      setComparisonData(null);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await compareCaseCompositeVariations(caseId, {
        targets: targetsToCompare,
      });
      const payload: VariationComparisonResponse = res?.data || res;
      setComparisonData(payload);
    } catch (err: any) {
      setError(
        err.response?.data?.message ||
          err.message ||
          "Failed to load multi-variation comparison analysis."
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen && selectedTargets.length > 0) {
      fetchComparison(selectedTargets);
    }
  }, [isOpen, selectedTargets, caseId]);

  const toggleVariationTarget = (compositeId: string, variationIndex: number) => {
    setSelectedTargets((prev) => {
      const exists = prev.some(
        (t) => t.compositeId === compositeId && t.variationIndex === variationIndex
      );
      if (exists) {
        if (prev.length <= 1) return prev;
        return prev.filter(
          (t) => !(t.compositeId === compositeId && t.variationIndex === variationIndex)
        );
      } else {
        if (prev.length >= 6) {
          return [...prev.slice(1), { compositeId, variationIndex }];
        }
        return [...prev, { compositeId, variationIndex }];
      }
    });
  };

  const applyPresetCurrentSession = () => {
    const targetComp =
      compositesList.find((c) => c._id === initialCompositeId) || compositesList[0];
    if (!targetComp || !Array.isArray(targetComp.variations)) return;
    setSelectedTargets(
      targetComp.variations.slice(0, 4).map((v: any) => ({
        compositeId: targetComp._id,
        variationIndex: v.variationIndex,
      }))
    );
  };

  const applyPresetAcrossVersions = () => {
    const crossTargets: Array<{ compositeId: string; variationIndex: number }> = [];
    for (const comp of compositesList.slice(0, 4)) {
      const primaryIdx = comp.selectedVariationIndex || comp.selectedVariation || 1;
      crossTargets.push({ compositeId: comp._id, variationIndex: primaryIdx });
    }
    if (crossTargets.length > 0) {
      setSelectedTargets(crossTargets);
    }
  };

  const handleSelectPreferred = async (variation: ComparedVariationItem) => {
    const key = `${variation.compositeId}:${variation.variationIndex}`;
    setSelectingKey(key);
    setError(null);
    setSelectionSuccessMsg(null);
    try {
      const notes =
        selectionNotesMap[key]?.trim() ||
        `Selected Variation #${variation.variationIndex} (v${variation.version}, seed ${variation.seed}) during multi-variation forensic comparison review.`;
      const res = await selectCompositeVariation(
        caseId,
        variation.compositeId,
        variation.variationIndex,
        "approved",
        notes
      );
      const updated = res?.data?.composite || res?.composite || res?.data || res;
      setSelectionSuccessMsg(
        `Variation #${variation.variationIndex} (Composite v${variation.version}) marked as Investigator Preferred Selection. All other variations remain preserved.`
      );
      if (onVariationSelected) {
        onVariationSelected(updated);
      }
      await fetchComparison(selectedTargets);
    } catch (err: any) {
      setError(
        err.response?.data?.message ||
          err.message ||
          "Failed to mark preferred variation."
      );
    } finally {
      setSelectingKey(null);
    }
  };

  const filteredDimensions = useMemo(() => {
    if (!comparisonData) return [];
    return comparisonData.dimensionComparison.filter((row) => {
      if (dimFilter === "differing") return row.hasAnyDifference;
      if (dimFilter === "conditioned") {
        return row.perVariation.some((pv) => pv.maskValue === 1);
      }
      if (dimFilter === "masked") {
        return row.perVariation.every((pv) => pv.maskValue === 0);
      }
      return true;
    });
  }, [comparisonData, dimFilter]);

  if (!isOpen) return null;

  const differingDimsCount =
    comparisonData?.summary.differingGCITDimensionsCount ?? 0;
  const identicalDimsCount = 20 - differingDimsCount;
  const unsupportedCatCount =
    comparisonData?.recordedAttributesComparison.filter(
      (r) => !r.isSupportedInGCIT
    ).length ?? 0;
  const uniqueSeedsCount = comparisonData
    ? new Set(comparisonData.summary.seedsCompared).size
    : 0;
  const onlySeedDiffers =
    differingDimsCount === 0 && uniqueSeedsCount > 1;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-md p-3 md:p-5 animate-in fade-in duration-150"
      data-testid="multi-variation-comparison-modal"
    >
      <div className="bg-bg-panel border border-line rounded-xl max-w-7xl w-full max-h-[94vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex flex-wrap items-center justify-between gap-3 p-4 border-b border-line bg-bg-inset">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded bg-gold/10 border border-gold/30">
              <Columns className="w-5 h-5 text-gold" />
            </div>
            <div>
              <div className="flex items-center gap-2 flex-wrap">
                <h3 className="text-base font-bold text-ink-primary">
                  Multi-Variation Forensic Comparison & Attribute Difference Analysis
                </h3>
                <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-bg-deepest border border-line text-gold uppercase">
                  Phase 7.1 • {selectedTargets.length} Selected
                </span>
                {comparisonData && (
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-bg-deepest border border-line text-ink-secondary uppercase">
                    Mode:{" "}
                    {comparisonData.summary.sameCompositeSession
                      ? "Intra-Composite Batch"
                      : "Cross-Version Lineage"}
                  </span>
                )}
              </div>
              <p className="text-xs text-ink-tertiary mt-0.5">
                Compare composite variations side-by-side, inspect 20-D GCIT conditioning & mask differences, and record preferred investigative selections.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Non-destructive Zoom Controls */}
            <div className="flex items-center gap-1 bg-bg-deepest border border-line rounded-lg px-2 py-1 text-xs">
              <span className="text-[10px] font-mono text-ink-disabled uppercase mr-1">
                Scale:
              </span>
              {[2, 3, 4].map((scale) => (
                <button
                  key={scale}
                  type="button"
                  onClick={() => setZoomScale(scale)}
                  className={`px-1.5 py-0.5 rounded font-mono text-[11px] transition-colors ${
                    zoomScale === scale
                      ? "bg-gold text-bg-deepest font-bold"
                      : "text-ink-secondary hover:text-ink-primary"
                  }`}
                  title={`Inspect at ${scale * 64}×${scale * 64}px display scale (preserves original lossless PNG)`}
                >
                  {scale}×
                </button>
              ))}
              <button
                type="button"
                onClick={() => setShowPixelGrid((prev) => !prev)}
                className={`ml-1 p-1 rounded border transition-colors ${
                  showPixelGrid
                    ? "bg-gold/15 border-gold/40 text-gold"
                    : "border-transparent text-ink-tertiary hover:text-ink-primary"
                }`}
                title="Toggle forensic pixel inspection grid"
              >
                <Grid className="w-3.5 h-3.5" />
              </button>
            </div>

            <button
              type="button"
              onClick={() => fetchComparison(selectedTargets)}
              disabled={loading || selectedTargets.length === 0}
              className="p-1.5 rounded-lg text-ink-tertiary hover:text-ink-primary hover:bg-bg-deepest border border-line transition-colors"
              title="Refresh comparison data"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin text-gold" : ""}`} />
            </button>

            <button
              type="button"
              onClick={onClose}
              className="p-1.5 rounded-lg text-ink-tertiary hover:text-ink-primary hover:bg-bg-deepest transition-colors"
              aria-label="Close comparison modal"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Variation Picker Bar */}
        <div className="px-4 py-2.5 bg-bg-deepest border-b border-line flex flex-wrap items-center justify-between gap-2 text-xs">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] font-semibold text-ink-tertiary uppercase tracking-wider mr-1 flex items-center gap-1">
              <Layers className="w-3.5 h-3.5 text-gold" /> Variations in Case:
            </span>
            {compositesList.length === 0 ? (
              <span className="text-ink-disabled italic">
                No generated composites found for this case.
              </span>
            ) : (
              compositesList.map((comp) =>
                (comp.variations || []).map((v: any) => {
                  const isTargetSelected = selectedTargets.some(
                    (t) =>
                      t.compositeId === comp._id &&
                      t.variationIndex === v.variationIndex
                  );
                  const isPreferredOnComp =
                    (comp.selectedVariationIndex === v.variationIndex ||
                      comp.selectedVariation === v.variationIndex ||
                      v.isSelected) &&
                    comp.status === "approved";
                  const roleLabel =
                    v.variationRole === "semantic_edited"
                      ? "Edited"
                      : v.variationRole === "semantic_baseline"
                      ? "Baseline"
                      : comp.editingType && comp.editingType !== "initial"
                      ? v.variationIndex === 1
                        ? "Edited"
                        : "Baseline"
                      : "Std";

                  return (
                    <button
                      key={`${comp._id}-${v.variationIndex}`}
                      type="button"
                      onClick={() =>
                        toggleVariationTarget(comp._id, v.variationIndex)
                      }
                      data-testid={`comparison-picker-v${comp.version || 1}-var${v.variationIndex}`}
                      className={`px-2 py-1 rounded border font-mono text-[11px] flex items-center gap-1.5 transition-all ${
                        isTargetSelected
                          ? "bg-gold/15 border-gold text-gold font-bold shadow-sm"
                          : "bg-bg-inset border-line text-ink-secondary hover:border-line-strong hover:text-ink-primary"
                      }`}
                    >
                      <span>
                        v{comp.version || 1} • Var #{v.variationIndex}
                      </span>
                      <span className="text-[9px] px-1 py-0.2 rounded bg-bg-deepest text-ink-tertiary">
                        {roleLabel}
                      </span>
                      {isPreferredOnComp && (
                        <span title="Investigator Preferred Selection">
                          <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                        </span>
                      )}
                    </button>
                  );
                })
              )
            )}
          </div>

          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={applyPresetCurrentSession}
              className="px-2 py-1 rounded bg-bg-inset hover:bg-bg-panel border border-line text-[11px] text-ink-secondary hover:text-ink-primary transition-colors"
            >
              Current Session Variations
            </button>
            {compositesList.length > 1 && (
              <button
                type="button"
                onClick={applyPresetAcrossVersions}
                className="px-2 py-1 rounded bg-bg-inset hover:bg-bg-panel border border-line text-[11px] text-ink-secondary hover:text-ink-primary transition-colors"
              >
                Compare Across Versions (v1..v{compositesList.length})
              </button>
            )}
          </div>
        </div>

        {/* Success Banner */}
        {selectionSuccessMsg && (
          <div
            className="px-4 py-2 bg-emerald-500/10 border-b border-emerald-500/30 flex items-center justify-between text-xs text-emerald-300"
            data-testid="variation-selection-success-banner"
          >
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
              <span>{selectionSuccessMsg}</span>
            </div>
            <button
              type="button"
              onClick={() => setSelectionSuccessMsg(null)}
              className="text-emerald-400 hover:text-emerald-200 text-[11px] font-mono"
            >
              DISMISS
            </button>
          </div>
        )}

        {/* Main Content Scrollable Area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-5">
          {loading ? (
            <div
              className="flex flex-col items-center justify-center py-20 space-y-3"
              data-testid="comparison-loading-state"
            >
              <div className="w-9 h-9 border-2 border-gold border-t-transparent rounded-full animate-spin" />
              <p className="text-xs font-mono text-ink-secondary">
                Computing 20-D GCIT conditioning & mask difference matrix...
              </p>
            </div>
          ) : error ? (
            <div
              className="p-6 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm flex items-start gap-3"
              data-testid="comparison-error-state"
            >
              <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
              <div className="flex-1">
                <div className="font-semibold">Comparison Analysis Error</div>
                <div className="text-xs mt-1 text-red-300">{error}</div>
              </div>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => fetchComparison(selectedTargets)}
              >
                Retry
              </Button>
            </div>
          ) : !comparisonData || comparisonData.variations.length === 0 ? (
            <div
              className="p-12 text-center border border-dashed border-line rounded-xl bg-bg-inset/40 space-y-2"
              data-testid="comparison-empty-state"
            >
              <Columns className="w-8 h-8 text-ink-disabled mx-auto" />
              <div className="text-sm font-semibold text-ink-secondary">
                No Composite Variations Selected for Comparison
              </div>
              <p className="text-xs text-ink-tertiary max-w-md mx-auto">
                Select two or more composite variations using the variation chips above to compare their visual outputs, 20-D GCIT conditioning vectors, mask states, and latent seeds side-by-side.
              </p>
            </div>
          ) : (
            <>
              {/* High-Level Comparison Summary Banner */}
              <div className="p-3.5 rounded-lg bg-bg-inset border border-line flex flex-col lg:flex-row lg:items-center justify-between gap-3">
                <div className="space-y-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-xs font-bold text-ink-primary flex items-center gap-1.5">
                      <Info className="w-4 h-4 text-gold" />
                      Provenance & Conditioning Difference Summary:
                    </span>
                    {differingDimsCount === 0 ? (
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-sky-500/15 text-sky-300 border border-sky-500/30">
                        IDENTICAL 20-D GCIT CONDITIONING & MASK
                      </span>
                    ) : (
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-500/15 text-amber-300 border border-amber-500/30">
                        {differingDimsCount} / 20 GCIT DIMENSIONS DIFFER
                      </span>
                    )}
                    {onlySeedDiffers && (
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-purple-500/15 text-purple-300 border border-purple-500/30">
                        STOCHASTIC LATENT SEED (z) VARIATION ONLY
                      </span>
                    )}
                  </div>
                  <p
                    className="text-xs text-ink-secondary leading-relaxed"
                    data-testid="comparison-seed-difference-note"
                  >
                    {comparisonData.summary.seedDifferenceNote}
                  </p>
                </div>

                {/* Quick Metric Pills */}
                <div className="flex items-center gap-2 shrink-0 font-mono text-xs">
                  <div className="px-2.5 py-1.5 rounded bg-bg-deepest border border-line text-center">
                    <div className="text-[9px] text-ink-disabled uppercase">Identical Dims</div>
                    <div className="text-emerald-400 font-bold">
                      {identicalDimsCount} / 20
                    </div>
                  </div>
                  <div className="px-2.5 py-1.5 rounded bg-bg-deepest border border-line text-center">
                    <div className="text-[9px] text-ink-disabled uppercase">Differing Dims</div>
                    <div className="text-amber-400 font-bold">
                      {differingDimsCount} / 20
                    </div>
                  </div>
                  <div className="px-2.5 py-1.5 rounded bg-bg-deepest border border-line text-center">
                    <div className="text-[9px] text-ink-disabled uppercase">Unsupported</div>
                    <div className="text-purple-400 font-bold">
                      {unsupportedCatCount}
                    </div>
                  </div>
                </div>
              </div>

              {/* Side-by-Side Variation Cards Grid */}
              <div
                className={`grid gap-4 ${
                  comparisonData.variations.length === 1
                    ? "grid-cols-1 max-w-md mx-auto"
                    : comparisonData.variations.length === 2
                    ? "grid-cols-1 md:grid-cols-2"
                    : comparisonData.variations.length === 3
                    ? "grid-cols-1 md:grid-cols-3"
                    : "grid-cols-1 sm:grid-cols-2 lg:grid-cols-4"
                }`}
                data-testid="side-by-side-variations-grid"
              >
                {comparisonData.variations.map((v) => {
                  const key = `${v.compositeId}:${v.variationIndex}`;
                  const isPreferred =
                    v.isSelected && v.compositeStatus === "approved";
                  const isSelectingThis = selectingKey === key;
                  const imagePxSize = zoomScale * 64;

                  const contAdj = v.traversalParameters?.continuousAdjustments || {};
                  const latTrav = v.traversalParameters?.latentTraversals || [];
                  const hasContinuousAdjustments =
                    v.variationRole === "semantic_edited" &&
                    Object.keys(contAdj).length > 0;
                  const hasLatentTraversals =
                    v.variationRole === "semantic_edited" && latTrav.length > 0;

                  return (
                    <div
                      key={key}
                      data-testid={`compared-variation-card-v${v.version}-var${v.variationIndex}`}
                      className={`rounded-xl border flex flex-col justify-between overflow-hidden transition-all ${
                        isPreferred
                          ? "bg-bg-inset border-emerald-500/60 shadow-lg shadow-emerald-950/20"
                          : "bg-bg-inset border-line hover:border-line-strong"
                      }`}
                    >
                      {/* Top Header of Card */}
                      <div>
                        <div className="p-3 border-b border-line bg-bg-deepest flex items-center justify-between gap-2">
                          <div className="flex items-center gap-1.5 flex-wrap">
                            <span className="font-mono text-xs font-bold text-ink-primary">
                              v{v.version} • Variation #{v.variationIndex}
                            </span>
                            <span
                              className={`px-1.5 py-0.5 rounded text-[9px] font-mono uppercase font-bold ${
                                v.variationRole === "semantic_edited"
                                  ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                                  : v.variationRole === "semantic_baseline"
                                  ? "bg-sky-500/20 text-sky-300 border border-sky-500/30"
                                  : "bg-bg-panel text-ink-secondary border border-line"
                              }`}
                            >
                              {v.variationRole === "semantic_edited"
                                ? "Semantic Edited"
                                : v.variationRole === "semantic_baseline"
                                ? "Unedited Baseline"
                                : "Standard Batch"}
                            </span>
                          </div>

                          {isPreferred ? (
                            <span
                              className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 flex items-center gap-1"
                              data-testid={`preferred-badge-v${v.version}-var${v.variationIndex}`}
                            >
                              <CheckCircle2 className="w-3 h-3" /> PREFERRED
                            </span>
                          ) : v.isSelected ? (
                            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-gold/15 text-gold border border-gold/30">
                              Active Draft
                            </span>
                          ) : null}
                        </div>

                        {/* 64x64 Composite Image Viewport with Non-Destructive Scale & Optional Pixel Grid */}
                        <div className="p-4 flex flex-col items-center justify-center bg-black/50 border-b border-line relative group">
                          <div
                            className="relative border border-line-strong rounded overflow-hidden shadow-inner bg-bg-deepest"
                            style={{
                              width: `${Math.min(imagePxSize, 256)}px`,
                              height: `${Math.min(imagePxSize, 256)}px`,
                            }}
                          >
                            <img
                              src={v.dataUrl || v.imageUrl}
                              alt={`Composite v${v.version} Variation #${v.variationIndex}`}
                              className="w-full h-full object-contain"
                              style={{ imageRendering: (v.width || 128) >= 128 ? "auto" : "pixelated" }}
                            />
                            {showPixelGrid && (
                              <div
                                className="absolute inset-0 pointer-events-none opacity-35"
                                style={{
                                  backgroundImage:
                                    "linear-gradient(to right, rgba(255,255,255,0.2) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,0.2) 1px, transparent 1px)",
                                  backgroundSize: `${Math.min(imagePxSize, 256) / 16}px ${
                                    Math.min(imagePxSize, 256) / 16
                                  }px`,
                                }}
                              />
                            )}
                          </div>

                          <div className="mt-2 flex items-center gap-2 text-[10px] font-mono text-ink-tertiary">
                            <span>Native {v.width || 128}×{v.height || 128} PNG</span>
                            <span>•</span>
                            <button
                              type="button"
                              onClick={() => setEnlargedVariation(v)}
                              className="text-gold hover:underline flex items-center gap-1"
                            >
                              <ZoomIn className="w-3 h-3" /> Enlarge {v.width === 64 ? "8× (512px)" : "4× (512px)"}
                            </button>
                          </div>
                        </div>

                        {/* Metadata & Provenance Details */}
                        <div className="p-3 space-y-2 text-xs">
                          <div className="grid grid-cols-2 gap-1.5 font-mono text-[11px]">
                            <div className="p-1.5 rounded bg-bg-deepest border border-line">
                              <span className="text-[9px] text-ink-disabled block uppercase">
                                Latent Seed (z)
                              </span>
                              <span className="text-gold font-bold">{v.seed}</span>
                            </div>
                            <div className="p-1.5 rounded bg-bg-deepest border border-line">
                              <span className="text-[9px] text-ink-disabled block uppercase">
                                Created At
                              </span>
                              <span className="text-ink-secondary">
                                {new Date(v.createdAt).toLocaleTimeString([], {
                                  hour: "2-digit",
                                  minute: "2-digit",
                                  second: "2-digit",
                                })}
                              </span>
                            </div>
                          </div>

                          {/* SHA-256 Digest */}
                          {v.sha256 && (
                            <div className="p-1.5 rounded bg-bg-deepest border border-line font-mono text-[10px] flex items-center justify-between gap-1">
                              <span className="text-ink-disabled flex items-center gap-1 shrink-0">
                                <Hash className="w-3 h-3 text-gold" /> SHA-256:
                              </span>
                              <span
                                className="text-ink-secondary truncate select-all"
                                title={v.sha256}
                              >
                                {v.sha256.slice(0, 16)}…{v.sha256.slice(-8)}
                              </span>
                            </div>
                          )}

                          {/* Semantic Adjustments / Latent Traversals if present */}
                          {(hasContinuousAdjustments || hasLatentTraversals) && (
                            <div className="p-2 rounded bg-amber-500/10 border border-amber-500/25 space-y-1">
                              <div className="text-[10px] font-mono font-bold text-amber-300 uppercase flex items-center gap-1">
                                <Sliders className="w-3 h-3" /> Semantic Edit Deltas
                              </div>
                              {hasContinuousAdjustments && (
                                <div className="flex flex-wrap gap-1">
                                  {Object.entries(contAdj).map(([dim, val]) => (
                                    <span
                                      key={dim}
                                      className="px-1.5 py-0.5 rounded bg-bg-deepest text-[10px] font-mono text-amber-200 border border-amber-500/30"
                                    >
                                      {dim}: {val > 0 ? `+${val}` : val}
                                    </span>
                                  ))}
                                </div>
                              )}
                              {hasLatentTraversals && (
                                <div className="flex flex-wrap gap-1">
                                  {latTrav.map((lt) => (
                                    <span
                                      key={lt.attribute}
                                      className="px-1.5 py-0.5 rounded bg-bg-deepest text-[10px] font-mono text-purple-300 border border-purple-500/30"
                                    >
                                      α({lt.attribute}):{" "}
                                      {lt.alpha > 0 ? `+${lt.alpha}` : lt.alpha}
                                    </span>
                                  ))}
                                </div>
                              )}
                            </div>
                          )}

                          {/* Approval / Selection Attribution */}
                          {(v.selectedBy || v.reviewedBy) && (
                            <div className="p-2 rounded bg-emerald-500/10 border border-emerald-500/25 text-[11px] space-y-0.5">
                              <div className="font-semibold text-emerald-300 flex items-center gap-1">
                                <UserCheck className="w-3.5 h-3.5" />
                                Selected by:{" "}
                                {getUserDisplayName(v.selectedBy || v.reviewedBy)}
                              </div>
                              {(v.selectedAt || v.reviewedAt) && (
                                <div className="text-[10px] font-mono text-ink-tertiary flex items-center gap-1">
                                  <Clock className="w-3 h-3" />
                                  {new Date(
                                    (v.selectedAt || v.reviewedAt) as string
                                  ).toLocaleString()}
                                </div>
                              )}
                              {v.selectionNotes && (
                                <div className="text-[11px] text-ink-secondary italic mt-1 border-t border-emerald-500/20 pt-1">
                                  &ldquo;{v.selectionNotes}&rdquo;
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Bottom Actions: Selection Notes, Mark Preferred, Explain Lineage */}
                      <div className="p-3 border-t border-line bg-bg-deepest/80 space-y-2">
                        <input
                          type="text"
                          placeholder="Optional rationale for preferred selection..."
                          value={selectionNotesMap[key] || ""}
                          onChange={(e) =>
                            setSelectionNotesMap((prev) => ({
                              ...prev,
                              [key]: e.target.value,
                            }))
                          }
                          data-testid={`selection-notes-input-v${v.version}-var${v.variationIndex}`}
                          className="w-full px-2.5 py-1.5 rounded bg-bg-inset border border-line text-xs text-ink-primary placeholder:text-ink-disabled focus:outline-none focus:border-gold"
                        />

                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            disabled={isSelectingThis}
                            onClick={() => handleSelectPreferred(v)}
                            data-testid={`select-preferred-btn-v${v.version}-var${v.variationIndex}`}
                            className={`flex-1 py-1.5 px-2.5 rounded text-xs font-semibold flex items-center justify-center gap-1.5 transition-all ${
                              isPreferred
                                ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 hover:bg-emerald-500/30"
                                : "bg-gold text-bg-deepest hover:bg-gold/90 font-bold"
                            }`}
                          >
                            <Check className="w-3.5 h-3.5" />
                            {isSelectingThis
                              ? "Recording..."
                              : isPreferred
                              ? "Update Selection Note"
                              : "Select as Preferred"}
                          </button>

                          {onOpenExplainability && (
                            <button
                              type="button"
                              onClick={() => onOpenExplainability(v.compositeId)}
                              className="py-1.5 px-2.5 rounded bg-bg-inset hover:bg-bg-panel border border-line text-xs text-gold flex items-center gap-1 transition-colors"
                              title="Open Phase 7.0 Evidence Traceability & Explainability (XAI) for this composite"
                            >
                              <FileSearch className="w-3.5 h-3.5" />
                              <span>XAI</span>
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Analysis Sub-Tabs */}
              <div className="border border-line rounded-xl overflow-hidden bg-bg-inset">
                <div className="flex flex-wrap items-center justify-between gap-2 px-4 border-b border-line bg-bg-deepest">
                  <div className="flex items-center gap-4">
                    <button
                      type="button"
                      onClick={() => setActiveTab("gcit_diff")}
                      data-testid="tab-gcit-diff-matrix"
                      className={`py-3 text-xs font-semibold border-b-2 transition-colors flex items-center gap-1.5 ${
                        activeTab === "gcit_diff"
                          ? "border-gold text-gold"
                          : "border-transparent text-ink-tertiary hover:text-ink-primary"
                      }`}
                    >
                      <Sliders className="w-3.5 h-3.5" />
                      20-D GCIT Conditioning & Mask Difference Matrix
                    </button>

                    <button
                      type="button"
                      onClick={() => setActiveTab("recorded_attributes")}
                      data-testid="tab-recorded-attributes"
                      className={`py-3 text-xs font-semibold border-b-2 transition-colors flex items-center gap-1.5 ${
                        activeTab === "recorded_attributes"
                          ? "border-gold text-gold"
                          : "border-transparent text-ink-tertiary hover:text-ink-primary"
                      }`}
                    >
                      <ShieldCheck className="w-3.5 h-3.5" />
                      Recorded, Uncertain & Unsupported Attributes (
                      {comparisonData.recordedAttributesComparison.length})
                    </button>

                    <button
                      type="button"
                      onClick={() => setActiveTab("selection_audit")}
                      data-testid="tab-selection-audit"
                      className={`py-3 text-xs font-semibold border-b-2 transition-colors flex items-center gap-1.5 ${
                        activeTab === "selection_audit"
                          ? "border-gold text-gold"
                          : "border-transparent text-ink-tertiary hover:text-ink-primary"
                      }`}
                    >
                      <History className="w-3.5 h-3.5" />
                      Selection Audit Log ({comparisonData.selectionHistory.length})
                    </button>
                  </div>

                  {activeTab === "gcit_diff" && (
                    <div className="flex items-center gap-1 py-1.5">
                      <Filter className="w-3.5 h-3.5 text-ink-disabled mr-1" />
                      {(
                        [
                          { id: "all", label: "All 20 Dims" },
                          {
                            id: "differing",
                            label: `Differing (${differingDimsCount})`,
                          },
                          { id: "conditioned", label: "Conditioned (m_i=1)" },
                          { id: "masked", label: "Masked (m_i=0)" },
                        ] as const
                      ).map((f) => (
                        <button
                          key={f.id}
                          type="button"
                          onClick={() => setDimFilter(f.id)}
                          className={`px-2 py-0.5 rounded text-[11px] font-mono transition-colors ${
                            dimFilter === f.id
                              ? "bg-gold/20 text-gold border border-gold/40 font-bold"
                              : "text-ink-tertiary hover:text-ink-primary"
                          }`}
                        >
                          {f.label}
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                {/* Tab 1: 20-D GCIT Conditioning & Mask Difference Matrix */}
                {activeTab === "gcit_diff" && (
                  <div className="overflow-x-auto">
                    <table
                      className="w-full text-left text-xs font-mono"
                      data-testid="gcit-difference-table"
                    >
                      <thead className="bg-bg-deepest/60 border-b border-line text-[10px] text-ink-disabled uppercase">
                        <tr>
                          <th className="p-2.5">Idx</th>
                          <th className="p-2.5">GCIT Dimension</th>
                          <th className="p-2.5">Category</th>
                          <th className="p-2.5 text-center">Polarity</th>
                          {comparisonData.variations.map((v) => (
                            <th
                              key={`${v.compositeId}-${v.variationIndex}`}
                              className="p-2.5 border-l border-line text-gold"
                            >
                              v{v.version} • Var #{v.variationIndex} (
                              {v.variationRole === "semantic_edited"
                                ? "Edited"
                                : v.variationRole === "semantic_baseline"
                                ? "Baseline"
                                : `Seed ${v.seed}`}
                              )
                            </th>
                          ))}
                          <th className="p-2.5 border-l border-line">
                            Difference Analysis
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-line/60">
                        {filteredDimensions.length === 0 ? (
                          <tr>
                            <td
                              colSpan={5 + comparisonData.variations.length}
                              className="p-6 text-center text-ink-disabled font-sans"
                            >
                              No GCIT dimensions match the current filter (
                              {dimFilter}).
                            </td>
                          </tr>
                        ) : (
                          filteredDimensions.map((row: GCITDimensionComparisonRow) => (
                            <tr
                              key={row.index}
                              className={`transition-colors ${
                                row.hasAnyDifference
                                  ? "bg-amber-500/10 hover:bg-amber-500/15"
                                  : "hover:bg-bg-panel/60"
                              }`}
                            >
                              <td className="p-2.5 text-ink-disabled">
                                {row.index}
                              </td>
                              <td className="p-2.5 font-bold text-ink-primary">
                                {row.dimensionName}
                              </td>
                              <td className="p-2.5 text-ink-tertiary">
                                {row.categoryDisplayName || row.category}
                              </td>
                              <td className="p-2.5 text-center">
                                {row.polarity === "inverted" ? (
                                  <span
                                    className="px-1.5 py-0.5 rounded text-[9px] bg-amber-500/20 text-amber-300 font-bold"
                                    title="facial_no_beard uses inverted CelebA polarity (+1.0 = No Beard / Clean-Shaven)"
                                  >
                                    INVERTED
                                  </span>
                                ) : (
                                  <span className="text-[10px] text-ink-disabled">
                                    DIRECT
                                  </span>
                                )}
                              </td>

                              {row.perVariation.map((pv) => {
                                const isCond = pv.maskValue === 1;
                                return (
                                  <td
                                    key={`${pv.compositeId}-${pv.variationIndex}`}
                                    className="p-2.5 border-l border-line/60"
                                  >
                                    <div className="flex items-center gap-2 flex-wrap">
                                      <span
                                        className={`px-1.5 py-0.5 rounded text-[11px] font-bold ${
                                          isCond
                                            ? pv.conditionValue > 0
                                              ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                                              : pv.conditionValue < 0
                                              ? "bg-red-500/20 text-red-300 border border-red-500/30"
                                              : "bg-sky-500/20 text-sky-300 border border-sky-500/30"
                                            : "bg-bg-deepest text-ink-disabled border border-line"
                                        }`}
                                      >
                                        c=
                                        {pv.conditionValue > 0
                                          ? `+${pv.conditionValue.toFixed(2)}`
                                          : pv.conditionValue.toFixed(2)}
                                      </span>
                                      <span
                                        className={`px-1.5 py-0.5 rounded text-[10px] ${
                                          isCond
                                            ? "bg-gold/15 text-gold border border-gold/30 font-bold"
                                            : "text-ink-disabled"
                                        }`}
                                      >
                                        m={pv.maskValue.toFixed(1)}
                                      </span>
                                      {pv.latentTraversalAlpha !== null &&
                                        pv.latentTraversalAlpha !== 0 && (
                                          <span className="px-1.5 py-0.5 rounded text-[10px] bg-purple-500/20 text-purple-300 border border-purple-500/30">
                                            α=
                                            {pv.latentTraversalAlpha > 0
                                              ? `+${pv.latentTraversalAlpha}`
                                              : pv.latentTraversalAlpha}
                                          </span>
                                        )}
                                    </div>
                                    <div className="text-[10px] text-ink-tertiary font-sans mt-1">
                                      {isCond ? (
                                        <span>
                                          Conditioned ({pv.evidenceState})
                                          {pv.recordedAttributeValue
                                            ? `: "${pv.recordedAttributeValue}"`
                                            : ""}
                                        </span>
                                      ) : (
                                        <span className="text-ink-disabled">
                                          Masked ({pv.status})
                                        </span>
                                      )}
                                    </div>
                                  </td>
                                );
                              })}

                              <td className="p-2.5 border-l border-line/60 font-sans text-[11px]">
                                {row.hasAnyDifference ? (
                                  <div className="space-y-1">
                                    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40">
                                      <AlertTriangle className="w-3 h-3" /> DIFFERS
                                    </span>
                                    <div className="text-ink-secondary">
                                      {row.differenceSummary}
                                    </div>
                                  </div>
                                ) : (
                                  <span className="text-ink-disabled flex items-center gap-1">
                                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500/70" />
                                    {row.differenceSummary}
                                  </span>
                                )}
                              </td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                )}

                {/* Tab 2: Recorded, Uncertain & Unsupported Attributes */}
                {activeTab === "recorded_attributes" && (
                  <div className="overflow-x-auto">
                    <table
                      className="w-full text-left text-xs"
                      data-testid="recorded-attributes-comparison-table"
                    >
                      <thead className="bg-bg-deepest/60 border-b border-line text-[10px] font-mono text-ink-disabled uppercase">
                        <tr>
                          <th className="p-2.5">Attribute Category</th>
                          <th className="p-2.5">Generator Support</th>
                          {comparisonData.variations.map((v) => (
                            <th
                              key={`${v.compositeId}-${v.variationIndex}`}
                              className="p-2.5 border-l border-line text-gold"
                            >
                              v{v.version} • Var #{v.variationIndex}
                            </th>
                          ))}
                          <th className="p-2.5 border-l border-line">
                            Forensic Conditioning Explanation
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-line/60">
                        {comparisonData.recordedAttributesComparison.length === 0 ? (
                          <tr>
                            <td
                              colSpan={3 + comparisonData.variations.length}
                              className="p-6 text-center text-ink-disabled"
                            >
                              No explicit witness or investigator attribute records attached to these variations.
                            </td>
                          </tr>
                        ) : (
                          comparisonData.recordedAttributesComparison.map(
                            (attrRow: RecordedAttributeComparisonRow) => (
                              <tr
                                key={attrRow.category}
                                className="hover:bg-bg-panel/60 transition-colors"
                              >
                                <td className="p-2.5 font-mono font-bold text-ink-primary">
                                  {attrRow.categoryDisplayName || attrRow.category}
                                </td>
                                <td className="p-2.5 font-mono text-[11px]">
                                  {attrRow.isSupportedInGCIT ? (
                                    <span className="px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                                      20-D GCIT Supported
                                    </span>
                                  ) : (
                                    <span className="px-1.5 py-0.5 rounded bg-purple-500/15 text-purple-300 border border-purple-500/30 flex items-center gap-1 w-fit">
                                      <Ban className="w-3 h-3" /> Unsupported by 20-D Generator
                                    </span>
                                  )}
                                </td>

                                {attrRow.perVariation.map((pv) => (
                                  <td
                                    key={`${pv.compositeId}-${pv.variationIndex}`}
                                    className="p-2.5 border-l border-line/60 space-y-1"
                                  >
                                    <div className="flex items-center gap-1.5 flex-wrap">
                                      <span className="font-mono font-semibold text-ink-primary">
                                        {pv.recordedValue || "—"}
                                      </span>
                                      <span
                                        className={`px-1.5 py-0.5 rounded text-[10px] font-mono uppercase ${
                                          pv.classification === "conditioned"
                                            ? "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30"
                                            : pv.classification === "masked_uncertain"
                                            ? "bg-amber-500/15 text-amber-300 border border-amber-500/30"
                                            : pv.classification ===
                                              "unsupported_attribute"
                                            ? "bg-purple-500/15 text-purple-300 border border-purple-500/30"
                                            : "bg-bg-deepest text-ink-disabled border border-line"
                                        }`}
                                      >
                                        {pv.classification.replace(/_/g, " ")}
                                      </span>
                                    </div>
                                    <div className="text-[11px] text-ink-tertiary">
                                      State:{" "}
                                      <span className="font-mono text-ink-secondary">
                                        {pv.evidenceState}
                                      </span>{" "}
                                      {pv.isCorrected
                                        ? "(investigator_override)"
                                        : "(nlp_extraction)"}
                                    </div>
                                    {onOpenExplainability && (
                                      <button
                                        type="button"
                                        onClick={() =>
                                          onOpenExplainability(
                                            pv.compositeId,
                                            attrRow.category
                                          )
                                        }
                                        className="text-[10px] font-mono text-gold hover:underline flex items-center gap-1 mt-0.5"
                                      >
                                        <Sparkles className="w-2.5 h-2.5" /> Inspect XAI Trace
                                      </button>
                                    )}
                                  </td>
                                ))}

                                <td className="p-2.5 border-l border-line/60 text-[11px] text-ink-secondary">
                                  {!attrRow.isSupportedInGCIT ? (
                                    <span className="text-purple-300">
                                      Recorded in forensic case documentation only. This attribute category has no corresponding dimension in the 20-D Conditional WGAN-GP architecture and does not condition pixel synthesis.
                                    </span>
                                  ) : attrRow.hasDifference ? (
                                    <span className="text-amber-300">
                                      Recorded attribute state or conditioning value differs across the compared composite variations.
                                    </span>
                                  ) : (
                                    <span className="text-ink-tertiary">
                                      {attrRow.perVariation[0]?.rationale ||
                                        "Consistent across compared variations."}
                                    </span>
                                  )}
                                </td>
                              </tr>
                            )
                          )
                        )}
                      </tbody>
                    </table>
                  </div>
                )}

                {/* Tab 3: Selection Audit Log */}
                {activeTab === "selection_audit" && (
                  <div className="p-4 space-y-3" data-testid="selection-audit-history-panel">
                    <div className="text-xs text-ink-secondary">
                      Complete, non-destructive audit trail of preferred variation selections recorded for these composite sessions. Selecting a preferred variation never deletes or overwrites alternative variations.
                    </div>

                    {comparisonData.selectionHistory.length === 0 ? (
                      <div className="p-6 text-center rounded-lg bg-bg-deepest border border-line text-xs text-ink-disabled">
                        No preferred variation selection events have been recorded yet for the compared composite(s).
                      </div>
                    ) : (
                      <div className="border border-line rounded-lg overflow-hidden">
                        <table className="w-full text-left text-xs font-mono">
                          <thead className="bg-bg-deepest border-b border-line text-[10px] text-ink-disabled uppercase">
                            <tr>
                              <th className="p-2.5">Timestamp</th>
                              <th className="p-2.5">Composite</th>
                              <th className="p-2.5">Selected Variation</th>
                              <th className="p-2.5">Previous</th>
                              <th className="p-2.5">Investigator</th>
                              <th className="p-2.5">Status</th>
                              <th className="p-2.5">Forensic Selection Notes</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-line/60 bg-bg-inset">
                            {comparisonData.selectionHistory.map((item, idx) => (
                              <tr key={idx} className="hover:bg-bg-panel/60">
                                <td className="p-2.5 text-ink-secondary">
                                  {new Date(item.selectedAt).toLocaleString()}
                                </td>
                                <td className="p-2.5 text-gold font-bold">
                                  v{item.version}
                                </td>
                                <td className="p-2.5 text-emerald-400 font-bold">
                                  Variation #{item.variationIndex}{" "}
                                  {item.seed !== undefined
                                    ? `(Seed ${item.seed})`
                                    : ""}
                                </td>
                                <td className="p-2.5 text-ink-tertiary">
                                  {item.previousVariationIndex !== undefined &&
                                  item.previousVariationIndex !== null
                                    ? `Var #${item.previousVariationIndex}`
                                    : "—"}
                                </td>
                                <td className="p-2.5 text-ink-primary">
                                  {getUserDisplayName(item.selectedBy)}{" "}
                                  {getUserRoleLabel(item.selectedBy)}
                                </td>
                                <td className="p-2.5 uppercase text-[10px] text-emerald-300">
                                  {item.status}
                                </td>
                                <td className="p-2.5 font-sans text-ink-secondary">
                                  {item.notes || "—"}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Mandatory Forensic & Methodological Disclosures */}
              <div
                className="p-3.5 rounded-lg bg-bg-deepest border border-line space-y-2 text-[11px] text-ink-tertiary"
                data-testid="comparison-forensic-disclosures"
              >
                <div className="font-bold text-ink-secondary uppercase tracking-wider text-[10px] flex items-center gap-1.5">
                  <HelpCircle className="w-3.5 h-3.5 text-gold" />
                  Mandatory Methodological & Forensic Disclosures
                </div>
                <ul className="list-disc list-inside space-y-1 leading-relaxed">
                  <li>
                    <strong className="text-ink-secondary">
                      Unconditioned Prior Disclosure (m_i = 0):
                    </strong>{" "}
                    {comparisonData.disclosures.unconditionedPriorDisclosure}
                  </li>
                  <li>
                    <strong className="text-ink-secondary">
                      No Pixel-Level Inference:
                    </strong>{" "}
                    {comparisonData.disclosures.noPixelInferenceDisclosure}
                  </li>
                  <li>
                    <strong className="text-ink-secondary">
                      Non-Biometric Investigative Aid:
                    </strong>{" "}
                    {comparisonData.disclosures.nonBiometricSelectionDisclosure}
                  </li>
                </ul>
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-line bg-bg-inset flex items-center justify-between text-xs">
          <div className="text-ink-tertiary font-mono text-[11px]">
            Non-destructive multi-variation comparison • Original lossless PNGs & SHA-256 digests preserved
          </div>
          <Button variant="secondary" size="sm" onClick={onClose}>
            Close Comparison
          </Button>
        </div>
      </div>

      {/* Non-Destructive Enlarged Inspection Overlay */}
      {enlargedVariation && (
        <div
          className="fixed inset-0 z-60 flex items-center justify-center bg-black/90 p-4"
          onClick={() => setEnlargedVariation(null)}
        >
          <div
            className="bg-bg-panel border border-line rounded-xl p-5 max-w-xl w-full space-y-4 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-line pb-3">
              <div>
                <h4 className="text-sm font-bold text-ink-primary">
                  Enlarged Forensic Inspection — Composite v{enlargedVariation.version} • Variation #{enlargedVariation.variationIndex}
                </h4>
                <p className="text-[11px] text-ink-tertiary font-mono">
                  Seed: {enlargedVariation.seed} • Display: 512×512px • Stored File: Lossless {enlargedVariation.width || 128}×{enlargedVariation.height || 128} PNG
                </p>
              </div>
              <button
                type="button"
                onClick={() => setEnlargedVariation(null)}
                className="p-1 rounded text-ink-tertiary hover:text-ink-primary"
              >
                <ZoomOut className="w-5 h-5" />
              </button>
            </div>

            <div className="flex justify-center bg-black p-4 rounded-lg border border-line">
              <div className="relative w-[384px] h-[384px] sm:w-[448px] sm:h-[448px] border border-line-strong">
                <img
                  src={enlargedVariation.dataUrl || enlargedVariation.imageUrl}
                  alt="Enlarged variation inspection"
                  className="w-full h-full object-contain"
                  style={{ imageRendering: (enlargedVariation.width || 128) >= 128 ? "auto" : "pixelated" }}
                />
              </div>
            </div>

            <div className="flex items-center justify-between text-xs font-mono text-ink-tertiary">
              <span className="truncate max-w-sm">
                SHA-256: {enlargedVariation.sha256 || "Verified"}
              </span>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => setEnlargedVariation(null)}
              >
                Exit Enlarged View
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
