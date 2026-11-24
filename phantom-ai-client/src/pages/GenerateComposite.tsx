import { useState, useEffect } from "react";
import {
  createWitnessDescription,
  analyzeWitness,
  addWitnessCorrection,
  approveWitnessAnalysis,
  getWitnessDescriptionsByCase,
  generateCaseComposite,
  getCaseComposites,
  selectCompositeVariation,
  getLatentDirections,
  editCaseComposite,
  getCompositeTrace,
  getWitnessDescriptionTrace,
} from "../services/imageService";
import type {
  LatentDirectionsInfo,
  CompositeTraceResponse,
  WitnessDescriptionTraceResponse,
} from "../services/imageService";
import { getAllCases, createCase } from "../services/caseService";
import {
  ScanFace,
  Fingerprint,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  HelpCircle,
  Cpu,
  Edit3,
  ShieldCheck,
  FolderPlus,
  Info,
  Layers,
  Sparkles,
  RefreshCw,
  Sliders,
  Check,
  ArrowLeftRight,
  CheckCheck,
  Maximize2,
  Download,
  Grid,
  X,
  Lock,
  RotateCcw,
  Wand2,
  Columns,
} from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "../components/ui/Card";
import { Textarea, Input } from "../components/ui/Input";
import { Button } from "../components/ui/Button";
import { EvidenceExplainabilityModal } from "../components/EvidenceExplainabilityModal";
import { MultiVariationComparisonModal } from "../components/MultiVariationComparisonModal";
import { EvidenceTag } from "../components/ui/EvidenceTag";
import { StatusBadge } from "../components/ui/Badge";
import { formatCaseId, formatTimestamp } from "../lib/utils";

export interface ExtractedAttributeItem {
  category: string;
  value: string | null;
  sourceText: string;
  start: number | null;
  end: number | null;
  evidenceState: "explicit" | "uncertain" | "negated" | "contradictory" | "unknown";
  extractedBy: string;
  confidence?: number | null;
}

export interface ModelMetadata {
  name?: string;
  architecture?: string;
  phase?: string;
  checkpoint?: string;
  checkpointSha256?: string;
}

export interface InvestigatorCorrection {
  attributeCategory: string;
  previousValue?: string | null;
  correctedValue: string | null;
  previousEvidenceState?: string;
  correctedEvidenceState?: string;
  reason: string;
  timestamp: string;
  rejected?: boolean;
}

export default function GenerateComposite() {
  const [statement, setStatement] = useState("");
  const [cases, setCases] = useState<any[]>([]);
  const [selectedCaseId, setSelectedCaseId] = useState<string>("");
  const [isCreatingCase, setIsCreatingCase] = useState(false);
  const [newCaseTitle, setNewCaseTitle] = useState("");

  const [generating, setGenerating] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [caseRef, setCaseRef] = useState(() => formatCaseId(Math.floor(Math.random() * 90000)));

  // ML Analysis State
  const [witnessDocId, setWitnessDocId] = useState<string | null>(null);
  const [modelMeta, setModelMeta] = useState<ModelMetadata | null>(null);
  const [attributesList, setAttributesList] = useState<ExtractedAttributeItem[]>([]);
  const [highlightedSpan, setHighlightedSpan] = useState<{
    start: number;
    end: number;
    category: string;
    spanText: string;
  } | null>(null);
  const [showUnmentioned, setShowUnmentioned] = useState(false);

  // Review & Correction State
  const [reviewStatus, setReviewStatus] = useState<"pending" | "in_review" | "approved" | "rejected">("pending");
  const [correctionsList, setCorrectionsList] = useState<InvestigatorCorrection[]>([]);
  const [selectedAttrForEdit, setSelectedAttrForEdit] = useState<ExtractedAttributeItem | null>(null);
  const [editValue, setEditValue] = useState("");
  const [editState, setEditState] = useState<"explicit" | "uncertain" | "negated" | "contradictory">("explicit");
  const [editReason, setEditReason] = useState("");
  const [editRejected, setEditRejected] = useState(false);
  const [savingCorrection, setSavingCorrection] = useState(false);
  const [approving, setApproving] = useState(false);

  // Witness descriptions & composites for the active case
  const [witnessDocs, setWitnessDocs] = useState<any[]>([]);
  const [selectedWitnessId, setSelectedWitnessId] = useState<string>("");

  // Composite generation & version state
  const [compositesList, setCompositesList] = useState<any[]>([]);
  const [selectedComposite, setSelectedComposite] = useState<any | null>(null);
  const [compareComposite, setCompareComposite] = useState<any | null>(null);
  const [isComparing, setIsComparing] = useState(false);
  const [numVariations, setNumVariations] = useState<number>(2);
  const [customSeed, setCustomSeed] = useState<string>("42");
  const [isGeneratingComposite, setIsGeneratingComposite] = useState(false);
  const [compositeError, setCompositeError] = useState<string | null>(null);
  const [compositeSuccessMsg, setCompositeSuccessMsg] = useState<string | null>(null);
  const [isSelectingVariation, setIsSelectingVariation] = useState(false);
  const [inspectedVariation, setInspectedVariation] = useState<any | null>(null);
  const [inspectZoomScale, setInspectZoomScale] = useState<number>(4);
  const [showPixelGrid, setShowPixelGrid] = useState<boolean>(false);

  // Semantic Editing Studio State
  const [latentDirectionsInfo, setLatentDirectionsInfo] = useState<LatentDirectionsInfo | null>(null);
  const [isEditingComposite, setIsEditingComposite] = useState<boolean>(false);
  const [editingTargetVariation, setEditingTargetVariation] = useState<any | null>(null);
  const [editingSeed, setEditingSeed] = useState<number>(42);
  const [continuousAdjustments, setContinuousAdjustments] = useState<Record<string, number>>({
    nose_big: 0.0,
    nose_pointy: 0.0,
    lips_big: 0.0,
    face_oval: 0.0,
    face_cheekbones: 0.0,
    hair_wavy: 0.0,
    hair_bangs: 0.0,
    facial_no_beard: 0.0,
    facial_stubble: 0.0,
  });
  const [latentTraversals, setLatentTraversals] = useState<Record<string, number>>({
    hair_blond: 0.0,
    hair_gray: 0.0,
    hair_bangs: 0.0,
    facial_goatee: 0.0,
    glasses: 0.0,
  });
  const [editNotes, setEditNotes] = useState<string>("");
  const [isSubmittingEdit, setIsSubmittingEdit] = useState<boolean>(false);
  const [editError, setEditError] = useState<string | null>(null);

  // Phase 7.0 Evidence Traceability & Explainability State
  const [traceModalOpen, setTraceModalOpen] = useState<boolean>(false);
  const [traceData, setTraceData] = useState<CompositeTraceResponse | WitnessDescriptionTraceResponse | null>(null);
  const [traceType, setTraceType] = useState<"composite" | "witness">("composite");
  const [traceInitialCategory, setTraceInitialCategory] = useState<string | undefined>(undefined);
  const [loadingTrace, setLoadingTrace] = useState<boolean>(false);

  // Phase 7.1 Multi-Variation Comparison & Attribute Difference State
  const [comparisonModalOpen, setComparisonModalOpen] = useState<boolean>(false);
  const [comparisonTargets, setComparisonTargets] = useState<
    Array<{ compositeId: string; variationIndex: number }>
  >([]);

  const toggleComparisonCandidate = (compositeId: string, variationIndex: number) => {
    setComparisonTargets((prev) => {
      const exists = prev.some(
        (t) => t.compositeId === compositeId && t.variationIndex === variationIndex
      );
      if (exists) {
        return prev.filter(
          (t) => !(t.compositeId === compositeId && t.variationIndex === variationIndex)
        );
      }
      if (prev.length >= 6) {
        return [...prev.slice(1), { compositeId, variationIndex }];
      }
      return [...prev, { compositeId, variationIndex }];
    });
  };

  const handleOpenComparisonModal = (
    customTargets?: Array<{ compositeId: string; variationIndex: number }>
  ) => {
    if (customTargets && customTargets.length > 0) {
      setComparisonTargets(customTargets);
    }
    setComparisonModalOpen(true);
  };

  const handleOpenCompositeTrace = async (compositeId?: string, targetCategory?: string) => {
    const compId = compositeId || selectedComposite?._id;
    if (!selectedCaseId || !compId) return;
    try {
      setLoadingTrace(true);
      setTraceType("composite");
      setTraceInitialCategory(targetCategory);
      setTraceModalOpen(true);
      const res = await getCompositeTrace(selectedCaseId, compId);
      if (res.success && res.data) {
        setTraceData(res.data);
      }
    } catch (err: any) {
      console.error("Failed to load composite trace:", err);
    } finally {
      setLoadingTrace(false);
    }
  };

  const handleOpenWitnessTrace = async (wId?: string, targetCategory?: string) => {
    const id = wId || witnessDocId;
    if (!id) return;
    try {
      setLoadingTrace(true);
      setTraceType("witness");
      setTraceInitialCategory(targetCategory);
      setTraceModalOpen(true);
      const res = await getWitnessDescriptionTrace(id);
      if (res.success && res.data) {
        setTraceData(res.data);
      }
    } catch (err: any) {
      console.error("Failed to load witness trace:", err);
    } finally {
      setLoadingTrace(false);
    }
  };

  // Load cases on mount (respect ?caseId= query parameter if present)
  useEffect(() => {
    async function loadCases() {
      try {
        const res = await getAllCases();
        const caseList = res.data?.cases || res.cases || res.data || [];
        if (Array.isArray(caseList) && caseList.length > 0) {
          setCases(caseList);
          const params = new URLSearchParams(window.location.search);
          const requestedCaseId = params.get("caseId");
          const matchedCase = requestedCaseId
            ? caseList.find((c: any) => c._id === requestedCaseId)
            : null;
          const initialCase = matchedCase || caseList[0];
          setSelectedCaseId(initialCase._id);
          setCaseRef(initialCase.caseNumber || caseRef);
        }
      } catch (err) {
        console.warn("Could not load existing cases:", err);
      }
    }
    loadCases();
  }, []);

  // Load witness statements and composites when selectedCaseId changes
  useEffect(() => {
    if (!selectedCaseId) return;
    async function loadCaseData() {
      try {
        const [witRes, compRes, latentRes] = await Promise.allSettled([
          getWitnessDescriptionsByCase(selectedCaseId),
          getCaseComposites(selectedCaseId),
          getLatentDirections(selectedCaseId),
        ]);

        if (witRes.status === "fulfilled") {
          const list =
            witRes.value?.data?.witnessDescriptions ||
            witRes.value?.witnessDescriptions ||
            witRes.value?.data ||
            [];
          const arr = Array.isArray(list) ? list : [];
          setWitnessDocs(arr);
          if (arr.length > 0 && !witnessDocId) {
            const approvedOne =
              arr.find((w: any) => w.reviewStatus === "approved") || arr[0];
            setSelectedWitnessId(approvedOne._id);
            setWitnessDocId(approvedOne._id);
            setStatement(approvedOne.rawStatement || "");
            setReviewStatus(approvedOne.reviewStatus || "pending");
            setAttributesList(approvedOne.extractedAttributes || []);
            setCorrectionsList(approvedOne.corrections || []);
          }
        }

        if (compRes.status === "fulfilled") {
          const list =
            compRes.value?.data?.composites ||
            compRes.value?.composites ||
            compRes.value?.data ||
            [];
          const comps = Array.isArray(list) ? list : [];
          setCompositesList(comps);
          if (comps.length > 0) {
            setSelectedComposite(comps[0]);
            if (comps.length > 1) {
              setCompareComposite(comps[1]);
            } else {
              setCompareComposite(null);
            }
          } else {
            setSelectedComposite(null);
            setCompareComposite(null);
          }
        }

        if (latentRes.status === "fulfilled") {
          const lData = latentRes.value?.data || latentRes.value;
          if (lData) {
            setLatentDirectionsInfo(lData);
          }
        }
      } catch (err) {
        console.warn("Could not load case data:", err);
      }
    }
    loadCaseData();
  }, [selectedCaseId]);

  async function handleCreateNewCase() {
    if (!newCaseTitle.trim()) return;
    try {
      const caseNumber = `CASE-${Date.now().toString().slice(-6)}`;
      const created = await createCase({
        caseNumber,
        title: newCaseTitle.trim(),
        description: `Automated forensic case for ${newCaseTitle.trim()}`,
      });
      const newDoc = created.data || created;
      setCases((prev) => [newDoc, ...prev]);
      setSelectedCaseId(newDoc._id);
      setCaseRef(newDoc.caseNumber || caseNumber);
      setIsCreatingCase(false);
      setNewCaseTitle("");
    } catch (err) {
      console.error("Failed to create case:", err);
      setErrorMsg("Failed to create new case record. Please try again.");
    }
  }

  async function handleGenerate() {
    if (!statement.trim()) return;

    setGenerating(true);
    setErrorMsg(null);
    setHighlightedSpan(null);

    try {
      // Ensure we have a valid case ID
      let targetCaseId = selectedCaseId;
      if (!targetCaseId) {
        if (cases.length > 0) {
          targetCaseId = cases[0]._id;
        } else {
          const autoCase = await createCase({
            caseNumber: `AUTO-${Date.now().toString().slice(-6)}`,
            title: "Forensic Investigation Session",
            description: "Automatically initialized case for witness intake",
          });
          const autoDoc = autoCase.data || autoCase;
          targetCaseId = autoDoc._id;
          setCases([autoDoc]);
          setSelectedCaseId(autoDoc._id);
        }
      }

      // 1. Ingest Witness Description
      const witnessRes = await createWitnessDescription({
        case: targetCaseId,
        rawStatement: statement.trim(),
      });
      const currentWitnessId = witnessRes?.data?._id || witnessRes?._id;
      if (!currentWitnessId) {
        throw new Error("Failed to record witness statement in database");
      }
      setWitnessDocId(currentWitnessId);

      // 2. Run Local BiLSTM-CRF Analysis via Node ML client
      const analysisRes = await analyzeWitness(currentWitnessId);
      const analysisData = analysisRes?.data || analysisRes;

      const attrsList: ExtractedAttributeItem[] =
        analysisData?.extractedAttributesList || [];
      const meta: ModelMetadata = analysisData?.modelMetadata || {
        name: "CharAwareBiLSTMAttributeLabeler",
        architecture: "CharAwareBiLSTM-CRF",
        phase: "3C.2",
        checkpointSha256: "2d567c79d759e76940968fadb02d412b530d3405e377a2d6454c64e9f2e7b4a3",
      };

      setAttributesList(attrsList);
      setModelMeta(meta);
      setReviewStatus(analysisData?.reviewStatus || "in_review");
      setSelectedWitnessId(currentWitnessId);

      // Keep witnessDocs list fresh
      setWitnessDocs((prev) => {
        const existing = prev.find((w) => w._id === currentWitnessId);
        if (existing) {
          return prev.map((w) =>
            w._id === currentWitnessId
              ? {
                  ...w,
                  rawStatement: statement.trim(),
                  reviewStatus: analysisData?.reviewStatus || "in_review",
                  extractedAttributes: attrsList,
                }
              : w
          );
        }
        return [
          {
            _id: currentWitnessId,
            rawStatement: statement.trim(),
            reviewStatus: analysisData?.reviewStatus || "in_review",
            extractedAttributes: attrsList,
          },
          ...prev,
        ];
      });
    } catch (err: any) {
      console.error("ANALYSIS FAILED:", err);
      const message =
        err?.response?.data?.message ||
        err?.message ||
        "Error communicating with the Python ML inference service.";
      setErrorMsg(message);
    } finally {
      setGenerating(false);
    }
  }

  function handleHighlightAttribute(attr: ExtractedAttributeItem) {
    if (
      highlightedSpan &&
      highlightedSpan.start === attr.start &&
      highlightedSpan.end === attr.end
    ) {
      setHighlightedSpan(null);
      return;
    }

    if (
      typeof attr.start === "number" &&
      typeof attr.end === "number" &&
      attr.start >= 0 &&
      attr.end <= statement.length
    ) {
      setHighlightedSpan({
        start: attr.start,
        end: attr.end,
        category: attr.category,
        spanText: statement.substring(attr.start, attr.end),
      });
    } else if (attr.sourceText && statement.includes(attr.sourceText)) {
      const idx = statement.indexOf(attr.sourceText);
      setHighlightedSpan({
        start: idx,
        end: idx + attr.sourceText.length,
        category: attr.category,
        spanText: attr.sourceText,
      });
    }
  }

  function openEditModal(attr: ExtractedAttributeItem) {
    setSelectedAttrForEdit(attr);
    setEditValue(attr.value || "");
    setEditState(
      attr.evidenceState === "unknown" ? "explicit" : (attr.evidenceState as any)
    );
    setEditReason("");
    setEditRejected(attr.evidenceState === "negated");
  }

  async function handleSaveCorrection() {
    if (!witnessDocId || !selectedAttrForEdit) return;
    if (!editReason.trim()) {
      alert("Please provide an investigator justification for this forensic audit trail.");
      return;
    }

    setSavingCorrection(true);
    try {
      await addWitnessCorrection(witnessDocId, {
        attributeCategory: selectedAttrForEdit.category,
        previousValue: selectedAttrForEdit.value,
        correctedValue: editRejected ? null : editValue.trim(),
        previousEvidenceState: selectedAttrForEdit.evidenceState,
        correctedEvidenceState: editRejected ? "negated" : editState,
        reason: editReason.trim(),
        rejected: editRejected,
      });

      // Update local attribute list
      setAttributesList((prev) =>
        prev.map((item) => {
          if (item.category.toLowerCase() === selectedAttrForEdit.category.toLowerCase()) {
            return {
              ...item,
              value: editRejected ? null : editValue.trim(),
              evidenceState: editRejected ? "negated" : editState,
              extractedBy: "investigator_correction",
              confidence: null,
            };
          }
          return item;
        })
      );

      // Append to corrections history
      setCorrectionsList((prev) => [
        {
          attributeCategory: selectedAttrForEdit.category,
          previousValue: selectedAttrForEdit.value,
          correctedValue: editRejected ? null : editValue.trim(),
          previousEvidenceState: selectedAttrForEdit.evidenceState,
          correctedEvidenceState: editRejected ? "negated" : editState,
          reason: editReason.trim(),
          timestamp: new Date().toISOString(),
          rejected: editRejected,
        },
        ...prev,
      ]);

      setSelectedAttrForEdit(null);
    } catch (err: any) {
      console.error("Failed to save correction:", err);
      alert(err?.response?.data?.message || "Failed to record correction.");
    } finally {
      setSavingCorrection(false);
    }
  }

  async function handleApprove() {
    if (!witnessDocId) return;
    setApproving(true);
    try {
      await approveWitnessAnalysis(witnessDocId, "approved");
      setReviewStatus("approved");
      setWitnessDocs((prev) =>
        prev.map((w) =>
          w._id === witnessDocId ? { ...w, reviewStatus: "approved" } : w
        )
      );
    } catch (err: any) {
      console.error("Failed to approve analysis:", err);
      alert(err?.response?.data?.message || "Failed to approve witness analysis.");
    } finally {
      setApproving(false);
    }
  }

  async function handleSynthesizeComposite() {
    if (!selectedCaseId) {
      setCompositeError("Please select or create an active case first.");
      return;
    }
    const targetWitnessId = witnessDocId || selectedWitnessId;
    if (!targetWitnessId) {
      setCompositeError("No witness description selected. Submit and approve a testimony first.");
      return;
    }

    if (reviewStatus !== "approved") {
      setCompositeError("Visual composite synthesis is strictly restricted until the facial attribute profile is approved by an investigator.");
      return;
    }

    setIsGeneratingComposite(true);
    setCompositeError(null);
    setCompositeSuccessMsg(null);

    try {
      const parsedSeed = customSeed ? parseInt(customSeed, 10) : undefined;
      const res = await generateCaseComposite(selectedCaseId, {
        witnessDescriptionId: targetWitnessId,
        seed: Number.isNaN(parsedSeed) ? undefined : parsedSeed,
        numVariations,
      });

      const newComposite = res?.data?.composite || res?.composite || res?.data;
      if (newComposite) {
        setCompositesList((prev) => [newComposite, ...prev]);
        setSelectedComposite(newComposite);
        setCompositeSuccessMsg(
          `Successfully synthesized Composite Version ${newComposite.version} (${newComposite.variations?.length || numVariations} variations)!`
        );
      }
    } catch (err: any) {
      console.error("Composite generation failed:", err);
      const msg =
        err?.response?.data?.message ||
        err?.message ||
        "Failed to generate facial composite.";
      setCompositeError(msg);
    } finally {
      setIsGeneratingComposite(false);
    }
  }

  async function handleSelectVariation(variationIndex: number) {
    if (!selectedCaseId || !selectedComposite) return;
    setIsSelectingVariation(true);
    try {
      const res = await selectCompositeVariation(
        selectedCaseId,
        selectedComposite._id,
        variationIndex
      );
      const updated = res?.data?.composite || res?.composite || res?.data;
      if (updated) {
        setSelectedComposite(updated);
        setCompositesList((prev) =>
          prev.map((c) => (c._id === updated._id ? updated : c))
        );
      }
    } catch (err: any) {
      console.error("Failed to select variation:", err);
      alert(err?.response?.data?.message || "Failed to select composite variation.");
    } finally {
      setIsSelectingVariation(false);
    }
  }

  function handleRandomizeSeed() {
    setCustomSeed(Math.floor(Math.random() * 100000).toString());
  }

  function handleDownloadImage(variation: any) {
    if (!variation.dataUrl && !variation.imageUrl) return;
    const a = document.createElement("a");
    a.href = variation.dataUrl || variation.imageUrl;
    a.download = `composite_${selectedCaseId || "case"}_v${selectedComposite?.version || 1}_var${variation.variationIndex}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  function handleStartEditing(variation: any) {
    setEditingTargetVariation(variation);
    setEditingSeed(variation.seed);
    setContinuousAdjustments({
      nose_big: 0.0,
      nose_pointy: 0.0,
      lips_big: 0.0,
      face_oval: 0.0,
      face_cheekbones: 0.0,
      hair_wavy: 0.0,
      hair_bangs: 0.0,
      facial_no_beard: 0.0,
      facial_stubble: 0.0,
    });
    setLatentTraversals({
      hair_blond: 0.0,
      hair_gray: 0.0,
      hair_bangs: 0.0,
      facial_goatee: 0.0,
      glasses: 0.0,
    });
    setEditNotes("");
    setEditError(null);
    setIsEditingComposite(true);
  }

  function handleContinuousChange(key: string, value: number) {
    setContinuousAdjustments((prev) => ({
      ...prev,
      [key]: value,
    }));
  }

  function handleLatentChange(key: string, value: number) {
    setLatentTraversals((prev) => ({
      ...prev,
      [key]: value,
    }));
  }

  function handleResetAdjustments() {
    setContinuousAdjustments({
      nose_big: 0.0,
      nose_pointy: 0.0,
      lips_big: 0.0,
      face_oval: 0.0,
      face_cheekbones: 0.0,
      hair_wavy: 0.0,
      hair_bangs: 0.0,
      facial_no_beard: 0.0,
      facial_stubble: 0.0,
    });
    setLatentTraversals({
      hair_blond: 0.0,
      hair_gray: 0.0,
      hair_bangs: 0.0,
      facial_goatee: 0.0,
      glasses: 0.0,
    });
    setEditError(null);
  }

  function handleCancelEditing() {
    setIsEditingComposite(false);
    setEditingTargetVariation(null);
    setEditError(null);
  }

  async function handleApplySemanticEdit() {
    if (!selectedCaseId || !selectedComposite || !editingTargetVariation) return;

    // Filter non-zero adjustments
    const nonZeroContinuous: Record<string, number> = {};
    for (const [k, v] of Object.entries(continuousAdjustments)) {
      if (Math.abs(v) > 0.001) {
        nonZeroContinuous[k] = parseFloat(v.toFixed(3));
      }
    }

    const nonZeroLatent: Record<string, number> = {};
    for (const [k, v] of Object.entries(latentTraversals)) {
      if (Math.abs(v) > 0.001) {
        nonZeroLatent[k] = parseFloat(v.toFixed(3));
      }
    }

    if (Object.keys(nonZeroContinuous).length === 0 && Object.keys(nonZeroLatent).length === 0) {
      setEditError("No adjustments have been modified. Move at least one slider to generate an edited composite.");
      return;
    }

    setIsSubmittingEdit(true);
    setEditError(null);

    try {
      const res = await editCaseComposite(selectedCaseId, selectedComposite._id, {
        selectedSeed: editingSeed,
        continuousAdjustments: nonZeroContinuous,
        latentTraversals: nonZeroLatent,
        notes: editNotes.trim() || undefined,
      });

      const newComposite = res?.data?.composite || res?.composite || res?.data;
      if (newComposite) {
        setCompositesList((prev) => [newComposite, ...prev]);
        setSelectedComposite(newComposite);
        setIsEditingComposite(false);
        setEditingTargetVariation(null);
        setCompositeSuccessMsg(
          `Successfully synthesized Composite Version ${newComposite.version} (edited from v${selectedComposite.version}, seed ${editingSeed})!`
        );
      }
    } catch (err: any) {
      console.error("Semantic edit failed:", err);
      const msg =
        err?.response?.data?.message ||
        err?.message ||
        "Failed to apply semantic editing.";
      setEditError(msg);
    } finally {
      setIsSubmittingEdit(false);
    }
  }

  const renderHighlightedStatement = () => {
    if (!statement) return null;
    if (!highlightedSpan) {
      return (
        <p className="text-sm text-ink-primary font-mono whitespace-pre-wrap leading-relaxed select-text">
          {statement}
        </p>
      );
    }

    const { start, end, category, spanText } = highlightedSpan;
    const before = statement.substring(0, start);
    const highlighted = statement.substring(start, end);
    const after = statement.substring(end);

    return (
      <div className="space-y-2">
        <p className="text-sm text-ink-primary font-mono whitespace-pre-wrap leading-relaxed select-text">
          <span>{before}</span>
          <mark className="bg-gold/25 border-b-2 border-gold text-ink-primary px-1 py-0.5 rounded-sm font-semibold">
            {highlighted}
          </mark>
          <span>{after}</span>
        </p>
        <div className="flex items-center gap-2 pt-1">
          <span className="text-xs label-technical text-gold bg-gold/10 px-2 py-0.5 rounded border border-gold/30">
            Traced Feature: [{category}] &ldquo;{spanText}&rdquo; (offsets {start}:{end})
          </span>
          <button
            onClick={() => setHighlightedSpan(null)}
            className="text-xs text-ink-disabled hover:text-ink-primary underline ml-auto"
          >
            Clear Highlight
          </button>
        </div>
      </div>
    );
  };

  const getEvidenceBadge = (state: string) => {
    switch (state) {
      case "explicit":
        return (
          <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
            <CheckCircle2 className="w-3 h-3" /> EXPLICIT
          </span>
        );
      case "uncertain":
        return (
          <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/30">
            <HelpCircle className="w-3 h-3" /> UNCERTAIN
          </span>
        );
      case "negated":
        return (
          <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/30">
            <XCircle className="w-3 h-3" /> NEGATED
          </span>
        );
      case "contradictory":
        return (
          <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/30">
            <AlertTriangle className="w-3 h-3" /> CONTRADICTORY
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded bg-slate-500/10 text-slate-400 border border-slate-500/30">
            UNOBSERVED
          </span>
        );
    }
  };

  // Separate facial reconstruction attributes from non-facial observables
  const facialAttributes = attributesList.filter(
    (a) => a.category !== "other_observable" && (showUnmentioned || a.value !== null || a.evidenceState !== "unknown")
  );

  const nonFacialObservables = attributesList.filter(
    (a) => a.category === "other_observable" && (a.value !== null || a.evidenceState !== "unknown")
  );

  const approvedFacialProfile = facialAttributes.filter(
    (a) => a.evidenceState !== "unknown" && a.evidenceState !== "negated" && a.value !== null
  );

  return (
    <div className="space-y-6">
      {/* Header with Case Selector and Model Provenance */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-xl font-bold text-ink-primary tracking-tight">
            Forensic Facial Composite & Evidence Review
          </h1>
          <p className="label-technical text-ink-disabled mt-1">
            Character-Aware BiLSTM-CRF Extraction & Investigator Supervision Pipeline
          </p>
        </div>

        <div className="flex items-center gap-3">
          <EvidenceTag caseId={caseRef} />
          {modelMeta && (
            <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded bg-bg-inset border border-line text-xs font-mono text-ink-tertiary">
              <Cpu className="w-3.5 h-3.5 text-gold" />
              <span>
                {modelMeta.architecture || "CharAwareBiLSTM-CRF"} (Phase {modelMeta.phase || "3C.2"})
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Error Alert Banner */}
      {errorMsg && (
        <div className="rounded-md border border-rose-500/40 bg-rose-500/10 p-4 text-sm text-rose-300 flex items-start gap-3">
          <AlertTriangle className="h-5 w-5 text-rose-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <p className="font-semibold text-rose-200">ML Pipeline Communication Error</p>
            <p className="text-rose-300/90">{errorMsg}</p>
            {errorMsg.toLowerCase().includes("offline") && (
              <p className="text-xs text-rose-400 font-mono mt-1">
                To start the local ML service:{" "}
                <code className="bg-bg-deepest px-1.5 py-0.5 rounded text-white">
                  cd phantom-ai-ml && uvicorn app.main:app --port 8000
                </code>
              </p>
            )}
          </div>
        </div>
      )}

      {/* Case Management Selector */}
      <Card>
        <CardContent className="py-3 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="label-technical text-ink-tertiary">Active Case:</span>
            {!isCreatingCase ? (
              <select
                className="h-9 px-3 rounded bg-bg-inset border border-line text-sm text-ink-primary focus:border-gold/50 outline-none"
                value={selectedCaseId}
                onChange={(e) => {
                  setSelectedCaseId(e.target.value);
                  const selected = cases.find((c) => c._id === e.target.value);
                  if (selected) setCaseRef(selected.caseNumber || caseRef);
                }}
              >
                {cases.map((c) => (
                  <option key={c._id} value={c._id}>
                    {c.caseNumber || "Case"} — {c.title}
                  </option>
                ))}
              </select>
            ) : (
              <div className="flex items-center gap-2">
                <Input
                  placeholder="Case title..."
                  value={newCaseTitle}
                  onChange={(e) => setNewCaseTitle(e.target.value)}
                  className="h-9 w-64"
                />
                <Button size="sm" onClick={handleCreateNewCase}>
                  Create
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setIsCreatingCase(false)}
                >
                  Cancel
                </Button>
              </div>
            )}
            {!isCreatingCase && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsCreatingCase(true)}
                className="gap-1.5"
              >
                <FolderPlus className="w-3.5 h-3.5" /> New Case
              </Button>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {witnessDocs.length > 0 && (
              <div className="flex items-center gap-2">
                <span className="label-technical text-ink-tertiary">Select Statement:</span>
                <select
                  className="h-9 px-3 rounded bg-bg-inset border border-line text-xs text-ink-primary focus:border-gold/50 outline-none max-w-[220px] truncate"
                  value={witnessDocId || ""}
                  onChange={(e) => {
                    const doc = witnessDocs.find((w) => w._id === e.target.value);
                    if (doc) {
                      setWitnessDocId(doc._id);
                      setSelectedWitnessId(doc._id);
                      setStatement(doc.rawStatement || "");
                      setReviewStatus(doc.reviewStatus || "pending");
                      setAttributesList(doc.extractedAttributes || []);
                      setCorrectionsList(doc.corrections || []);
                    }
                  }}
                >
                  {witnessDocs.map((w, idx) => (
                    <option key={w._id} value={w._id}>
                      #{idx + 1} ({w.reviewStatus.toUpperCase()}) — {w.rawStatement.slice(0, 30)}...
                    </option>
                  ))}
                </select>
              </div>
            )}

            {witnessDocId && (
              <div className="flex items-center gap-2">
                <span className="label-technical text-ink-disabled font-mono text-xs">
                  {witnessDocId.slice(-6)}
                </span>
                <StatusBadge
                  tone={
                    reviewStatus === "approved"
                      ? "verified"
                      : reviewStatus === "rejected"
                      ? "critical"
                      : "pending"
                  }
                >
                  {reviewStatus.toUpperCase()}
                </StatusBadge>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Witness Statement Intake */}
        <Card className="flex flex-col">
          <CardHeader>
            <CardTitle>Witness Statement Intake</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 flex-1 flex flex-col">
            <Textarea
              id="statement"
              label="Verbatim Witness Testimony"
              placeholder="Record the witness's exact statement — describe facial structure, hair, eyes, nose, facial hair, distinguishing scars or tattoos, and any uncertain or negated details..."
              maxLength={2000}
              value={statement}
              onChange={(e) => setStatement(e.target.value)}
              className="min-h-[140px]"
            />

            <Button
              onClick={handleGenerate}
              loading={generating}
              disabled={!statement.trim() || generating}
              className="w-full"
            >
              <ScanFace className="h-4 w-4" />
              {generating
                ? "Extracting Attributes via BiLSTM-CRF…"
                : "Analyze Statement (Local BiLSTM-CRF)"}
            </Button>

            {/* Traceable Highlight Box */}
            {statement && (
              <div className="mt-4 pt-4 border-t border-line/60">
                <div className="flex items-center justify-between mb-2">
                  <span className="label-technical text-ink-tertiary">
                    Interactive Evidence Traceability
                  </span>
                  <span className="text-xs text-ink-disabled">
                    Click any extracted feature below to highlight exact text span
                  </span>
                </div>
                <div className="p-3 rounded bg-bg-inset border border-line min-h-[80px]">
                  {renderHighlightedStatement()}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* AI Extracted Attributes with Traceability */}
        <Card className="flex flex-col">
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>BiLSTM-CRF Extracted Attributes</CardTitle>
              <p className="text-[11px] text-ink-disabled mt-0.5">
                Facial reconstruction attributes grounded in source testimony
              </p>
            </div>
            {attributesList.length > 0 && (
              <div className="flex items-center gap-2">
                <StatusBadge tone="verified">Evidence States Grounded</StatusBadge>
              </div>
            )}
          </CardHeader>
          <CardContent className="flex-1 flex flex-col justify-between space-y-4">
            {attributesList.length === 0 && !generating && (
              <div className="py-12 text-center text-ink-tertiary">
                <Fingerprint className="h-10 w-10 text-ink-disabled mx-auto mb-2 opacity-50" />
                <p className="text-sm">Submit a witness statement to extract facial descriptors.</p>
                <p className="text-xs text-ink-disabled mt-1">
                  Extraction runs using genuine sequence-level CRF without paid external APIs.
                </p>
              </div>
            )}

            {generating && (
              <div className="space-y-3 py-6">
                {Array.from({ length: 6 }).map((_, i) => (
                  <div
                    key={i}
                    className="h-10 rounded bg-bg-inset animate-pulse"
                    style={{ animationDelay: `${i * 0.1}s` }}
                  />
                ))}
              </div>
            )}

            {attributesList.length > 0 && (
              <div className="space-y-4">
                {/* Calibration Disclosure Note */}
                <div className="flex items-start gap-2 p-2 rounded bg-bg-inset border border-line/70 text-xs text-ink-disabled">
                  <Info className="w-3.5 h-3.5 text-gold shrink-0 mt-0.5" />
                  <span>
                    <strong>Confidence Audit Note:</strong> Numerical score is a heuristic extraction metric based on model emissions & uncertainty cues, not a calibrated posterior probability of ground truth.
                  </span>
                </div>

                {/* Filter Toggle & Trace All */}
                <div className="flex items-center justify-between text-xs pb-1 border-b border-line/40">
                  <div className="flex items-center gap-2">
                    <span className="text-ink-tertiary font-mono">
                      Facial Features ({facialAttributes.length})
                    </span>
                    <button
                      onClick={() => handleOpenWitnessTrace()}
                      className="text-xs text-gold hover:underline flex items-center gap-1 font-mono ml-2"
                      title="Inspect full 15-point evidence trace for all witness features"
                    >
                      <ShieldCheck className="w-3.5 h-3.5" /> Explain All
                    </button>
                  </div>
                  <button
                    onClick={() => setShowUnmentioned(!showUnmentioned)}
                    className="text-xs text-ink-disabled hover:text-ink-secondary underline"
                  >
                    {showUnmentioned ? "Hide unobserved categories" : "Show all categories"}
                  </button>
                </div>

                <div className="max-h-[300px] overflow-y-auto divide-y divide-line/60 pr-1">
                  {facialAttributes.map((attr, idx) => {
                    const isHighlighted =
                      highlightedSpan &&
                      highlightedSpan.start === attr.start &&
                      highlightedSpan.end === attr.end;

                    return (
                      <div
                        key={`${attr.category}-${idx}`}
                        className={`py-2 px-2 rounded transition-colors flex items-center justify-between gap-3 ${
                          isHighlighted ? "bg-gold/10 border border-gold/30" : "hover:bg-bg-inset/50"
                        }`}
                      >
                        <div
                          className="flex-1 cursor-pointer"
                          onClick={() => handleHighlightAttribute(attr)}
                        >
                          <div className="flex items-center gap-2">
                            <span className="font-mono text-xs font-semibold text-ink-primary uppercase">
                              {attr.category.replace(/_/g, " ")}
                            </span>
                            {getEvidenceBadge(attr.evidenceState)}
                            {attr.start !== null && attr.end !== null && (
                              <span className="text-[10px] font-mono text-ink-disabled">
                                [{attr.start}:{attr.end}]
                              </span>
                            )}
                          </div>

                          <div className="mt-1 flex items-center gap-2">
                            <span className="text-sm text-ink-primary font-medium">
                              {attr.evidenceState === "negated" ? (
                                <span className="line-through text-rose-400/80">
                                  {attr.sourceText || "Negated feature"}
                                </span>
                              ) : attr.value ? (
                                attr.value
                              ) : (
                                <span className="text-ink-disabled italic text-xs">
                                  Not observed in testimony
                                </span>
                              )}
                            </span>
                            {attr.sourceText && attr.evidenceState !== "negated" && (
                              <span className="text-xs text-ink-tertiary font-mono italic">
                                &ldquo;{attr.sourceText}&rdquo;
                              </span>
                            )}
                          </div>
                        </div>

                        <div className="flex items-center gap-1.5 shrink-0">
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => handleOpenWitnessTrace(undefined, attr.category)}
                            className="h-7 px-2 text-xs text-gold/80 hover:text-gold hover:bg-gold/10"
                            title="Inspect 15-point evidence trace, character offsets, NLP provenance & GCIT conditioning"
                          >
                            <HelpCircle className="w-3.5 h-3.5 mr-1" /> Trace
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => openEditModal(attr)}
                            className="h-7 px-2 text-xs text-ink-tertiary hover:text-ink-primary"
                          >
                            <Edit3 className="w-3.5 h-3.5 mr-1" /> Audit
                          </Button>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Non-Facial Observations (Separated) */}
                {nonFacialObservables.length > 0 && (
                  <div className="mt-3 p-2.5 rounded bg-bg-inset/80 border border-line space-y-1.5">
                    <div className="flex items-center justify-between">
                      <span className="label-technical text-xs text-ink-secondary flex items-center gap-1.5">
                        <Layers className="w-3.5 h-3.5 text-gold" />
                        Non-Facial & Scene Observations ({nonFacialObservables.length})
                      </span>
                      <span className="text-[10px] text-ink-disabled">
                        Excluded from facial geometry
                      </span>
                    </div>
                    {nonFacialObservables.map((nfo, idx) => (
                      <div
                        key={idx}
                        onClick={() => handleHighlightAttribute(nfo)}
                        className="cursor-pointer text-xs flex items-center justify-between py-1 border-t border-line/40 text-ink-primary"
                      >
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-ink-disabled">[OBSERVABLE]</span>
                          <span className="font-medium">{nfo.value}</span>
                          {nfo.sourceText && (
                            <span className="text-ink-tertiary font-mono italic">&ldquo;{nfo.sourceText}&rdquo;</span>
                          )}
                        </div>
                        {getEvidenceBadge(nfo.evidenceState)}
                      </div>
                    ))}
                  </div>
                )}

                {/* Review & Approval Action Bar */}
                <div className="pt-3 border-t border-line flex items-center justify-between">
                  <div className="text-xs text-ink-disabled">
                    Human-in-the-Loop Supervision Mandatory
                  </div>
                  <Button
                    size="sm"
                    variant={reviewStatus === "approved" ? "secondary" : "primary"}
                    onClick={handleApprove}
                    disabled={approving || reviewStatus === "approved"}
                    className="gap-1.5"
                  >
                    <ShieldCheck className="w-4 h-4 text-emerald-400" />
                    {reviewStatus === "approved" ? "Reconstruction Approved" : "Approve Extraction"}
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Investigator Correction Modal / Drawer */}
      {selectedAttrForEdit && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
          <Card className="w-full max-w-lg border-line shadow-2xl">
            <CardHeader className="flex flex-row items-center justify-between border-b border-line pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <Edit3 className="w-4 h-4 text-gold" />
                Audit Attribute: {selectedAttrForEdit.category.toUpperCase()}
              </CardTitle>
              <button
                onClick={() => setSelectedAttrForEdit(null)}
                className="text-ink-disabled hover:text-ink-primary text-sm font-mono"
              >
                ✕
              </button>
            </CardHeader>
            <CardContent className="space-y-4 pt-4">
              <div>
                <label className="label-technical text-ink-tertiary">Original Extracted Span</label>
                <div className="mt-1 p-2 rounded bg-bg-inset border border-line text-sm font-mono text-ink-primary">
                  &ldquo;{selectedAttrForEdit.sourceText || "None (unobserved)"}&rdquo; (State:{" "}
                  {selectedAttrForEdit.evidenceState})
                </div>
              </div>

              <div className="flex items-center gap-2 pt-1">
                <input
                  type="checkbox"
                  id="reject-toggle"
                  checked={editRejected}
                  onChange={(e) => setEditRejected(e.target.checked)}
                  className="rounded border-line"
                />
                <label htmlFor="reject-toggle" className="text-sm font-medium text-rose-400">
                  Reject / Suppress Extraction (Negated or False Detection)
                </label>
              </div>

              {!editRejected && (
                <>
                  <Input
                    label="Corrected Attribute Value"
                    value={editValue}
                    onChange={(e) => setEditValue(e.target.value)}
                    placeholder="Enter investigator corrected descriptor..."
                  />

                  <div>
                    <label className="label-technical text-ink-tertiary">Evidence State</label>
                    <select
                      value={editState}
                      onChange={(e) => setEditState(e.target.value as any)}
                      className="mt-1 w-full h-10 px-3 rounded bg-bg-inset border border-line text-sm text-ink-primary outline-none focus:border-gold/50"
                    >
                      <option value="explicit">Explicit (Confirmed in statement)</option>
                      <option value="uncertain">Uncertain (Hedged by witness)</option>
                      <option value="negated">Negated (Explicitly absent)</option>
                      <option value="contradictory">Contradictory (Conflict noted)</option>
                    </select>
                  </div>
                </>
              )}

              <div>
                <label className="label-technical text-ink-tertiary">
                  Investigator Justification / Audit Reason <span className="text-rose-400">*</span>
                </label>
                <textarea
                  value={editReason}
                  onChange={(e) => setEditReason(e.target.value)}
                  placeholder="Detail the justification for this override for forensic audit compliance..."
                  className="mt-1 w-full h-20 p-2.5 rounded bg-bg-inset border border-line text-sm text-ink-primary outline-none focus:border-gold/50"
                />
              </div>

              <div className="flex items-center justify-end gap-2 pt-2 border-t border-line">
                <Button
                  variant="ghost"
                  onClick={() => setSelectedAttrForEdit(null)}
                  disabled={savingCorrection}
                >
                  Cancel
                </Button>
                <Button
                  onClick={handleSaveCorrection}
                  loading={savingCorrection}
                  disabled={!editReason.trim() || savingCorrection}
                >
                  Save Correction
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Investigator Correction Audit Trail */}
      {correctionsList.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-gold" />
              Forensic Audit Trail & Investigator Corrections ({correctionsList.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-line text-ink-tertiary label-technical">
                    <th className="py-2">Category</th>
                    <th className="py-2">Previous State/Value</th>
                    <th className="py-2">Corrected State/Value</th>
                    <th className="py-2">Investigator Reason</th>
                    <th className="py-2">Timestamp</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line/60">
                  {correctionsList.map((c, i) => (
                    <tr key={i} className="hover:bg-bg-inset/40">
                      <td className="py-2 font-mono font-semibold uppercase">{c.attributeCategory}</td>
                      <td className="py-2 text-ink-disabled line-through">
                        {c.previousValue || "none"} ({c.previousEvidenceState})
                      </td>
                      <td className="py-2 text-gold font-medium">
                        {c.rejected ? "SUPPRESSED" : c.correctedValue} ({c.correctedEvidenceState})
                      </td>
                      <td className="py-2 text-ink-secondary">{c.reason}</td>
                      <td className="py-2 text-ink-disabled font-mono">
                        {formatTimestamp(c.timestamp)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Operational Phase 5: Facial Composite Reconstruction Engine */}
      <Card>
        <CardHeader className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <ScanFace className="w-5 h-5 text-gold" />
              <CardTitle>Facial Composite Reconstruction Engine — Phase 5</CardTitle>
            </div>
            <p className="text-xs text-ink-disabled mt-1">
              Conditional WGAN-GP Generative Synthesis • Projection Conditioning • Identity-Disjoint CelebA Training
            </p>
          </div>
          <div className="flex items-center gap-2">
            <StatusBadge
              tone={reviewStatus === "approved" ? "verified" : "pending"}
            >
              {reviewStatus === "approved"
                ? "Profile Approved for Synthesis"
                : "Approval Required to Synthesize"}
            </StatusBadge>
            {compositesList.length > 0 && (
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-bg-inset border border-line text-ink-tertiary">
                {compositesList.length} {compositesList.length === 1 ? "Version" : "Versions"}
              </span>
            )}
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Generation Controls Toolbar */}
          <div className="p-4 rounded-lg bg-bg-inset border border-line space-y-4">
            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
              {/* Left: Configuration Controls */}
              <div className="flex flex-wrap items-center gap-4 text-xs">
                <div>
                  <label className="block text-[11px] text-ink-disabled font-medium mb-1 uppercase tracking-wider">
                    Variations Count
                  </label>
                  <div className="flex items-center rounded border border-line bg-bg-deepest p-0.5">
                    {[1, 2, 3, 4].map((n) => (
                      <button
                        key={n}
                        type="button"
                        onClick={() => setNumVariations(n)}
                        className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                          numVariations === n
                            ? "bg-gold text-bg-deepest font-semibold"
                            : "text-ink-tertiary hover:text-ink-primary"
                        }`}
                      >
                        {n}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block text-[11px] text-ink-disabled font-medium mb-1 uppercase tracking-wider">
                    Generation Seed
                  </label>
                  <div className="flex items-center gap-1">
                    <Input
                      type="text"
                      value={customSeed}
                      onChange={(e) => setCustomSeed(e.target.value.replace(/[^0-9]/g, ""))}
                      className="h-8 w-24 text-xs font-mono"
                      placeholder="Seed"
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={handleRandomizeSeed}
                      className="h-8 px-2 text-ink-tertiary hover:text-ink-primary"
                      title="Generate random seed"
                    >
                      <RefreshCw className="w-3.5 h-3.5" />
                    </Button>
                  </div>
                </div>

                <div className="flex flex-col justify-end">
                  <span className="text-[11px] text-ink-disabled font-medium mb-1 uppercase tracking-wider">
                    Approved Descriptors
                  </span>
                  <span className="text-xs font-mono text-gold bg-gold/10 px-2 py-1 rounded border border-gold/30">
                    {approvedFacialProfile.length} Active Features
                  </span>
                </div>
              </div>

              {/* Right: Synthesis Action Button */}
              <div className="flex items-center gap-3">
                <Button
                  onClick={handleSynthesizeComposite}
                  disabled={reviewStatus !== "approved" || isGeneratingComposite}
                  loading={isGeneratingComposite}
                  className="gap-2"
                >
                  <Sparkles className="w-4 h-4 text-gold" />
                  {isGeneratingComposite
                    ? "Executing WGAN-GP Synthesis…"
                    : "Synthesize Facial Composite Variations"}
                </Button>
              </div>
            </div>

            {/* Warning when unapproved */}
            {reviewStatus !== "approved" && (
              <div className="flex items-center gap-2 text-xs text-amber-400 bg-amber-500/10 p-2.5 rounded border border-amber-500/30">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                <span>
                  <strong>Forensic Gate Active:</strong> Visual composite synthesis is disabled until the extracted attributes above are verified, corrected if needed, and formally approved.
                </span>
              </div>
            )}
          </div>

          {/* Active Generation Progress */}
          {isGeneratingComposite && (
            <div className="rounded-md border border-gold/40 bg-gold/10 p-4 text-xs text-gold flex items-center gap-3 animate-pulse">
              <RefreshCw className="h-4 w-4 text-gold animate-spin shrink-0" />
              <div className="space-y-0.5">
                <p className="font-semibold text-gold">Executing Accelerated Generator Inference...</p>
                <p className="text-ink-tertiary">
                  Running single batched forward pass on {numVariations} variations across conditioned GCIT dimensions.
                </p>
              </div>
            </div>
          )}

          {/* Composite Error Alert */}
          {compositeError && (
            <div className="rounded-md border border-rose-500/40 bg-rose-500/10 p-4 text-xs text-rose-300 flex items-start gap-3">
              <AlertTriangle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <p className="font-semibold text-rose-200">Synthesis Error</p>
                <p className="text-rose-300/90">{compositeError}</p>
              </div>
            </div>
          )}

          {/* Composite Success Alert */}
          {compositeSuccessMsg && (
            <div className="rounded-md border border-emerald-500/40 bg-emerald-500/10 p-3 text-xs text-emerald-300 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />
                <span>{compositeSuccessMsg}</span>
              </div>
              <button
                onClick={() => setCompositeSuccessMsg(null)}
                className="text-emerald-400 hover:text-emerald-200 text-xs underline"
              >
                Dismiss
              </button>
            </div>
          )}

          {/* Unconditioned Attributes / Modality Warnings */}
          {selectedComposite?.warnings && selectedComposite.warnings.length > 0 && (
            <div className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-xs text-amber-300 space-y-1">
              <div className="flex items-center gap-2 font-semibold text-amber-200">
                <Info className="h-4 w-4 text-amber-400 shrink-0" />
                <span>Conditioning & Modality Warnings</span>
              </div>
              <ul className="list-disc list-inside space-y-0.5 text-amber-300/90 pl-1">
                {selectedComposite.warnings.map((w: string, i: number) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Version History & Comparison Bar */}
          {compositesList.length > 0 && (
            <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-line/60">
              <div className="flex flex-wrap items-center gap-2">
                <span className="label-technical text-ink-disabled mr-1">Composite History:</span>
                {compositesList.map((comp) => {
                  const isSelected = selectedComposite?._id === comp._id;
                  const hasApprovedVar = comp.variations?.some((v: any) => v.isSelected);
                  return (
                    <button
                      key={comp._id}
                      type="button"
                      onClick={() => setSelectedComposite(comp)}
                      className={`px-3 py-1 rounded text-xs font-mono flex items-center gap-1.5 transition-colors border ${
                        isSelected
                          ? "bg-gold/15 text-gold border-gold/40 font-semibold"
                          : "bg-bg-inset text-ink-tertiary border-line hover:text-ink-primary"
                      }`}
                    >
                      <Layers className="w-3 h-3" />
                      <span>Version {comp.version}</span>
                      {hasApprovedVar ? (
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" title="Has primary approved variation" />
                      ) : (
                        <span className="w-1.5 h-1.5 rounded-full bg-slate-500" title="Draft" />
                      )}
                    </button>
                  );
                })}
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  data-testid="open-variation-comparison-btn"
                  onClick={() => handleOpenComparisonModal()}
                  className="gap-1.5 text-xs font-mono border-gold/40 text-gold hover:bg-gold/10"
                >
                  <Columns className="w-3.5 h-3.5 text-gold" />
                  Compare Variations & GCIT Diffs
                  {comparisonTargets.length > 0 ? ` (${comparisonTargets.length})` : ""}
                </Button>

                {compositesList.length > 1 && (
                  <Button
                    type="button"
                    variant="secondary"
                    size="sm"
                    onClick={() => setIsComparing(!isComparing)}
                    className="gap-1.5 text-xs font-mono"
                  >
                    <ArrowLeftRight className="w-3.5 h-3.5 text-gold" />
                    {isComparing ? "Exit Quick Dual View" : "Quick Dual View"}
                  </Button>
                )}
              </div>
            </div>
          )}

          {/* Dual Column Side-by-Side Version Comparison */}
          {isComparing && selectedComposite && compareComposite ? (
            <div className="space-y-4 pt-2">
              <div className="flex items-center justify-between p-3 rounded bg-bg-inset border border-line text-xs">
                <div className="flex items-center gap-2">
                  <ArrowLeftRight className="w-4 h-4 text-gold" />
                  <span className="font-semibold text-ink-primary">
                    Side-by-Side Version Comparison: v{selectedComposite.version} vs v{compareComposite.version}
                  </span>
                </div>
                <div className="flex items-center gap-4 text-ink-tertiary">
                  <div className="flex items-center gap-1.5">
                    <span>Compare with:</span>
                    <select
                      className="h-7 px-2 rounded bg-bg-deepest border border-line text-xs font-mono text-ink-primary"
                      value={compareComposite._id}
                      onChange={(e) => {
                        const target = compositesList.find((c) => c._id === e.target.value);
                        if (target) setCompareComposite(target);
                      }}
                    >
                      {compositesList.map((c) => (
                        <option key={c._id} value={c._id}>
                          Version {c.version} ({formatTimestamp(c.createdAt)})
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Column A */}
                <div className="space-y-3 p-4 rounded-lg bg-bg-inset border border-line">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs font-bold text-gold">
                      VERSION {selectedComposite.version} (Active)
                    </span>
                    <span className="text-[11px] text-ink-disabled font-mono">
                      {formatTimestamp(selectedComposite.createdAt)}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    {selectedComposite.variations?.map((v: any) => (
                      <div key={v.variationIndex} className="space-y-1.5 text-center">
                        <img
                          src={v.dataUrl || v.imageUrl}
                          alt={`v${selectedComposite.version} var${v.variationIndex}`}
                          className="w-full aspect-square object-contain rounded bg-black border border-line"
                          style={{ imageRendering: (v.width || 128) >= 128 ? "auto" : "pixelated" }}
                        />
                        <div className="flex items-center justify-between text-[10px] text-ink-tertiary font-mono px-1">
                          <span>Var #{v.variationIndex}</span>
                          <span>Seed: {v.seed}</span>
                        </div>
                        {v.isSelected && (
                          <span className="inline-block text-[10px] text-emerald-400 font-semibold bg-emerald-500/10 border border-emerald-500/30 rounded px-1.5 py-0.5">
                            Primary Lead
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Column B */}
                <div className="space-y-3 p-4 rounded-lg bg-bg-inset border border-line">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs font-bold text-ink-primary">
                      VERSION {compareComposite.version} (Comparison)
                    </span>
                    <span className="text-[11px] text-ink-disabled font-mono">
                      {formatTimestamp(compareComposite.createdAt)}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    {compareComposite.variations?.map((v: any) => (
                      <div key={v.variationIndex} className="space-y-1.5 text-center">
                        <img
                          src={v.dataUrl || v.imageUrl}
                          alt={`v${compareComposite.version} var${v.variationIndex}`}
                          className="w-full aspect-square object-contain rounded bg-black border border-line"
                          style={{ imageRendering: (v.width || 128) >= 128 ? "auto" : "pixelated" }}
                        />
                        <div className="flex items-center justify-between text-[10px] text-ink-tertiary font-mono px-1">
                          <span>Var #{v.variationIndex}</span>
                          <span>Seed: {v.seed}</span>
                        </div>
                        {v.isSelected && (
                          <span className="inline-block text-[10px] text-emerald-400 font-semibold bg-emerald-500/10 border border-emerald-500/30 rounded px-1.5 py-0.5">
                            Primary Lead
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ) : selectedComposite ? (
            /* Single Active Composite Version Display */
            <div className="space-y-6">
              {/* Header metadata for active composite */}
              <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded bg-bg-inset border border-line text-xs">
                <div className="flex items-center gap-3">
                  <span className="font-mono font-bold text-gold">
                    Version {selectedComposite.version}
                  </span>
                  <span className="text-ink-disabled font-mono">
                    Created: {formatTimestamp(selectedComposite.createdAt)}
                  </span>
                  <span className="text-ink-disabled">•</span>
                  <span className="text-ink-tertiary">
                    {selectedComposite.variations?.length || 0} Variations
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="label-technical text-ink-disabled">Status:</span>
                  <span
                    className={`font-mono text-xs uppercase px-2 py-0.5 rounded border ${
                      selectedComposite.status === "approved"
                        ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
                        : "bg-slate-500/15 text-slate-300 border-slate-500/30"
                    }`}
                  >
                    {selectedComposite.status === "approved"
                      ? "[INVESTIGATOR-APPROVED COMPOSITE]"
                      : "[GENERATED DRAFT COMPOSITE]"}
                  </span>
                </div>
              </div>

              {/* Investigator Semantic Editing Studio */}
              {isEditingComposite && editingTargetVariation && (
                <div className="p-5 rounded-xl border border-gold/40 bg-bg-panel shadow-2xl space-y-6 animate-in fade-in duration-200">
                  {/* Studio Header */}
                  <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-line">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2.5">
                        <Sliders className="w-5 h-5 text-gold" />
                        <h3 className="text-sm font-bold text-ink-primary tracking-wide">
                          Investigator Semantic Editing Studio
                        </h3>
                        <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-gold/15 text-gold border border-gold/30">
                          Phase 5.7
                        </span>
                      </div>
                      <p className="text-xs text-ink-tertiary">
                        Controlled continuous GCIT traversal & empirical hyperplane traversal in Z preserving seed identity.
                      </p>
                    </div>

                    <div className="flex items-center gap-3">
                      {/* Fixed Seed Lock Badge */}
                      <div className="flex items-center gap-1.5 px-3 py-1 rounded bg-bg-deepest border border-emerald-500/40 text-xs font-mono text-emerald-400 shadow-sm">
                        <Lock className="w-3.5 h-3.5 text-emerald-400" />
                        <span>Seed: {editingSeed} (Locked)</span>
                      </div>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={handleCancelEditing}
                        className="h-8 w-8 p-0 text-ink-tertiary hover:text-ink-primary"
                        title="Close Studio"
                      >
                        <X className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>

                  {/* Top: Base Composite Reference & Instructions */}
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4 p-3 rounded-lg bg-bg-inset border border-line items-center">
                    <div className="flex items-center gap-3 md:col-span-1">
                      <img
                        src={editingTargetVariation.dataUrl || editingTargetVariation.imageUrl}
                        alt="Editing Baseline"
                        className="w-16 h-16 rounded object-contain bg-black border border-line"
                        style={{ imageRendering: (editingTargetVariation.width || 128) >= 128 ? "auto" : "pixelated" }}
                      />
                      <div className="text-xs font-mono space-y-0.5">
                        <span className="text-gold font-bold block">Original Baseline</span>
                        <span className="text-ink-tertiary block">Version {selectedComposite.version}</span>
                        <span className="text-ink-disabled block">Var #{editingTargetVariation.variationIndex}</span>
                      </div>
                    </div>

                    <div className="md:col-span-3 text-xs text-ink-secondary space-y-1">
                      <p>
                        <strong>Scientifically Validated Traversal:</strong> Adjust continuous feature intensity without retraining or perturbing the frozen generator checkpoint.
                      </p>
                      <p className="text-[11px] text-ink-tertiary">
                        Continuous GCIT modifiers (<span className="text-gold font-mono">Part A</span>) alter the projection conditioning layer directly. Latent hyperplanes (<span className="text-gold font-mono">Part B</span>) traverse along validated linear SVM normals in Z space.
                      </p>
                    </div>
                  </div>

                  {/* Controls Grid */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Column 1: Continuous GCIT Conditioning (Part A) */}
                    <div className="space-y-4 p-4 rounded-lg bg-bg-inset/70 border border-line">
                      <div className="flex items-center justify-between pb-2 border-b border-line/60">
                        <span className="text-xs font-bold text-gold uppercase tracking-wider flex items-center gap-1.5">
                          <Sliders className="w-3.5 h-3.5" />
                          Part A: Continuous GCIT Traversal (c_i ∈ [-1, +1])
                        </span>
                        <span className="text-[10px] text-ink-disabled font-mono">
                          Direct Projection Conditioning
                        </span>
                      </div>

                      {/* Facial Geometry Controls */}
                      <div className="space-y-3">
                        <span className="text-[11px] font-semibold text-ink-tertiary uppercase tracking-wider block">
                          Facial Geometry & Proportions
                        </span>

                        {/* Nose Size */}
                        <div className="space-y-1">
                          <div className="flex justify-between text-xs font-mono">
                            <span className="text-ink-primary">Nose Size (nose_big)</span>
                            <span className="text-gold">
                              {continuousAdjustments.nose_big > 0 ? `+${continuousAdjustments.nose_big.toFixed(2)}` : continuousAdjustments.nose_big.toFixed(2)}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 text-[10px] text-ink-disabled">
                            <span>Small (-1.0)</span>
                            <input
                              type="range"
                              min="-1.0"
                              max="1.0"
                              step="0.05"
                              value={continuousAdjustments.nose_big}
                              onChange={(e) => handleContinuousChange("nose_big", parseFloat(e.target.value))}
                              className="flex-1 accent-amber-500 cursor-pointer h-1.5 bg-bg-deepest rounded"
                            />
                            <span>Large (+1.0)</span>
                          </div>
                        </div>

                        {/* Pointy Nose */}
                        <div className="space-y-1">
                          <div className="flex justify-between text-xs font-mono">
                            <span className="text-ink-primary">Pointy Nose (nose_pointy)</span>
                            <span className="text-gold">
                              {continuousAdjustments.nose_pointy > 0 ? `+${continuousAdjustments.nose_pointy.toFixed(2)}` : continuousAdjustments.nose_pointy.toFixed(2)}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 text-[10px] text-ink-disabled">
                            <span>Rounded (-1.0)</span>
                            <input
                              type="range"
                              min="-1.0"
                              max="1.0"
                              step="0.05"
                              value={continuousAdjustments.nose_pointy}
                              onChange={(e) => handleContinuousChange("nose_pointy", parseFloat(e.target.value))}
                              className="flex-1 accent-amber-500 cursor-pointer h-1.5 bg-bg-deepest rounded"
                            />
                            <span>Pointy (+1.0)</span>
                          </div>
                        </div>

                        {/* Lip Fullness */}
                        <div className="space-y-1">
                          <div className="flex justify-between text-xs font-mono">
                            <span className="text-ink-primary">Lip Fullness (lips_big)</span>
                            <span className="text-gold">
                              {continuousAdjustments.lips_big > 0 ? `+${continuousAdjustments.lips_big.toFixed(2)}` : continuousAdjustments.lips_big.toFixed(2)}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 text-[10px] text-ink-disabled">
                            <span>Thin (-1.0)</span>
                            <input
                              type="range"
                              min="-1.0"
                              max="1.0"
                              step="0.05"
                              value={continuousAdjustments.lips_big}
                              onChange={(e) => handleContinuousChange("lips_big", parseFloat(e.target.value))}
                              className="flex-1 accent-amber-500 cursor-pointer h-1.5 bg-bg-deepest rounded"
                            />
                            <span>Full (+1.0)</span>
                          </div>
                        </div>

                        {/* Face Oval */}
                        <div className="space-y-1">
                          <div className="flex justify-between text-xs font-mono">
                            <span className="text-ink-primary">Face Shape (face_oval)</span>
                            <span className="text-gold">
                              {continuousAdjustments.face_oval > 0 ? `+${continuousAdjustments.face_oval.toFixed(2)}` : continuousAdjustments.face_oval.toFixed(2)}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 text-[10px] text-ink-disabled">
                            <span>Angular (-1.0)</span>
                            <input
                              type="range"
                              min="-1.0"
                              max="1.0"
                              step="0.05"
                              value={continuousAdjustments.face_oval}
                              onChange={(e) => handleContinuousChange("face_oval", parseFloat(e.target.value))}
                              className="flex-1 accent-amber-500 cursor-pointer h-1.5 bg-bg-deepest rounded"
                            />
                            <span>Oval (+1.0)</span>
                          </div>
                        </div>

                        {/* High Cheekbones */}
                        <div className="space-y-1">
                          <div className="flex justify-between text-xs font-mono">
                            <span className="text-ink-primary">Cheekbones (face_cheekbones)</span>
                            <span className="text-gold">
                              {continuousAdjustments.face_cheekbones > 0 ? `+${continuousAdjustments.face_cheekbones.toFixed(2)}` : continuousAdjustments.face_cheekbones.toFixed(2)}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 text-[10px] text-ink-disabled">
                            <span>Subtle (-1.0)</span>
                            <input
                              type="range"
                              min="-1.0"
                              max="1.0"
                              step="0.05"
                              value={continuousAdjustments.face_cheekbones}
                              onChange={(e) => handleContinuousChange("face_cheekbones", parseFloat(e.target.value))}
                              className="flex-1 accent-amber-500 cursor-pointer h-1.5 bg-bg-deepest rounded"
                            />
                            <span>High (+1.0)</span>
                          </div>
                        </div>
                      </div>

                      {/* Hair Texture & Facial Hair Controls */}
                      <div className="space-y-3 pt-2 border-t border-line/40">
                        <span className="text-[11px] font-semibold text-ink-tertiary uppercase tracking-wider block">
                          Hair Texture & Facial Hair
                        </span>

                        {/* Hair Waviness */}
                        <div className="space-y-1">
                          <div className="flex justify-between text-xs font-mono">
                            <span className="text-ink-primary">Hair Texture (hair_wavy)</span>
                            <span className="text-gold">
                              {continuousAdjustments.hair_wavy > 0 ? `+${continuousAdjustments.hair_wavy.toFixed(2)}` : continuousAdjustments.hair_wavy.toFixed(2)}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 text-[10px] text-ink-disabled">
                            <span>Straight (-1.0)</span>
                            <input
                              type="range"
                              min="-1.0"
                              max="1.0"
                              step="0.05"
                              value={continuousAdjustments.hair_wavy}
                              onChange={(e) => handleContinuousChange("hair_wavy", parseFloat(e.target.value))}
                              className="flex-1 accent-amber-500 cursor-pointer h-1.5 bg-bg-deepest rounded"
                            />
                            <span>Wavy (+1.0)</span>
                          </div>
                        </div>

                        {/* Beard Presence */}
                        <div className="space-y-1">
                          <div className="flex justify-between text-xs font-mono">
                            <span className="text-ink-primary">Beard (facial_no_beard)</span>
                            <span className="text-gold">
                              {continuousAdjustments.facial_no_beard > 0 ? `+${continuousAdjustments.facial_no_beard.toFixed(2)}` : continuousAdjustments.facial_no_beard.toFixed(2)}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 text-[10px] text-ink-disabled">
                            <span>Full Beard (-1.0)</span>
                            <input
                              type="range"
                              min="-1.0"
                              max="1.0"
                              step="0.05"
                              value={continuousAdjustments.facial_no_beard}
                              onChange={(e) => handleContinuousChange("facial_no_beard", parseFloat(e.target.value))}
                              className="flex-1 accent-amber-500 cursor-pointer h-1.5 bg-bg-deepest rounded"
                            />
                            <span>Clean Shaven (+1.0)</span>
                          </div>
                        </div>

                        {/* 5 O'Clock Stubble */}
                        <div className="space-y-1">
                          <div className="flex justify-between text-xs font-mono">
                            <span className="text-ink-primary">5 O'Clock Shadow (facial_stubble)</span>
                            <span className="text-gold">
                              {continuousAdjustments.facial_stubble > 0 ? `+${continuousAdjustments.facial_stubble.toFixed(2)}` : continuousAdjustments.facial_stubble.toFixed(2)}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 text-[10px] text-ink-disabled">
                            <span>Clean (-1.0)</span>
                            <input
                              type="range"
                              min="-1.0"
                              max="1.0"
                              step="0.05"
                              value={continuousAdjustments.facial_stubble}
                              onChange={(e) => handleContinuousChange("facial_stubble", parseFloat(e.target.value))}
                              className="flex-1 accent-amber-500 cursor-pointer h-1.5 bg-bg-deepest rounded"
                            />
                            <span>Heavy Stubble (+1.0)</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Column 2: Latent Hyperplane Traversal (Part B) */}
                    <div className="space-y-4 p-4 rounded-lg bg-bg-inset/70 border border-line">
                      <div className="flex items-center justify-between pb-2 border-b border-line/60">
                        <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider flex items-center gap-1.5">
                          <Sparkles className="w-3.5 h-3.5" />
                          Part B: Latent Hyperplanes in Z (α ∈ [-2.0, +2.0])
                        </span>
                        <span className="text-[10px] text-emerald-400/80 font-mono">
                          {latentDirectionsInfo?.num_supported_directions || 5} Validated Directions ({latentDirectionsInfo?.num_rejected_directions || 15} Rejected)
                        </span>
                      </div>

                      <p className="text-[11px] text-ink-tertiary">
                        Linear SVM normal shifts $z&apos; = z + \alpha n_i$ in R^100. Evaluated against the independent frozen forensic evaluator across 50 held-out test seeds.
                      </p>

                      <div className="space-y-3.5">
                        {/* 1. Blond Hair */}
                        <div className="space-y-1 p-2.5 rounded bg-bg-deepest border border-line">
                          <div className="flex justify-between text-xs font-mono">
                            <span className="text-ink-primary font-semibold">Blond Hair (hair_blond)</span>
                            <span className="text-gold font-bold">
                              {latentTraversals.hair_blond > 0 ? `+${latentTraversals.hair_blond.toFixed(2)}` : latentTraversals.hair_blond.toFixed(2)}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 text-[10px] text-ink-disabled">
                            <span>Dark (-2.0)</span>
                            <input
                              type="range"
                              min="-2.0"
                              max="2.0"
                              step="0.1"
                              value={latentTraversals.hair_blond}
                              onChange={(e) => handleLatentChange("hair_blond", parseFloat(e.target.value))}
                              className="flex-1 accent-emerald-500 cursor-pointer h-1.5 bg-bg-inset rounded"
                            />
                            <span>Blond (+2.0)</span>
                          </div>
                          <div className="text-[10px] text-ink-disabled font-mono flex justify-between pt-0.5">
                            <span>Effect: +0.467 • Consistency: 100%</span>
                            <span>Selectivity: 1.34×</span>
                          </div>
                        </div>

                        {/* 2. Gray Hair */}
                        <div className="space-y-1 p-2.5 rounded bg-bg-deepest border border-line">
                          <div className="flex justify-between text-xs font-mono">
                            <span className="text-ink-primary font-semibold">Gray / Aging Hair (hair_gray)</span>
                            <span className="text-gold font-bold">
                              {latentTraversals.hair_gray > 0 ? `+${latentTraversals.hair_gray.toFixed(2)}` : latentTraversals.hair_gray.toFixed(2)}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 text-[10px] text-ink-disabled">
                            <span>Young (-2.0)</span>
                            <input
                              type="range"
                              min="-2.0"
                              max="2.0"
                              step="0.1"
                              value={latentTraversals.hair_gray}
                              onChange={(e) => handleLatentChange("hair_gray", parseFloat(e.target.value))}
                              className="flex-1 accent-emerald-500 cursor-pointer h-1.5 bg-bg-inset rounded"
                            />
                            <span>Gray (+2.0)</span>
                          </div>
                          <div className="text-[10px] text-ink-disabled font-mono flex justify-between pt-0.5">
                            <span>Effect: +0.509 • Consistency: 100%</span>
                            <span>Selectivity: 1.41×</span>
                          </div>
                        </div>

                        {/* 3. Bangs */}
                        <div className="space-y-1 p-2.5 rounded bg-bg-deepest border border-line">
                          <div className="flex justify-between text-xs font-mono">
                            <span className="text-ink-primary font-semibold">Bangs / Fringe (hair_bangs)</span>
                            <span className="text-gold font-bold">
                              {latentTraversals.hair_bangs > 0 ? `+${latentTraversals.hair_bangs.toFixed(2)}` : latentTraversals.hair_bangs.toFixed(2)}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 text-[10px] text-ink-disabled">
                            <span>Receding (-2.0)</span>
                            <input
                              type="range"
                              min="-2.0"
                              max="2.0"
                              step="0.1"
                              value={latentTraversals.hair_bangs}
                              onChange={(e) => handleLatentChange("hair_bangs", parseFloat(e.target.value))}
                              className="flex-1 accent-emerald-500 cursor-pointer h-1.5 bg-bg-inset rounded"
                            />
                            <span>Full Bangs (+2.0)</span>
                          </div>
                          <div className="text-[10px] text-ink-disabled font-mono flex justify-between pt-0.5">
                            <span>Effect: +0.375 • Consistency: 96%</span>
                            <span>Selectivity: 2.16×</span>
                          </div>
                        </div>

                        {/* 4. Goatee */}
                        <div className="space-y-1 p-2.5 rounded bg-bg-deepest border border-line">
                          <div className="flex justify-between text-xs font-mono">
                            <span className="text-ink-primary font-semibold">Goatee (facial_goatee)</span>
                            <span className="text-gold font-bold">
                              {latentTraversals.facial_goatee > 0 ? `+${latentTraversals.facial_goatee.toFixed(2)}` : latentTraversals.facial_goatee.toFixed(2)}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 text-[10px] text-ink-disabled">
                            <span>Clean (-2.0)</span>
                            <input
                              type="range"
                              min="-2.0"
                              max="2.0"
                              step="0.1"
                              value={latentTraversals.facial_goatee}
                              onChange={(e) => handleLatentChange("facial_goatee", parseFloat(e.target.value))}
                              className="flex-1 accent-emerald-500 cursor-pointer h-1.5 bg-bg-inset rounded"
                            />
                            <span>Goatee (+2.0)</span>
                          </div>
                          <div className="text-[10px] text-ink-disabled font-mono flex justify-between pt-0.5">
                            <span>Effect: +0.391 • Consistency: 98%</span>
                            <span>Selectivity: 1.23×</span>
                          </div>
                        </div>

                        {/* 5. Eyeglasses */}
                        <div className="space-y-1 p-2.5 rounded bg-bg-deepest border border-line">
                          <div className="flex justify-between text-xs font-mono">
                            <span className="text-ink-primary font-semibold">Eyeglasses (glasses)</span>
                            <span className="text-gold font-bold">
                              {latentTraversals.glasses > 0 ? `+${latentTraversals.glasses.toFixed(2)}` : latentTraversals.glasses.toFixed(2)}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 text-[10px] text-ink-disabled">
                            <span>No Glasses (-2.0)</span>
                            <input
                              type="range"
                              min="-2.0"
                              max="2.0"
                              step="0.1"
                              value={latentTraversals.glasses}
                              onChange={(e) => handleLatentChange("glasses", parseFloat(e.target.value))}
                              className="flex-1 accent-emerald-500 cursor-pointer h-1.5 bg-bg-inset rounded"
                            />
                            <span>Glasses (+2.0)</span>
                          </div>
                          <div className="text-[10px] text-ink-disabled font-mono flex justify-between pt-0.5">
                            <span>Effect: +0.448 • Consistency: 100%</span>
                            <span>Selectivity: 1.36×</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Forensic Boundary: Unsupported Features Notice */}
                  <div className="p-3.5 rounded-lg border border-amber-500/30 bg-amber-500/5 text-xs text-ink-secondary space-y-1.5">
                    <div className="flex items-center gap-2 font-bold text-amber-400">
                      <AlertTriangle className="w-4 h-4 text-amber-400" />
                      <span>Forensic Boundary: Scientific Validation & Unsupported Attributes</span>
                    </div>
                    <p className="text-[11px] leading-relaxed text-ink-tertiary">
                      <strong>15 Rejected Latent Dimensions:</strong> Latent directions for facial geometry (nose, lips, face shape) in latent space R^100 exhibited severe entanglement where unintended side-effects exceeded intended changes. For geometry, always use <em>Continuous GCIT Traversal (Part A)</em>.
                    </p>
                    <p className="text-[11px] leading-relaxed text-ink-tertiary">
                      <strong>Strictly Unsupported Features:</strong> <em>Smiling / Facial Expression</em> (unannotated in GCIT, unscored by evaluator; cannot be fabricated), <em>Continuous Hair Length</em> (only discrete bald/bangs labels exist), and <em>Arbitrary 3D Mesh Geometry</em> (the model is a 2D planar projection generator with no 3D landmarks).
                    </p>
                  </div>

                  {/* Notes & Actions Bar */}
                  <div className="space-y-3 pt-3 border-t border-line">
                    <div className="space-y-1">
                      <label className="text-xs font-mono text-ink-tertiary">
                        Investigator Adjustment Rationale & Notes (Audit Log):
                      </label>
                      <Input
                        placeholder="e.g. Witness recalled suspect had higher cheekbones and thinner lips..."
                        value={editNotes}
                        onChange={(e) => setEditNotes(e.target.value)}
                        className="text-xs"
                      />
                    </div>

                    {editError && (
                      <div className="p-2.5 rounded bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4 shrink-0" />
                        <span>{editError}</span>
                      </div>
                    )}

                    <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
                      <Button
                        type="button"
                        variant="secondary"
                        size="sm"
                        onClick={handleResetAdjustments}
                        disabled={isSubmittingEdit}
                        className="text-xs gap-1.5"
                      >
                        <RotateCcw className="w-3.5 h-3.5" />
                        Reset All Adjustments
                      </Button>

                      <div className="flex items-center gap-2">
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          onClick={handleCancelEditing}
                          disabled={isSubmittingEdit}
                          className="text-xs"
                        >
                          Cancel
                        </Button>
                        <Button
                          type="button"
                          variant="primary"
                          size="sm"
                          onClick={handleApplySemanticEdit}
                          disabled={isSubmittingEdit}
                          className="text-xs gap-1.5"
                        >
                          {isSubmittingEdit ? (
                            <>
                              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                              Synthesizing Version {selectedComposite.version + 1}...
                            </>
                          ) : (
                            <>
                              <Wand2 className="w-3.5 h-3.5" />
                              Save as Composite Version {selectedComposite.version + 1}
                            </>
                          )}
                        </Button>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Variations Gallery Grid */}
              <div>
                <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                  <h4 className="text-xs font-semibold text-ink-primary uppercase tracking-wider">
                    Generated Variations Gallery
                  </h4>
                  <div className="flex items-center gap-2">
                    {comparisonTargets.length > 0 && (
                      <button
                        type="button"
                        onClick={() => setComparisonTargets([])}
                        className="text-[11px] font-mono text-ink-tertiary hover:text-ink-primary underline"
                      >
                        Clear Queue ({comparisonTargets.length})
                      </button>
                    )}
                    <Button
                      type="button"
                      variant="secondary"
                      size="sm"
                      onClick={() => handleOpenComparisonModal()}
                      className="h-7 text-xs gap-1.5 font-mono"
                    >
                      <Columns className="w-3.5 h-3.5 text-gold" />
                      {comparisonTargets.length >= 2
                        ? `Compare ${comparisonTargets.length} Queued Variations`
                        : "Side-by-Side Attribute & Seed Comparison"}
                    </Button>
                  </div>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  {selectedComposite.variations?.map((v: any) => {
                    const isPrimary =
                      v.isSelected ||
                      selectedComposite.selectedVariation === v.variationIndex ||
                      selectedComposite.selectedVariationIndex === v.variationIndex;
                    const isQueuedForCompare = comparisonTargets.some(
                      (t) =>
                        t.compositeId === selectedComposite._id &&
                        t.variationIndex === v.variationIndex
                    );
                    return (
                      <div
                        key={v.variationIndex}
                        className={`p-3 rounded-lg border transition-all flex flex-col justify-between space-y-3 ${
                          isPrimary
                            ? "bg-bg-inset border-emerald-500/40 shadow-sm shadow-emerald-500/10"
                            : isQueuedForCompare
                            ? "bg-bg-inset border-gold/50 shadow-sm shadow-gold/10"
                            : "bg-bg-inset/70 border-line hover:border-line/90"
                        }`}
                      >
                        <div className="space-y-2">
                          {/* Image rendering: standard smooth interpolation for >=128px, pixelated for legacy <128px */}
                          <div
                            className="relative group aspect-square rounded overflow-hidden bg-black border border-line flex items-center justify-center cursor-pointer"
                            onClick={() => setInspectedVariation(v)}
                            title="Click to inspect enlarged at 4× / 6× / 8×"
                          >
                            <img
                              src={v.dataUrl || v.imageUrl}
                              alt={`Composite variation ${v.variationIndex}`}
                              className="w-full h-full object-contain transition-transform group-hover:scale-105"
                              style={{ imageRendering: (v.width || 128) >= 128 ? "auto" : "pixelated" }}
                            />
                            <div className="absolute top-1.5 left-1.5">
                              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-bg-deepest/90 border border-line text-ink-primary">
                                #{v.variationIndex}
                              </span>
                            </div>
                            <div className="absolute top-1.5 right-1.5">
                              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-bg-deepest/90 border border-line text-gold">
                                Seed: {v.seed}
                              </span>
                            </div>
                            {/* Hover overlay hint */}
                            <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-1.5 text-xs text-white font-mono pointer-events-none">
                              <Maximize2 className="w-4 h-4 text-gold" />
                              <span>Inspect (4× / 8×)</span>
                            </div>
                          </div>

                          {/* Status Badge */}
                          <div className="pt-1">
                            {isPrimary ? (
                              <span className="w-full justify-center px-2 py-1 rounded text-[11px] font-semibold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 flex items-center gap-1.5">
                                <ShieldCheck className="w-3.5 h-3.5" />
                                [INVESTIGATOR-APPROVED COMPOSITE]
                              </span>
                            ) : (
                              <span className="w-full justify-center px-2 py-1 rounded text-[11px] font-medium bg-bg-deepest text-ink-tertiary border border-line flex items-center gap-1.5">
                                <Sparkles className="w-3 h-3 text-gold/60" />
                                [GENERATED DRAFT COMPOSITE]
                              </span>
                            )}
                          </div>

                          {/* Dimensions & file details */}
                          <div className="flex items-center justify-between text-[10px] font-mono text-ink-disabled px-1">
                            <span>{v.width || 64}x{v.height || 64} px (Native)</span>
                            <span>PNG (Lossless)</span>
                          </div>
                        </div>

                        {/* Actions Toolbar */}
                        <div className="pt-2 border-t border-line/60 space-y-2">
                          <div className="flex items-center gap-2">
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              onClick={() => setInspectedVariation(v)}
                              className="flex-1 text-xs gap-1 h-8"
                              title="Enlarge and inspect with pixel grid"
                            >
                              <Maximize2 className="w-3.5 h-3.5 text-gold" />
                              Inspect
                            </Button>
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              onClick={() =>
                                toggleComparisonCandidate(
                                  selectedComposite._id,
                                  v.variationIndex
                                )
                              }
                              data-testid={`queue-compare-var-${v.variationIndex}`}
                              className={`text-xs gap-1 h-8 px-2 ${
                                isQueuedForCompare
                                  ? "bg-gold/20 text-gold border border-gold/40 font-semibold"
                                  : "text-ink-tertiary hover:text-ink-primary"
                              }`}
                              title="Queue variation for side-by-side comparison"
                            >
                              <Columns className="w-3.5 h-3.5" />
                              {isQueuedForCompare ? "Queued" : "Compare"}
                            </Button>
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              onClick={() => handleDownloadImage(v)}
                              className="text-xs gap-1 h-8 px-2"
                              title="Download lossless PNG"
                            >
                              <Download className="w-3.5 h-3.5 text-ink-tertiary hover:text-ink-primary" />
                            </Button>
                          </div>

                          <Button
                            type="button"
                            variant="secondary"
                            size="sm"
                            onClick={() => handleStartEditing(v)}
                            className={`w-full text-xs gap-1.5 border transition-all ${
                              isEditingComposite && editingTargetVariation?.variationIndex === v.variationIndex
                                ? "bg-gold/20 text-gold border-gold font-semibold"
                                : "hover:border-gold/50 text-gold/90"
                            }`}
                          >
                            <Sliders className="w-3.5 h-3.5 text-gold" />
                            {isEditingComposite && editingTargetVariation?.variationIndex === v.variationIndex
                              ? "Active in Editing Studio"
                              : "Semantic Editing Studio"}
                          </Button>

                          {isPrimary ? (
                            <div className="text-center text-xs text-emerald-400 font-medium py-1 flex items-center justify-center gap-1.5">
                              <CheckCheck className="w-4 h-4" />
                              Selected as Primary Lead
                            </div>
                          ) : (
                            <Button
                              type="button"
                              variant="secondary"
                              size="sm"
                              onClick={() => handleSelectVariation(v.variationIndex)}
                              disabled={isSelectingVariation}
                              className="w-full text-xs gap-1.5"
                            >
                              <Check className="w-3.5 h-3.5" />
                              Select as Primary Composite
                            </Button>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Conditioning Vector & Model Provenance Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                {/* Conditioned GCIT Attributes Breakdown */}
                <div className="p-3.5 rounded-lg bg-bg-inset border border-line space-y-2 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-ink-primary label-technical flex items-center gap-1.5">
                      <Sliders className="w-3.5 h-3.5 text-gold" />
                      Conditioned GCIT Attributes
                    </span>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => handleOpenCompositeTrace()}
                        className="text-xs text-gold hover:underline flex items-center gap-1 font-mono"
                        title="Inspect full 15-point evidence trace, 20-D GCIT conditioning vector & cryptographic image hashes"
                      >
                        <ShieldCheck className="w-3.5 h-3.5" /> Explain Lineage
                      </button>
                      <span className="text-[10px] text-ink-disabled font-mono">
                        {selectedComposite.conditionedAttributes?.length || 0} Active
                      </span>
                    </div>
                  </div>

                  {selectedComposite.conditionedAttributes &&
                  selectedComposite.conditionedAttributes.length > 0 ? (
                    <div className="flex flex-wrap gap-1.5 pt-1">
                      {selectedComposite.conditionedAttributes.map((attr: string, idx: number) => (
                        <span
                          key={idx}
                          className="px-2 py-0.5 rounded text-[11px] font-mono bg-gold/10 text-gold border border-gold/30"
                        >
                          {attr}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <p className="text-xs text-ink-disabled italic">
                      Zero attributes actively conditioned (neutral unconditional baseline).
                    </p>
                  )}

                  <div className="pt-2 text-[11px] text-ink-disabled">
                    <strong>Unconditioned Features:</strong> Unmentioned attributes remain unspecified (c = 0.0, mask = 0.0). They are drawn stochastically from the dataset prior and are neither guaranteed present nor guaranteed absent or suppressed.
                  </div>
                </div>

                {/* Generator Model Provenance */}
                <div className="p-3.5 rounded-lg bg-bg-inset border border-line space-y-2 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-ink-primary label-technical flex items-center gap-1.5">
                      <Cpu className="w-3.5 h-3.5 text-gold" />
                      Generator Model Provenance
                    </span>
                    <span className="text-[10px] text-emerald-400 font-mono">
                      Phase 5.4 Checkpoint Verified
                    </span>
                  </div>

                  <div className="space-y-1 font-mono text-[11px] text-ink-tertiary pt-1">
                    <div className="flex justify-between">
                      <span className="text-ink-disabled">Architecture:</span>
                      <span className="text-ink-primary">
                        {selectedComposite.generatorMetadata?.architecture || "Conditional WGAN-GP (Projection)"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-ink-disabled">Parameters:</span>
                      <span className="text-ink-primary">
                        {selectedComposite.generatorMetadata?.parameterCount?.toLocaleString() || "4,102,336"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-ink-disabled">Checkpoint SHA-256:</span>
                      <span
                        className="text-gold truncate max-w-[190px]"
                        title={selectedComposite.generatorMetadata?.checkpointSha256}
                      >
                        {selectedComposite.generatorMetadata?.checkpointSha256 || "a1714389d0d1eddad10fae74a736ca41de5e230eada095e6868e5daeeb2a1964"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-ink-disabled">Device / Backend:</span>
                      <span className="text-ink-primary uppercase">
                        {selectedComposite.generatorMetadata?.device || "PyTorch"}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* Empty State: No Composite Synthesized Yet */
            <div className="rounded-lg border-2 border-dashed border-line bg-bg-inset p-8 text-center space-y-3">
              <ScanFace className="w-12 h-12 text-gold/40 mx-auto" />
              <div>
                <h4 className="text-sm font-semibold text-ink-primary uppercase tracking-wider">
                  No Facial Composites Synthesized Yet
                </h4>
                <p className="text-xs text-ink-tertiary mt-1 max-w-md mx-auto">
                  When the witness statement is analyzed and approved by the investigator, click &ldquo;Synthesize Facial Composite Variations&rdquo; above to generate evidence-grounded forensic facial composites at native high resolution.
                </p>
              </div>
              <div className="inline-block text-[11px] text-ink-disabled border border-line rounded px-3 py-1 bg-bg-deepest font-mono">
                Generator Checkpoint: a1714389... (Phase 5.4 Verified)
              </div>
            </div>
          )}

          {/* Mandatory Forensic Science Disclaimer */}
          <div className="p-3 rounded-lg border border-line/80 bg-bg-deepest text-[11px] text-ink-disabled leading-relaxed space-y-1">
            <div className="font-semibold text-ink-secondary flex items-center gap-1.5">
              <ShieldCheck className="w-3.5 h-3.5 text-gold" />
              Forensic Science Standard Notice & Operational Boundaries
            </div>
            <p>
              Reconstructed facial composites are investigative approximation aids generated solely to assist law enforcement in developing investigative leads. Composites must never be represented or utilized as biometric identification evidence in judicial proceedings or arrest determinations. Unmentioned facial attributes remain unconditioned and do not reflect ground-truth identity.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Interactive Composite Inspection & Pixel Grid Modal */}
      {inspectedVariation && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="bg-bg-panel border border-line rounded-xl max-w-2xl w-full max-h-[92vh] flex flex-col shadow-2xl overflow-hidden animate-in fade-in duration-150">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-line bg-bg-inset">
              <div className="flex items-center gap-2.5">
                <ScanFace className="w-5 h-5 text-gold" />
                <div>
                  <h3 className="text-sm font-bold text-ink-primary">
                    Composite Inspection — Variation #{inspectedVariation.variationIndex}
                  </h3>
                  <p className="text-[11px] font-mono text-ink-tertiary">
                    Seed: {inspectedVariation.seed} • Version {selectedComposite?.version || 1}
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setInspectedVariation(null)}
                className="p-1 rounded-md text-ink-tertiary hover:text-ink-primary hover:bg-bg-deepest transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Toolbar */}
            {(() => {
              const nativeWidth = inspectedVariation.width || 128;
              const nativeHeight = inspectedVariation.height || 128;
              const displayDim = inspectZoomScale === 4 ? 256 : inspectZoomScale === 6 ? 384 : 512;
              const zoomFactor = displayDim / nativeWidth;

              return (
                <>
                  <div className="p-3 border-b border-line/60 bg-bg-deepest flex flex-wrap items-center justify-between gap-3 text-xs">
                    {/* Zoom Scale Controls */}
                    <div className="flex items-center gap-2">
                      <span className="text-ink-disabled font-mono text-[11px]">Display Zoom:</span>
                      <button
                        type="button"
                        onClick={() => setInspectZoomScale(4)}
                        className={`px-2.5 py-1 rounded text-xs font-mono transition-colors border ${
                          inspectZoomScale === 4
                            ? "bg-gold/20 text-gold border-gold/50 font-bold"
                            : "bg-bg-inset text-ink-tertiary border-line hover:text-ink-primary"
                        }`}
                      >
                        {nativeWidth === 128 ? "2× (256px)" : "4× (256px)"}
                      </button>
                      <button
                        type="button"
                        onClick={() => setInspectZoomScale(6)}
                        className={`px-2.5 py-1 rounded text-xs font-mono transition-colors border ${
                          inspectZoomScale === 6
                            ? "bg-gold/20 text-gold border-gold/50 font-bold"
                            : "bg-bg-inset text-ink-tertiary border-line hover:text-ink-primary"
                        }`}
                      >
                        {nativeWidth === 128 ? "3× (384px)" : "6× (384px)"}
                      </button>
                      <button
                        type="button"
                        onClick={() => setInspectZoomScale(8)}
                        className={`px-2.5 py-1 rounded text-xs font-mono transition-colors border ${
                          inspectZoomScale === 8
                            ? "bg-gold/20 text-gold border-gold/50 font-bold"
                            : "bg-bg-inset text-ink-tertiary border-line hover:text-ink-primary"
                        }`}
                      >
                        {nativeWidth === 128 ? "4× (512px)" : "8× (512px)"}
                      </button>
                    </div>

                    {/* Grid & Download */}
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => setShowPixelGrid(!showPixelGrid)}
                        className={`px-2.5 py-1 rounded text-xs font-mono flex items-center gap-1.5 transition-colors border ${
                          showPixelGrid
                            ? "bg-gold/20 text-gold border-gold/50 font-semibold"
                            : "bg-bg-inset text-ink-tertiary border-line hover:text-ink-primary"
                        }`}
                        title={`Toggle ${nativeWidth}×${nativeHeight} forensic pixel inspection grid`}
                      >
                        <Grid className="w-3.5 h-3.5" />
                        Pixel Grid: {showPixelGrid ? "ON" : "OFF"}
                      </button>

                      <Button
                        type="button"
                        variant="secondary"
                        size="sm"
                        onClick={() => handleDownloadImage(inspectedVariation)}
                        className="gap-1.5 text-xs font-mono h-7"
                      >
                        <Download className="w-3.5 h-3.5 text-gold" />
                        Download PNG
                      </Button>
                    </div>
                  </div>

                  {/* Modal Body */}
                  <div className="p-6 overflow-y-auto space-y-4 flex flex-col items-center justify-center bg-bg-deepest/60">
                    {/* Image Preview Canvas */}
                    <div
                      className="relative rounded border-2 border-line bg-black shadow-inner flex items-center justify-center overflow-hidden"
                      style={{
                        width: `${displayDim}px`,
                        height: `${displayDim}px`,
                        maxWidth: "100%",
                      }}
                    >
                      <img
                        src={inspectedVariation.dataUrl || inspectedVariation.imageUrl}
                        alt={`Variation ${inspectedVariation.variationIndex}`}
                        className="w-full h-full object-contain"
                        style={{ imageRendering: nativeWidth >= 128 ? "auto" : "pixelated" }}
                      />
                      {showPixelGrid && (
                        <div
                          className="absolute inset-0 pointer-events-none opacity-25"
                          style={{
                            backgroundImage: `linear-gradient(to right, rgba(255,255,255,0.4) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,0.4) 1px, transparent 1px)`,
                            backgroundSize: `${displayDim / nativeWidth}px ${displayDim / nativeHeight}px`,
                          }}
                        />
                      )}
                    </div>

                    {/* Resolution & Interpolation Disclosure */}
                    <div className="w-full max-w-lg p-3 rounded-lg border border-line bg-bg-panel text-[11px] text-ink-secondary space-y-1 text-center">
                      <div className="font-semibold text-gold flex items-center justify-center gap-1.5">
                        <Info className="w-3.5 h-3.5" />
                        Scientific Display Notice
                      </div>
                      <p className="text-ink-tertiary">
                        Native model resolution is strictly <strong>{nativeWidth} × {nativeHeight} pixels</strong> ({nativeWidth >= 128 ? "128×128 High-Resolution Upgrade — 16,384 px" : "Phase 5.4 64×64 baseline"}).
                        Enlarged view scale is {zoomFactor.toFixed(1).replace(/\.0$/, "")}× ({displayDim}×{displayDim}px). {nativeWidth >= 128 ? "Standard smooth browser interpolation applied." : "Nearest-neighbor pixel rendering preserved for 64×64 baseline."}
                      </p>
                    </div>

                    {/* Technical Specifications */}
                    <div className="w-full max-w-lg grid grid-cols-2 gap-2 text-[11px] font-mono p-3 rounded-lg bg-bg-inset border border-line">
                      <div>
                        <span className="text-ink-disabled block">Native Resolution:</span>
                        <span className="text-ink-primary font-semibold">{nativeWidth} × {nativeHeight} ({nativeWidth >= 128 ? "4× Detail Gain" : "8-bit RGB"})</span>
                      </div>
                      <div>
                        <span className="text-ink-disabled block">Encoding / Format:</span>
                        <span className="text-ink-primary font-semibold">Lossless PNG</span>
                      </div>
                      <div>
                        <span className="text-ink-disabled block">Seed:</span>
                        <span className="text-gold font-semibold">{inspectedVariation.seed}</span>
                      </div>
                      <div>
                        <span className="text-ink-disabled block">Variation:</span>
                        <span className="text-ink-primary font-semibold">#{inspectedVariation.variationIndex}</span>
                      </div>
                    </div>
                  </div>
                </>
              );
            })()}

            {/* Footer */}
            <div className="p-3 border-t border-line bg-bg-inset flex items-center justify-between">
              <div className="text-xs text-ink-tertiary font-mono">
                {inspectedVariation.isSelected || selectedComposite?.selectedVariation === inspectedVariation.variationIndex ? (
                  <span className="text-emerald-400 flex items-center gap-1 font-semibold">
                    <CheckCheck className="w-4 h-4" />
                    Currently Selected as Primary Lead
                  </span>
                ) : (
                  <span>Draft Composite Candidate</span>
                )}
              </div>

              <div className="flex items-center gap-2">
                {!(inspectedVariation.isSelected || selectedComposite?.selectedVariation === inspectedVariation.variationIndex) && (
                  <Button
                    type="button"
                    variant="primary"
                    size="sm"
                    onClick={async () => {
                      await handleSelectVariation(inspectedVariation.variationIndex);
                      setInspectedVariation((prev: any) => ({ ...prev, isSelected: true }));
                    }}
                    disabled={isSelectingVariation}
                    className="text-xs gap-1.5"
                  >
                    <Check className="w-3.5 h-3.5" />
                    Set as Primary Lead
                  </Button>
                )}
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  onClick={() => setInspectedVariation(null)}
                  className="text-xs"
                >
                  Close
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Phase 7.0 Evidence Traceability & Explainability (XAI) Modal */}
      <EvidenceExplainabilityModal
        isOpen={traceModalOpen}
        onClose={() => setTraceModalOpen(false)}
        traceData={traceData}
        type={traceType}
        initialCategory={traceInitialCategory}
        loading={loadingTrace}
      />

      {/* Phase 7.1 Multi-Variation Comparison & Attribute Difference Modal */}
      <MultiVariationComparisonModal
        isOpen={comparisonModalOpen}
        onClose={() => setComparisonModalOpen(false)}
        caseId={selectedCaseId}
        compositesList={compositesList}
        initialCompositeId={selectedComposite?._id}
        initialTargets={comparisonTargets.length >= 2 ? comparisonTargets : undefined}
        onVariationSelected={(updatedComp) => {
          if (!updatedComp || !updatedComp._id) return;
          setCompositesList((prev) =>
            prev.map((c) => (c._id === updatedComp._id ? { ...c, ...updatedComp } : c))
          );
          if (selectedComposite?._id === updatedComp._id) {
            setSelectedComposite((prev: any) => ({ ...prev, ...updatedComp }));
          }
        }}
        onOpenExplainability={(compId, category) => {
          handleOpenCompositeTrace(compId, category);
        }}
      />
    </div>
  );
}
