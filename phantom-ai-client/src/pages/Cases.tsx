import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search,
  Plus,
  Columns,
  ShieldCheck,
  ChevronDown,
  ChevronUp,
  Layers,
  ExternalLink,
  CheckCircle2,
  ScanFace,
  Archive,
  RotateCcw,
  Trash2,
  Lock,
  User,
  AlertTriangle,
  X,
  FileCheck2,
} from "lucide-react";

import { Card } from "../components/ui/Card";
import { Input } from "../components/ui/Input";
import { Button } from "../components/ui/Button";
import { StatusBadge } from "../components/ui/Badge";
import type { BadgeStatus } from "../components/ui/Badge";
import { EvidenceTag } from "../components/ui/EvidenceTag";
import { EmptyState } from "../components/ui/EmptyState";
import { PageTransition } from "../components/layout/PageTransition";
import { MultiVariationComparisonModal } from "../components/MultiVariationComparisonModal";
import { EvidenceExplainabilityModal } from "../components/EvidenceExplainabilityModal";
import { CaseCollaborationWorkspace } from "../components/CaseCollaborationWorkspace";

import {
  getAllCases,
  createCase,
  resolveCase,
  reopenCase,
  archiveCase,
  restoreCase,
  deleteCase,
} from "../services/caseService";
import {
  getCaseComposites,
  getCompositeTrace,
} from "../services/imageService";
import type {
  CompositeTraceResponse,
  WitnessDescriptionTraceResponse,
} from "../services/imageService";

type CaseViewFilter =
  | "active"
  | "assigned"
  | "under_review"
  | "resolved"
  | "archived"
  | "all";

interface LifecycleModalState {
  type: "resolve" | "reopen" | "archive" | "restore" | "delete" | null;
  caseItem: any | null;
  isOpen: boolean;
}

export default function Cases() {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [cases, setCases] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeView, setActiveView] = useState<CaseViewFilter>("active");

  // New Case Creation State
  const [isCreating, setIsCreating] = useState(false);
  const [newTitle, setNewTitle] = useState("");

  // Expanded Case Detail & Composites State
  const [expandedCaseId, setExpandedCaseId] = useState<string | null>(null);
  const [caseCompositesMap, setCaseCompositesMap] = useState<Record<string, any[]>>({});
  const [loadingCaseComposites, setLoadingCaseComposites] = useState<string | null>(null);

  // Phase 7.1 Multi-Variation Comparison Modal State
  const [comparisonModalOpen, setComparisonModalOpen] = useState(false);
  const [activeComparisonCaseId, setActiveComparisonCaseId] = useState<string>("");
  const [activeComparisonCompositeId, setActiveComparisonCompositeId] = useState<
    string | undefined
  >(undefined);

  // Phase 7.0 Evidence Explainability Modal State
  const [traceModalOpen, setTraceModalOpen] = useState(false);
  const [traceData, setTraceData] = useState<
    CompositeTraceResponse | WitnessDescriptionTraceResponse | null
  >(null);
  const [traceInitialCategory, setTraceInitialCategory] = useState<string | undefined>(
    undefined
  );
  const [loadingTrace, setLoadingTrace] = useState(false);

  // Lifecycle Action Modal State
  const [lifecycleModal, setLifecycleModal] = useState<LifecycleModalState>({
    type: null,
    caseItem: null,
    isOpen: false,
  });
  const [modalForm, setModalForm] = useState({
    summary: "",
    reason: "",
  });
  const [submittingAction, setSubmittingAction] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  useEffect(() => {
    loadCases(activeView);
  }, [activeView]);

  async function loadCases(viewToLoad: CaseViewFilter = activeView) {
    try {
      setLoading(true);
      const response = await getAllCases({ view: viewToLoad });
      setCases(response.data?.cases || response.cases || []);
    } catch (error) {
      console.error("Failed to load cases", error);
    } finally {
      setLoading(false);
    }
  }

  async function handleCreateCase() {
    if (!newTitle.trim()) return;
    try {
      const caseNumber = `CASE-${Date.now().toString().slice(-6)}`;
      const res = await createCase({
        caseNumber,
        title: newTitle.trim(),
        description: "Forensic investigation case record",
      });
      const created = res.data?.case || res.case || res.data;
      if (created) {
        setCases((prev) => [created, ...prev]);
        setNewTitle("");
        setIsCreating(false);
      }
    } catch (err) {
      console.error("Failed to create case", err);
    }
  }

  function openLifecycleModal(
    type: "resolve" | "reopen" | "archive" | "restore" | "delete",
    caseItem: any
  ) {
    setLifecycleModal({
      type,
      caseItem,
      isOpen: true,
    });
    setModalForm({ summary: "", reason: "" });
    setActionError(null);
  }

  function closeLifecycleModal() {
    setLifecycleModal({ type: null, caseItem: null, isOpen: false });
    setModalForm({ summary: "", reason: "" });
    setActionError(null);
  }

  async function executeLifecycleAction() {
    if (!lifecycleModal.type || !lifecycleModal.caseItem) return;
    const caseId = lifecycleModal.caseItem._id;
    try {
      setSubmittingAction(true);
      setActionError(null);

      if (lifecycleModal.type === "resolve") {
        if (!modalForm.summary.trim()) {
          setActionError("Resolution summary is required to close/resolve a case.");
          setSubmittingAction(false);
          return;
        }
        await resolveCase(caseId, {
          resolutionSummary: modalForm.summary.trim(),
          resolutionReason: modalForm.reason.trim() || undefined,
        });
      } else if (lifecycleModal.type === "reopen") {
        if (!modalForm.reason.trim()) {
          setActionError("Reopen reason is required.");
          setSubmittingAction(false);
          return;
        }
        await reopenCase(caseId, {
          reopenReason: modalForm.reason.trim(),
        });
      } else if (lifecycleModal.type === "archive") {
        await archiveCase(caseId, {
          archiveReason: modalForm.reason.trim() || undefined,
        });
      } else if (lifecycleModal.type === "restore") {
        await restoreCase(caseId, {
          restoreReason: modalForm.reason.trim() || undefined,
        });
      } else if (lifecycleModal.type === "delete") {
        await deleteCase(caseId, {
          deletionReason: modalForm.reason.trim() || undefined,
        });
      }

      closeLifecycleModal();
      await loadCases(activeView);
    } catch (err: any) {
      console.error("Failed to execute lifecycle action:", err);
      setActionError(
        err?.response?.data?.message || err?.message || "Failed to execute lifecycle action"
      );
    } finally {
      setSubmittingAction(false);
    }
  }

  async function toggleExpandCase(caseId: string) {
    if (expandedCaseId === caseId) {
      setExpandedCaseId(null);
      return;
    }
    setExpandedCaseId(caseId);
    if (!caseCompositesMap[caseId]) {
      setLoadingCaseComposites(caseId);
      try {
        const res = await getCaseComposites(caseId);
        const comps = res.data?.composites || res.composites || res.data || [];
        setCaseCompositesMap((prev) => ({
          ...prev,
          [caseId]: Array.isArray(comps) ? comps : [],
        }));
      } catch (err) {
        console.error("Failed to load composites for case:", err);
      } finally {
        setLoadingCaseComposites(null);
      }
    }
  }

  async function handleOpenCompositeTrace(
    caseId: string,
    compositeId: string,
    category?: string
  ) {
    try {
      setLoadingTrace(true);
      setTraceInitialCategory(category);
      setTraceModalOpen(true);
      const res = await getCompositeTrace(caseId, compositeId);
      if (res.success && res.data) {
        setTraceData(res.data);
      }
    } catch (err) {
      console.error("Failed to load composite trace:", err);
    } finally {
      setLoadingTrace(false);
    }
  }

  const filtered = cases.filter(
    (c) =>
      c.title?.toLowerCase().includes(query.toLowerCase()) ||
      c.caseNumber?.toLowerCase().includes(query.toLowerCase())
  );

  const getStatusTone = (status: string): BadgeStatus => {
    switch (status) {
      case "open":
        return "active";
      case "in_progress":
        return "info";
      case "under_review":
        return "pending";
      case "resolved":
        return "verified";
      case "archived":
      case "closed":
        return "neutral";
      default:
        return "neutral";
    }
  };

  const VIEW_TABS: { key: CaseViewFilter; label: string }[] = [
    { key: "active", label: "Active Cases" },
    { key: "assigned", label: "Assigned to Me" },
    { key: "under_review", label: "Under Review" },
    { key: "resolved", label: "Resolved Cases" },
    { key: "archived", label: "Archived" },
    { key: "all", label: "All Cases" },
  ];

  return (
    <PageTransition>
      <div className="space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-lg font-semibold text-ink-primary">
              Forensic Investigation Cases
            </h1>
            <p className="label-technical text-ink-disabled mt-1">
              {cases.length} cases on file • Inspect composite sessions, compare variations, review XAI lineage & collaborate on case timelines
            </p>
          </div>

          {!isCreating ? (
            <Button size="sm" onClick={() => setIsCreating(true)}>
              <Plus className="h-3.5 w-3.5" />
              New case
            </Button>
          ) : (
            <div className="flex items-center gap-2">
              <Input
                placeholder="Enter case title..."
                value={newTitle}
                onChange={(e) => setNewTitle(e.target.value)}
                className="h-8 w-56 text-xs"
              />
              <Button size="sm" onClick={handleCreateCase}>
                Create
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setIsCreating(false)}
              >
                Cancel
              </Button>
            </div>
          )}
        </div>

        {/* Case View Filter Tabs */}
        <div className="flex items-center gap-2 border-b border-line pb-2 overflow-x-auto">
          {VIEW_TABS.map((tab) => {
            const isSelected = activeView === tab.key;
            return (
              <button
                key={tab.key}
                type="button"
                onClick={() => setActiveView(tab.key)}
                className={`px-3 py-1.5 text-xs font-medium rounded-md whitespace-nowrap transition-all cursor-pointer ${
                  isSelected
                    ? "bg-[#18263A] text-white border border-[#27B3A3]/50 shadow-xs"
                    : "text-ink-tertiary hover:text-ink-primary hover:bg-bg-inset border border-transparent"
                }`}
              >
                {tab.label}
              </button>
            );
          })}
        </div>

        <div className="relative max-w-sm">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-disabled" />
          <Input
            id="case-search"
            placeholder="Search cases..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="pl-9"
          />
        </div>

        {loading ? (
          <p className="text-sm text-ink-disabled">Loading cases...</p>
        ) : filtered.length === 0 ? (
          <EmptyState
            title="No matching cases"
            description="Adjust your search, change the view filter, or create a new case."
          />
        ) : (
          <div className="space-y-3">
            {filtered.map((c, i) => {
              const isExpanded = expandedCaseId === c._id;
              const comps = caseCompositesMap[c._id] || [];
              const isLocked = c.status === "resolved" || c.status === "archived";
              const creatorName =
                c.createdBy?.name || c.createdBy?.investigatorId || "Lead Investigator";

              return (
                <motion.div
                  key={c._id}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.04 }}
                >
                  <Card className={`overflow-hidden border transition-colors ${
                    isLocked ? "border-line/60 bg-bg-surface/50" : "border-line hover:border-gold/30"
                  }`}>
                    <div
                      className="flex flex-wrap items-center justify-between gap-4 px-5 py-4 cursor-pointer"
                      onClick={() => toggleExpandCase(c._id)}
                    >
                      <div className="flex items-center gap-4">
                        <EvidenceTag caseId={c.caseNumber} />
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium text-ink-primary block">
                              {c.title}
                            </span>
                            {isLocked && (
                              <span className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono bg-bg-inset text-ink-disabled border border-line">
                                <Lock className="w-2.5 h-2.5" />
                                READ-ONLY
                              </span>
                            )}
                          </div>
                          <div className="flex items-center gap-2 mt-0.5 text-[11px] text-ink-tertiary font-mono">
                            <span>Created {new Date(c.createdAt).toLocaleDateString()}</span>
                            <span>•</span>
                            <span className="flex items-center gap-1">
                              <User className="w-3 h-3 text-ink-disabled" />
                              Owner: <strong className="text-ink-secondary">{creatorName}</strong>
                            </span>
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center gap-2.5 flex-wrap">
                        <StatusBadge tone={getStatusTone(c.status)}>
                          {c.status?.replace("_", " ")?.toUpperCase()}
                        </StatusBadge>

                        {!isLocked ? (
                          <Button
                            size="sm"
                            variant="secondary"
                            onClick={(e) => {
                              e.stopPropagation();
                              navigate(`/generate?caseId=${c._id}`);
                            }}
                            className="h-7 text-xs gap-1 font-mono"
                          >
                            <ExternalLink className="w-3 h-3 text-gold" />
                            Studio
                          </Button>
                        ) : null}

                        {/* Lifecycle Contextual Actions */}
                        {c.status !== "resolved" && c.status !== "archived" && (
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              openLifecycleModal("resolve", c);
                            }}
                            className="px-2 py-1 rounded text-xs font-medium text-emerald-700 hover:text-emerald-800 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/30 transition-colors flex items-center gap-1 cursor-pointer"
                            title="Resolve Case"
                          >
                            <FileCheck2 className="w-3 h-3" />
                            Resolve
                          </button>
                        )}

                        {c.status === "resolved" && (
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              openLifecycleModal("reopen", c);
                            }}
                            className="px-2 py-1 rounded text-xs font-medium text-amber-700 hover:text-amber-800 bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/30 transition-colors flex items-center gap-1 cursor-pointer"
                            title="Reopen Case"
                          >
                            <RotateCcw className="w-3 h-3" />
                            Reopen
                          </button>
                        )}

                        {c.status !== "archived" && (
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              openLifecycleModal("archive", c);
                            }}
                            className="px-2 py-1 rounded text-xs font-medium text-[#7893B3] hover:text-[#B8C7DA] bg-[#101B2D]/40 hover:bg-[#101B2D]/80 border border-[#2A3B54] transition-colors flex items-center gap-1 cursor-pointer"
                            title="Archive Case"
                          >
                            <Archive className="w-3 h-3" />
                            Archive
                          </button>
                        )}

                        {c.status === "archived" && (
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              openLifecycleModal("restore", c);
                            }}
                            className="px-2 py-1 rounded text-xs font-medium text-[#27B3A3] hover:text-[#27B3A3]/80 bg-[#27B3A3]/10 hover:bg-[#27B3A3]/20 border border-[#27B3A3]/30 transition-colors flex items-center gap-1 cursor-pointer"
                            title="Restore Case"
                          >
                            <RotateCcw className="w-3 h-3" />
                            Restore
                          </button>
                        )}

                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            openLifecycleModal("delete", c);
                          }}
                          className="p-1 rounded text-ink-disabled hover:text-rose-500 hover:bg-rose-500/10 transition-colors cursor-pointer"
                          title="Soft Delete Case (Preserves Forensic Data)"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>

                        <button
                          type="button"
                          className="p-1 text-ink-tertiary hover:text-ink-primary ml-1"
                          aria-label="Toggle case composites"
                        >
                          {isExpanded ? (
                            <ChevronUp className="w-4 h-4" />
                          ) : (
                            <ChevronDown className="w-4 h-4" />
                          )}
                        </button>
                      </div>
                    </div>

                    {/* Expanded Composites & Multi-Variation Comparison Panel */}
                    {isExpanded && (
                      <div className="px-5 py-4 border-t border-line bg-bg-inset/70 space-y-4">
                        {loadingCaseComposites === c._id ? (
                          <div className="text-xs font-mono text-ink-tertiary py-4">
                            Loading composite versions & variations for {c.caseNumber}...
                          </div>
                        ) : comps.length === 0 ? (
                          <div className="flex items-center justify-between py-3 text-xs text-ink-tertiary">
                            <div className="flex items-center gap-2">
                              <ScanFace className="w-4 h-4 text-ink-disabled" />
                              <span>
                                No facial composite sessions generated for this case yet.
                              </span>
                            </div>
                            <Button
                              size="sm"
                              variant="primary"
                              onClick={() => navigate(`/generate?caseId=${c._id}`)}
                              className="h-7 text-xs"
                            >
                              Generate Composites in Studio
                            </Button>
                          </div>
                        ) : (
                          <>
                            <div className="flex flex-wrap items-center justify-between gap-2">
                              <div className="text-xs font-mono text-ink-secondary flex items-center gap-2">
                                <Layers className="w-3.5 h-3.5 text-gold" />
                                <span>
                                  {comps.length} Composite Version(s) on File
                                </span>
                              </div>

                              <div className="flex items-center gap-2">
                                <Button
                                  size="sm"
                                  variant="secondary"
                                  onClick={() => {
                                    setActiveComparisonCaseId(c._id);
                                    setActiveComparisonCompositeId(comps[0]._id);
                                    setComparisonModalOpen(true);
                                  }}
                                  className="h-7 text-xs gap-1.5 font-mono border-gold/40 text-gold"
                                >
                                  <Columns className="w-3.5 h-3.5 text-gold" />
                                  Compare Variations & Attribute Differences
                                </Button>
                              </div>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                              {comps.map((comp: any) => (
                                <div
                                  key={comp._id}
                                  className="p-3 rounded-lg bg-bg-deepest border border-line space-y-2.5 text-xs"
                                >
                                  <div className="flex items-center justify-between">
                                    <span className="font-mono font-bold text-gold">
                                      Composite v{comp.version}
                                    </span>
                                    <span
                                      className={`px-1.5 py-0.5 rounded text-[10px] font-mono uppercase ${
                                        comp.status === "approved"
                                          ? "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30"
                                          : "bg-bg-inset text-ink-tertiary border border-line"
                                      }`}
                                    >
                                      {comp.status}
                                    </span>
                                  </div>

                                  <div className="flex items-center gap-2 overflow-x-auto py-1">
                                    {(comp.variations || []).map((v: any) => {
                                      const isSelected =
                                        comp.selectedVariationIndex ===
                                          v.variationIndex &&
                                        comp.status === "approved";
                                      return (
                                        <div
                                          key={v.variationIndex}
                                          className={`relative p-1 rounded border shrink-0 ${
                                            isSelected
                                              ? "border-emerald-500 bg-emerald-500/10"
                                              : "border-line bg-black"
                                          }`}
                                        >
                                          <img
                                            src={v.dataUrl || v.imageUrl}
                                            alt={`v${comp.version} var${v.variationIndex}`}
                                            className="w-14 h-14 object-contain"
                                            style={{ imageRendering: (v.width || 128) >= 128 ? "auto" : "pixelated" }}
                                          />
                                          <div className="text-[9px] font-mono text-center text-ink-tertiary mt-0.5 flex items-center justify-center gap-0.5">
                                            <span>#{v.variationIndex}</span>
                                            {isSelected && (
                                              <CheckCircle2 className="w-2.5 h-2.5 text-emerald-400" />
                                            )}
                                          </div>
                                        </div>
                                      );
                                    })}
                                  </div>

                                  <div className="flex items-center justify-between pt-1 border-t border-line/60">
                                    <button
                                      type="button"
                                      onClick={() => {
                                        setActiveComparisonCaseId(c._id);
                                        setActiveComparisonCompositeId(comp._id);
                                        setComparisonModalOpen(true);
                                      }}
                                      className="text-[11px] font-mono text-gold hover:underline flex items-center gap-1"
                                    >
                                      <Columns className="w-3 h-3" /> Compare Batch
                                    </button>
                                    <button
                                      type="button"
                                      onClick={() =>
                                        handleOpenCompositeTrace(c._id, comp._id)
                                      }
                                      className="text-[11px] font-mono text-ink-secondary hover:text-gold flex items-center gap-1"
                                    >
                                      <ShieldCheck className="w-3 h-3 text-gold" /> XAI Lineage
                                    </button>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </>
                        )}

                        {/* Phase 7.7: Investigation Timeline, Workflow Progress & Investigator Collaboration */}
                        <CaseCollaborationWorkspace
                          caseId={c._id}
                          caseNumber={c.caseNumber}
                          compositesList={comps}
                          caseStatus={c.status}
                        />
                      </div>
                    )}
                  </Card>
                </motion.div>
              );
            })}
          </div>
        )}

        {/* Phase 7.1 Multi-Variation Comparison Modal */}
        <MultiVariationComparisonModal
          isOpen={comparisonModalOpen}
          onClose={() => setComparisonModalOpen(false)}
          caseId={activeComparisonCaseId}
          compositesList={caseCompositesMap[activeComparisonCaseId] || []}
          initialCompositeId={activeComparisonCompositeId}
          onVariationSelected={(updatedComp) => {
            if (!updatedComp || !activeComparisonCaseId) return;
            setCaseCompositesMap((prev) => ({
              ...prev,
              [activeComparisonCaseId]: (
                prev[activeComparisonCaseId] || []
              ).map((comp) =>
                comp._id === updatedComp._id ? { ...comp, ...updatedComp } : comp
              ),
            }));
          }}
          onOpenExplainability={(compId, category) => {
            handleOpenCompositeTrace(activeComparisonCaseId, compId, category);
          }}
        />

        {/* Phase 7.0 Evidence Traceability Modal */}
        <EvidenceExplainabilityModal
          isOpen={traceModalOpen}
          onClose={() => setTraceModalOpen(false)}
          traceData={traceData}
          type="composite"
          initialCategory={traceInitialCategory}
          loading={loadingTrace}
        />

        {/* Case Lifecycle Action Modal */}
        <AnimatePresence>
          {lifecycleModal.isOpen && (
            <div className="fixed inset-0 bg-[#101B2D]/80 backdrop-blur-xs z-50 flex items-center justify-center p-4">
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="bg-[#18263A] border border-[#2A3B54] rounded-lg max-w-lg w-full p-6 text-white shadow-2xl space-y-4"
              >
                <div className="flex items-center justify-between border-b border-[#2A3B54] pb-3">
                  <div className="flex items-center gap-2.5">
                    {lifecycleModal.type === "resolve" && (
                      <div className="w-8 h-8 rounded bg-emerald-500/15 border border-emerald-500/40 flex items-center justify-center text-emerald-400">
                        <FileCheck2 className="w-4 h-4" />
                      </div>
                    )}
                    {lifecycleModal.type === "reopen" && (
                      <div className="w-8 h-8 rounded bg-amber-500/15 border border-amber-500/40 flex items-center justify-center text-amber-400">
                        <RotateCcw className="w-4 h-4" />
                      </div>
                    )}
                    {lifecycleModal.type === "archive" && (
                      <div className="w-8 h-8 rounded bg-[#7893B3]/15 border border-[#7893B3]/40 flex items-center justify-center text-[#B8C7DA]">
                        <Archive className="w-4 h-4" />
                      </div>
                    )}
                    {lifecycleModal.type === "restore" && (
                      <div className="w-8 h-8 rounded bg-[#27B3A3]/15 border border-[#27B3A3]/40 flex items-center justify-center text-[#27B3A3]">
                        <RotateCcw className="w-4 h-4" />
                      </div>
                    )}
                    {lifecycleModal.type === "delete" && (
                      <div className="w-8 h-8 rounded bg-rose-500/15 border border-rose-500/40 flex items-center justify-center text-rose-400">
                        <Trash2 className="w-4 h-4" />
                      </div>
                    )}
                    <div>
                      <h3 className="text-sm font-semibold capitalize">
                        {lifecycleModal.type === "resolve" && "Mark Case as Resolved"}
                        {lifecycleModal.type === "reopen" && "Reopen Case Investigation"}
                        {lifecycleModal.type === "archive" && "Archive Case Record"}
                        {lifecycleModal.type === "restore" && "Restore Case to Active"}
                        {lifecycleModal.type === "delete" && "Soft Delete Case Record"}
                      </h3>
                      <p className="font-mono text-[11px] text-[#7893B3]">
                        {lifecycleModal.caseItem?.caseNumber} — {lifecycleModal.caseItem?.title}
                      </p>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={closeLifecycleModal}
                    className="p-1 text-[#7893B3] hover:text-white rounded"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>

                {/* Explanatory Policy Callout */}
                <div className={`p-3 rounded-md text-xs border ${
                  lifecycleModal.type === "delete"
                    ? "bg-rose-950/20 border-rose-500/30 text-rose-200"
                    : lifecycleModal.type === "resolve"
                    ? "bg-emerald-950/20 border-emerald-500/30 text-emerald-200"
                    : "bg-[#101B2D]/60 border-[#2A3B54] text-[#B8C7DA]"
                }`}>
                  {lifecycleModal.type === "resolve" && (
                    <p>
                      Resolving marks the case as concluded. Active generation is locked to preserve evidence integrity. The case will remain fully viewable and auditable.
                    </p>
                  )}
                  {lifecycleModal.type === "reopen" && (
                    <p>
                      Reopening restores this case to active status, permitting new witness evidence inputs, composite generation, and investigator assignments.
                    </p>
                  )}
                  {lifecycleModal.type === "archive" && (
                    <p>
                      Archiving moves the case to cold storage. It is excluded from the active dashboard and workspace views, but remains preserved and restorable.
                    </p>
                  )}
                  {lifecycleModal.type === "restore" && (
                    <p>
                      Restoring moves this archived case back to active or resolved status for inspection.
                    </p>
                  )}
                  {lifecycleModal.type === "delete" && (
                    <div className="flex items-start gap-2">
                      <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                      <div>
                        <strong>Forensic Chain-of-Custody Safeguard:</strong> This operation performs a safe soft-delete. In accordance with forensic integrity rules, witness descriptions, extracted evidence, and composite variations are <strong>never permanently purged</strong> from the database.
                      </div>
                    </div>
                  )}
                </div>

                {/* Form Fields */}
                <div className="space-y-3">
                  {lifecycleModal.type === "resolve" && (
                    <div>
                      <label className="block text-xs font-mono text-[#B8C7DA] mb-1">
                        Resolution Summary <span className="text-rose-400">*</span>
                      </label>
                      <textarea
                        rows={2}
                        placeholder="e.g., Suspect identified through composite match; warrant issued."
                        value={modalForm.summary}
                        onChange={(e) =>
                          setModalForm((prev) => ({ ...prev, summary: e.target.value }))
                        }
                        className="w-full bg-[#101B2D] border border-[#2A3B54] rounded p-2 text-xs text-white focus:outline-hidden focus:border-[#27B3A3]"
                      />
                    </div>
                  )}

                  <div>
                    <label className="block text-xs font-mono text-[#B8C7DA] mb-1">
                      {lifecycleModal.type === "reopen" ? (
                        <>Reopen Reason <span className="text-rose-400">*</span></>
                      ) : (
                        "Reason / Notes (Optional)"
                      )}
                    </label>
                    <textarea
                      rows={2}
                      placeholder={
                        lifecycleModal.type === "reopen"
                          ? "e.g., New witness testimony received."
                          : "Optional explanatory notes for security audit log..."
                      }
                      value={modalForm.reason}
                      onChange={(e) =>
                        setModalForm((prev) => ({ ...prev, reason: e.target.value }))
                      }
                      className="w-full bg-[#101B2D] border border-[#2A3B54] rounded p-2 text-xs text-white focus:outline-hidden focus:border-[#27B3A3]"
                    />
                  </div>
                </div>

                {actionError && (
                  <div className="p-2.5 rounded bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center gap-2">
                    <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                    <span>{actionError}</span>
                  </div>
                )}

                {/* Action Footer */}
                <div className="flex items-center justify-end gap-2 pt-2 border-t border-[#2A3B54]">
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={closeLifecycleModal}
                    disabled={submittingAction}
                    className="text-xs text-[#7893B3] hover:text-white"
                  >
                    Cancel
                  </Button>
                  <Button
                    size="sm"
                    onClick={executeLifecycleAction}
                    disabled={submittingAction}
                    className={`text-xs font-medium ${
                      lifecycleModal.type === "delete"
                        ? "bg-rose-600 hover:bg-rose-700 text-white"
                        : "bg-[#27B3A3] hover:bg-[#27B3A3]/90 text-white"
                    }`}
                  >
                    {submittingAction
                      ? "Processing..."
                      : lifecycleModal.type === "resolve"
                      ? "Confirm Resolution"
                      : lifecycleModal.type === "reopen"
                      ? "Confirm Reopening"
                      : lifecycleModal.type === "archive"
                      ? "Confirm Archival"
                      : lifecycleModal.type === "restore"
                      ? "Confirm Restore"
                      : "Soft Delete Case"}
                  </Button>
                </div>
              </motion.div>
            </div>
          )}
        </AnimatePresence>
      </div>
    </PageTransition>
  );
}
