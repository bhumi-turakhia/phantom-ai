import { useEffect, useState } from "react";
import {
  Activity,
  Users,
  MessageSquare,
  CheckCircle2,
  Clock,
  AlertTriangle,
  CircleDashed,
  UserPlus,
  UserMinus,
  Send,
  Edit3,
  Trash2,
  History,
  ShieldCheck,
  Filter,
  ArrowUpDown,
  RefreshCw,
  Link as LinkIcon,
  FileText,
} from "lucide-react";
import { Button } from "./ui/Button";
import { Input } from "./ui/Input";
import {
  getCaseWorkspace,
  getCaseTimeline,
  getEligibleInvestigators,
  addCaseParticipant,
  removeCaseParticipant,
  addCaseComment,
  updateCaseComment,
  deleteCaseComment,
} from "../services/caseService";
import type {
  CaseProgressData,
  CaseParticipantsData,
  CaseCommentsData,
  CaseTimelineData,
  EligibleInvestigator,
  CaseCommentNoteType,
  WorkflowStageState,
  RecordProvenanceType,
} from "../services/caseService";

interface CaseCollaborationWorkspaceProps {
  caseId: string;
  caseNumber: string;
  compositesList?: any[];
  caseStatus?: string;
}

const STAGE_STYLE: Record<
  WorkflowStageState,
  {
    badge: string;
    border: string;
    label: string;
  }
> = {
  completed: {
    badge: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
    border: "border-emerald-500/40 bg-emerald-950/15",
    label: "Completed",
  },
  pending: {
    badge: "bg-amber-500/15 text-amber-300 border-amber-500/30",
    border: "border-amber-500/40 bg-amber-950/15",
    label: "Pending",
  },
  blocked: {
    badge: "bg-rose-500/15 text-rose-300 border-rose-500/30",
    border: "border-rose-500/40 bg-rose-950/15",
    label: "Blocked",
  },
  not_started: {
    badge: "bg-bg-inset text-ink-disabled border-line",
    border: "border-line bg-bg-deepest/60",
    label: "Not Started",
  },
};

const PROVENANCE_BADGE: Record<
  RecordProvenanceType,
  { label: string; className: string }
> = {
  merged_audit_and_record: {
    label: "Audit + Persisted Record",
    className: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  },
  security_audit: {
    label: "SecurityAudit Event",
    className: "bg-sky-500/15 text-sky-300 border-sky-500/30",
  },
  persisted_record: {
    label: "Persisted DB Record",
    className: "bg-gold/15 text-gold border-gold/30",
  },
  historical_derived_state: {
    label: "Historical Derived State",
    className: "bg-purple-500/15 text-purple-300 border-purple-500/30",
  },
};

const NOTE_TYPE_LABELS: Record<
  CaseCommentNoteType,
  { label: string; badgeClass: string }
> = {
  general: {
    label: "General Note",
    badgeClass: "bg-bg-inset text-ink-secondary border-line",
  },
  lead_follow_up: {
    label: "Lead Follow-Up",
    badgeClass: "bg-amber-500/15 text-amber-300 border-amber-500/30",
  },
  supervisory_review: {
    label: "Supervisory Review",
    badgeClass: "bg-purple-500/15 text-purple-300 border-purple-500/30",
  },
  quality_assurance: {
    label: "Quality Assurance",
    badgeClass: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  },
};

export function CaseCollaborationWorkspace({
  caseId,
  caseNumber,
  compositesList = [],
  caseStatus,
}: CaseCollaborationWorkspaceProps) {
  const [activeTab, setActiveTab] = useState<
    "timeline" | "participants" | "comments"
  >("timeline");
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [progress, setProgress] = useState<CaseProgressData | null>(null);
  const [participants, setParticipants] = useState<CaseParticipantsData | null>(
    null
  );
  const [commentsData, setCommentsData] = useState<CaseCommentsData | null>(
    null
  );
  const [timeline, setTimeline] = useState<CaseTimelineData | null>(null);
  const [eligibleUsers, setEligibleUsers] = useState<EligibleInvestigator[]>(
    []
  );

  // Timeline filters
  const [timelineOrder, setTimelineOrder] = useState<"desc" | "asc">("desc");
  const [timelineCategory, setTimelineCategory] = useState<string>("all");
  const [expandedEventId, setExpandedEventId] = useState<string | null>(null);

  // Participant assignment form
  const [selectedUserId, setSelectedUserId] = useState<string>("");
  const [manualEmailOrId, setManualEmailOrId] = useState<string>("");
  const [assignmentNotes, setAssignmentNotes] = useState<string>("");
  const [assignPermission, setAssignPermission] = useState<"editor" | "read_only">("editor");
  const [assigning, setAssigning] = useState(false);
  const [participantFeedback, setParticipantFeedback] = useState<string | null>(
    null
  );

  // Comment creation & editing state
  const [newCommentContent, setNewCommentContent] = useState<string>("");
  const [newCommentType, setNewCommentType] =
    useState<CaseCommentNoteType>("general");
  const [linkedCompositeId, setLinkedCompositeId] = useState<string>("");
  const [postingComment, setPostingComment] = useState(false);

  const [editingCommentId, setEditingCommentId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState<string>("");
  const [editNoteType, setEditNoteType] =
    useState<CaseCommentNoteType>("general");
  const [editReason, setEditReason] = useState<string>("");
  const [savingEdit, setSavingEdit] = useState(false);
  const [expandedHistoryCommentId, setExpandedHistoryCommentId] = useState<
    string | null
  >(null);

  useEffect(() => {
    if (caseId) {
      loadWorkspaceData();
    }
  }, [caseId]);

  async function loadWorkspaceData() {
    try {
      setLoading(true);
      setErrorMsg(null);
      const [wsRes, eligRes] = await Promise.all([
        getCaseWorkspace(caseId),
        getEligibleInvestigators(caseId).catch(() => ({ data: [] })),
      ]);
      const data = wsRes.data || wsRes;
      if (data) {
        setProgress(data.progress || null);
        setParticipants(data.participants || null);
        setCommentsData(data.comments || null);
        setTimeline(data.timeline || null);
      }
      const eligList = eligRes.data || eligRes || [];
      if (Array.isArray(eligList)) {
        setEligibleUsers(eligList);
      }
    } catch (err: any) {
      setErrorMsg(
        err?.response?.data?.message ||
          "Failed to load case timeline and collaboration workspace."
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleFilterTimeline(
    nextOrder: "asc" | "desc",
    nextCategory: string
  ) {
    setTimelineOrder(nextOrder);
    setTimelineCategory(nextCategory);
    try {
      const res = await getCaseTimeline(caseId, {
        order: nextOrder,
        category: nextCategory,
        includeDeletedComments: true,
      });
      const tl = res.data || res;
      if (tl) {
        setTimeline(tl);
      }
    } catch (err) {
      console.error("Failed to filter timeline:", err);
    }
  }

  async function handleAddParticipant() {
    if (!selectedUserId && !manualEmailOrId.trim()) return;
    try {
      setAssigning(true);
      setParticipantFeedback(null);
      setErrorMsg(null);

      const payload: {
        userId?: string;
        email?: string;
        investigatorId?: string;
        notes?: string;
        permission?: "editor" | "read_only";
      } = {
        permission: assignPermission,
      };

      if (selectedUserId) {
        payload.userId = selectedUserId;
      } else if (manualEmailOrId.includes("@")) {
        payload.email = manualEmailOrId.trim();
      } else {
        payload.investigatorId = manualEmailOrId.trim();
      }

      if (assignmentNotes.trim()) {
        payload.notes = assignmentNotes.trim();
      }

      const res = await addCaseParticipant(caseId, payload);
      const updatedParticipants = res.data || res;
      setParticipants(updatedParticipants);
      setSelectedUserId("");
      setManualEmailOrId("");
      setAssignmentNotes("");
      setParticipantFeedback("Investigator assigned to case and recorded in audit trail.");

      // Refresh timeline
      handleFilterTimeline(timelineOrder, timelineCategory);
    } catch (err: any) {
      setErrorMsg(
        err?.response?.data?.message || "Failed to assign investigator to case."
      );
    } finally {
      setAssigning(false);
    }
  }

  async function handleRemoveParticipant(participantId: string, name: string) {
    try {
      setErrorMsg(null);
      setParticipantFeedback(null);
      const res = await removeCaseParticipant(
        caseId,
        participantId,
        `Removed ${name} from active case roster`
      );
      const updatedParticipants = res.data || res;
      setParticipants(updatedParticipants);
      setParticipantFeedback(
        `Removed ${name} from case assignment roster (logged in SecurityAudit).`
      );
      handleFilterTimeline(timelineOrder, timelineCategory);
    } catch (err: any) {
      setErrorMsg(
        err?.response?.data?.message || "Failed to remove participant from case."
      );
    }
  }

  async function handlePostComment() {
    if (!newCommentContent.trim()) return;
    try {
      setPostingComment(true);
      setErrorMsg(null);

      let linkedResource:
        | {
            resourceType: "composite";
            resourceId: string;
            label?: string;
          }
        | undefined;

      if (linkedCompositeId) {
        const compObj = compositesList.find((c) => c._id === linkedCompositeId);
        linkedResource = {
          resourceType: "composite",
          resourceId: linkedCompositeId,
          label: compObj
            ? `Composite v${compObj.version}`
            : "Linked Composite",
        };
      }

      await addCaseComment(caseId, {
        content: newCommentContent.trim(),
        noteType: newCommentType,
        linkedResource,
      });

      setNewCommentContent("");
      setLinkedCompositeId("");
      await loadWorkspaceData();
    } catch (err: any) {
      setErrorMsg(
        err?.response?.data?.message || "Failed to post investigator comment."
      );
    } finally {
      setPostingComment(false);
    }
  }

  async function handleSaveCommentEdit(commentId: string) {
    if (!editContent.trim()) return;
    try {
      setSavingEdit(true);
      setErrorMsg(null);
      await updateCaseComment(caseId, commentId, {
        content: editContent.trim(),
        noteType: editNoteType,
        editReason: editReason.trim() || undefined,
      });
      setEditingCommentId(null);
      setEditContent("");
      setEditReason("");
      await loadWorkspaceData();
    } catch (err: any) {
      setErrorMsg(
        err?.response?.data?.message || "Failed to update investigator comment."
      );
    } finally {
      setSavingEdit(false);
    }
  }

  async function handleDeleteComment(commentId: string) {
    try {
      setErrorMsg(null);
      await deleteCaseComment(
        caseId,
        commentId,
        "Soft-deleted by authorized investigator via collaboration panel"
      );
      await loadWorkspaceData();
    } catch (err: any) {
      setErrorMsg(
        err?.response?.data?.message || "Failed to remove investigator comment."
      );
    }
  }

  if (loading && !progress) {
    return (
      <div className="p-4 rounded-lg border border-line bg-bg-deepest text-xs font-mono text-ink-tertiary">
        Loading Phase 7.7 Investigation Timeline & Collaboration Workspace for{" "}
        {caseNumber}...
      </div>
    );
  }

  const alreadyAssignedIds = new Set<string>();
  if (participants?.creator?.id) {
    alreadyAssignedIds.add(participants.creator.id);
  }
  for (const inv of participants?.assignedInvestigators || []) {
    alreadyAssignedIds.add(inv.id);
  }
  const assignableUsers = eligibleUsers.filter(
    (u) => !alreadyAssignedIds.has(u.id)
  );

  return (
    <div className="mt-4 pt-4 border-t border-line space-y-4">
      {/* Section Header */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-gold" />
          <span className="text-xs font-mono font-semibold text-ink-primary uppercase tracking-wider">
            Investigation Timeline, Workflow Progress & Collaboration ({caseNumber})
          </span>
        </div>

        <Button
          size="sm"
          variant="ghost"
          onClick={loadWorkspaceData}
          className="h-7 text-[11px] font-mono gap-1"
        >
          <RefreshCw className="w-3 h-3" />
          Refresh Workspace
        </Button>
      </div>

      {errorMsg && (
        <div className="p-2.5 rounded border border-rose-500/40 bg-rose-950/30 text-xs font-mono text-rose-300 flex items-center justify-between">
          <span>{errorMsg}</span>
          <button
            type="button"
            onClick={() => setErrorMsg(null)}
            className="text-[10px] underline ml-3"
          >
            Dismiss
          </button>
        </div>
      )}

      {(caseStatus === "resolved" || caseStatus === "archived") && (
        <div className="flex items-center gap-2.5 p-3 rounded-lg border border-amber-500/40 bg-amber-950/20 text-xs font-mono text-amber-200">
          <ShieldCheck className="w-4 h-4 text-amber-400 shrink-0" />
          <div>
            <strong className="text-white">Case is {caseStatus.toUpperCase()} (Read-Only Mode):</strong> All witness statements, extracted features, and composite variations are locked to protect chain-of-custody integrity. Timeline comments and case notes remain auditable.
          </div>
        </div>
      )}

      {/* 1. 7-Stage Case Workflow Progress Bar */}
      {progress && (
        <div className="p-3.5 rounded-lg border border-line bg-bg-deepest space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono font-semibold text-gold">
                Case Workflow Progress: {progress.completedStagesCount}/
                {progress.totalStagesCount} Stages Completed (
                {progress.completionPercentage}%)
              </span>
              {progress.hasBlockingIssues && (
                <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-rose-500/15 text-rose-300 border border-rose-500/30">
                  Action Required: Stage Blocked
                </span>
              )}
            </div>
            <span className="text-[11px] font-mono text-ink-tertiary">
              Current Stage: #{progress.currentStage.stageNumber}{" "}
              {progress.currentStage.label} ({progress.currentStage.status})
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-7 gap-2">
            {progress.stages.map((st) => {
              const style = STAGE_STYLE[st.status];
              return (
                <div
                  key={st.stageKey}
                  className={`p-2.5 rounded border ${style.border} flex flex-col justify-between space-y-1.5`}
                >
                  <div className="flex items-center justify-between gap-1">
                    <span className="text-[10px] font-mono text-ink-tertiary">
                      0{st.stageNumber}
                    </span>
                    <span
                      className={`px-1.5 py-0.5 rounded text-[9px] font-mono uppercase border ${style.badge} flex items-center gap-1`}
                    >
                      {st.status === "completed" && (
                        <CheckCircle2 className="w-2.5 h-2.5" />
                      )}
                      {st.status === "pending" && (
                        <Clock className="w-2.5 h-2.5" />
                      )}
                      {st.status === "blocked" && (
                        <AlertTriangle className="w-2.5 h-2.5" />
                      )}
                      {st.status === "not_started" && (
                        <CircleDashed className="w-2.5 h-2.5" />
                      )}
                      {style.label}
                    </span>
                  </div>

                  <div>
                    <div className="text-xs font-semibold text-ink-primary">
                      {st.label}
                    </div>
                    <p className="text-[10px] text-ink-secondary mt-0.5 leading-snug">
                      {st.summary}
                    </p>
                  </div>

                  {st.completedAt && (
                    <div className="text-[9px] font-mono text-ink-disabled pt-1 border-t border-line/40 truncate">
                      {new Date(st.completedAt).toLocaleString()}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* 2. Sub-Navigation Tabs */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line pb-2">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setActiveTab("timeline")}
            className={`px-3 py-1.5 rounded text-xs font-mono flex items-center gap-1.5 transition-colors ${
              activeTab === "timeline"
                ? "bg-gold/15 text-gold border border-gold/40"
                : "bg-bg-deepest text-ink-secondary border border-line hover:text-ink-primary"
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            Activity Timeline ({timeline?.totalEvents ?? 0})
          </button>

          <button
            type="button"
            onClick={() => setActiveTab("participants")}
            className={`px-3 py-1.5 rounded text-xs font-mono flex items-center gap-1.5 transition-colors ${
              activeTab === "participants"
                ? "bg-gold/15 text-gold border border-gold/40"
                : "bg-bg-deepest text-ink-secondary border border-line hover:text-ink-primary"
            }`}
          >
            <Users className="w-3.5 h-3.5" />
            Investigators & Assignments ({participants?.totalParticipants ?? 1})
          </button>

          <button
            type="button"
            onClick={() => setActiveTab("comments")}
            className={`px-3 py-1.5 rounded text-xs font-mono flex items-center gap-1.5 transition-colors ${
              activeTab === "comments"
                ? "bg-gold/15 text-gold border border-gold/40"
                : "bg-bg-deepest text-ink-secondary border border-line hover:text-ink-primary"
            }`}
          >
            <MessageSquare className="w-3.5 h-3.5" />
            Case Notes & Comments ({commentsData?.totalComments ?? 0})
          </button>
        </div>
      </div>

      {/* TAB 1: INVESTIGATION ACTIVITY TIMELINE */}
      {activeTab === "timeline" && timeline && (
        <div className="space-y-3">
          {/* Filter & Sort Controls */}
          <div className="flex flex-wrap items-center justify-between gap-2 p-2.5 rounded bg-bg-deepest border border-line">
            <div className="flex flex-wrap items-center gap-1.5">
              <Filter className="w-3.5 h-3.5 text-ink-tertiary mr-1" />
              {[
                { key: "all", label: "All Events" },
                { key: "case", label: "Case" },
                { key: "collaboration", label: "Assignments" },
                { key: "evidence", label: "Witness Evidence" },
                { key: "nlp", label: "NLP Extraction" },
                { key: "composite", label: "Composites" },
                { key: "comparison", label: "Comparison & Selection" },
                { key: "report", label: "Reports" },
                { key: "comment", label: "Comments" },
              ].map((cat) => (
                <button
                  key={cat.key}
                  type="button"
                  onClick={() =>
                    handleFilterTimeline(timelineOrder, cat.key)
                  }
                  className={`px-2 py-0.5 rounded text-[10px] font-mono border transition-colors ${
                    timelineCategory === cat.key
                      ? "bg-gold/15 text-gold border-gold/40"
                      : "bg-bg-inset text-ink-tertiary border-line hover:text-ink-primary"
                  }`}
                >
                  {cat.label}
                </button>
              ))}
            </div>

            <button
              type="button"
              onClick={() =>
                handleFilterTimeline(
                  timelineOrder === "desc" ? "asc" : "desc",
                  timelineCategory
                )
              }
              className="px-2.5 py-1 rounded text-[10px] font-mono bg-bg-inset border border-line text-ink-secondary hover:text-gold flex items-center gap-1"
            >
              <ArrowUpDown className="w-3 h-3" />
              {timelineOrder === "desc"
                ? "Sorted: Newest First"
                : "Sorted: Oldest First (Chronological)"}
            </button>
          </div>

          {/* Provenance Legend */}
          <div className="flex flex-wrap items-center gap-3 text-[10px] font-mono text-ink-tertiary px-1">
            <span>Record Provenance Breakdown:</span>
            <span className="text-emerald-300">
              • Merged Audit + Record:{" "}
              {timeline.provenanceSummary.merged_audit_and_record}
            </span>
            <span className="text-sky-300">
              • SecurityAudit Events:{" "}
              {timeline.provenanceSummary.security_audit}
            </span>
            <span className="text-gold">
              • Persisted DB Records:{" "}
              {timeline.provenanceSummary.persisted_record}
            </span>
            <span className="text-purple-300">
              • Historical Derived State:{" "}
              {timeline.provenanceSummary.historical_derived_state}
            </span>
          </div>

          {/* Timeline Event List */}
          {timeline.events.length === 0 ? (
            <div className="p-4 rounded border border-line bg-bg-deepest text-xs font-mono text-ink-tertiary text-center">
              No timeline events match the selected filter ({timelineCategory}).
            </div>
          ) : (
            <div className="space-y-2 max-h-[420px] overflow-y-auto pr-1">
              {timeline.events.map((ev) => {
                const prov =
                  PROVENANCE_BADGE[ev.recordProvenance] ||
                  PROVENANCE_BADGE.persisted_record;
                const isExpanded = expandedEventId === ev.eventId;

                return (
                  <div
                    key={ev.eventId}
                    className="p-3 rounded-lg border border-line bg-bg-deepest hover:border-gold/30 transition-colors space-y-1.5 text-xs"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="px-1.5 py-0.5 rounded bg-bg-inset border border-line font-mono text-[10px] text-gold font-semibold">
                          {ev.eventType}
                        </span>
                        <span
                          className={`px-1.5 py-0.5 rounded border font-mono text-[9px] ${prov.className}`}
                        >
                          {prov.label}
                        </span>
                        {ev.resourceRef && (
                          <span className="px-1.5 py-0.5 rounded bg-bg-inset border border-line font-mono text-[10px] text-ink-secondary flex items-center gap-1">
                            <FileText className="w-2.5 h-2.5 text-gold" />
                            {ev.resourceRef.label}
                          </span>
                        )}
                      </div>

                      <div className="flex items-center gap-3 text-[10px] font-mono text-ink-tertiary">
                        {ev.actor ? (
                          <span>
                            Actor:{" "}
                            <strong className="text-ink-primary">
                              {ev.actor.name}
                            </strong>
                            {ev.actor.role ? ` (${ev.actor.role})` : ""}
                          </span>
                        ) : (
                          <span>Actor: System / Historical</span>
                        )}
                        <span>{new Date(ev.timestamp).toLocaleString()}</span>
                      </div>
                    </div>

                    <p className="text-ink-primary leading-relaxed">
                      {ev.description}
                    </p>

                    <div className="flex flex-wrap items-center justify-between gap-2 pt-1 text-[10px] font-mono text-ink-disabled">
                      <span>Provenance: {ev.provenanceNote}</span>
                      <button
                        type="button"
                        onClick={() =>
                          setExpandedEventId(isExpanded ? null : ev.eventId)
                        }
                        className="text-gold hover:underline"
                      >
                        {isExpanded ? "Hide Metadata" : "Inspect Metadata"}
                      </button>
                    </div>

                    {isExpanded && (
                      <pre className="p-2 rounded bg-black/60 border border-line text-[10px] font-mono text-ink-secondary overflow-x-auto">
                        {JSON.stringify(ev.metadata, null, 2)}
                      </pre>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* TAB 2: INVESTIGATOR COLLABORATION & PARTICIPANTS */}
      {activeTab === "participants" && participants && (
        <div className="space-y-4">
          {participantFeedback && (
            <div className="p-2.5 rounded border border-emerald-500/40 bg-emerald-950/20 text-xs font-mono text-emerald-300">
              {participantFeedback}
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Active Case Roster */}
            <div className="p-3.5 rounded-lg border border-line bg-bg-deepest space-y-3">
              <div className="text-xs font-mono font-semibold text-gold uppercase flex items-center gap-1.5">
                <Users className="w-3.5 h-3.5" />
                Authorized Case Investigators ({participants.totalParticipants})
              </div>

              <div className="space-y-2">
                {participants.creator && (
                  <div className="p-2.5 rounded border border-gold/30 bg-gold/5 flex items-center justify-between text-xs">
                    <div>
                      <div className="font-semibold text-ink-primary flex items-center gap-2">
                        <span>{participants.creator.name}</span>
                        <span className="px-1.5 py-0.5 rounded text-[9px] font-mono uppercase bg-gold/20 text-gold border border-gold/40">
                          Case Creator / Owner
                        </span>
                      </div>
                      <div className="text-[11px] font-mono text-ink-tertiary mt-0.5">
                        {participants.creator.email} • Role:{" "}
                        {participants.creator.role || "investigator"}
                        {participants.creator.investigatorId
                          ? ` • ID: ${participants.creator.investigatorId}`
                          : ""}
                      </div>
                    </div>
                  </div>
                )}

                {participants.assignedInvestigators.filter(
                  (inv) => inv.id !== participants.creator?.id
                ).length === 0 ? (
                  <div className="p-3 rounded border border-line bg-bg-inset/50 text-xs font-mono text-ink-tertiary">
                    No additional investigators currently assigned to this case.
                  </div>
                ) : (
                  participants.assignedInvestigators
                    .filter((inv) => inv.id !== participants.creator?.id)
                    .map((inv) => (
                      <div
                        key={inv.id}
                        className="p-2.5 rounded border border-line bg-bg-inset flex items-center justify-between text-xs"
                      >
                        <div>
                          <div className="font-semibold text-ink-primary flex items-center gap-2">
                            <span>{inv.name}</span>
                            <span className="px-1.5 py-0.5 rounded text-[9px] font-mono uppercase bg-sky-500/15 text-sky-300 border border-sky-500/30">
                              Assigned Investigator
                            </span>
                            <span className={`px-1.5 py-0.5 rounded text-[9px] font-mono uppercase ${
                              (inv as any).permission === "read_only"
                                ? "bg-amber-500/15 text-amber-300 border border-amber-500/30"
                                : "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30"
                            }`}>
                              {(inv as any).permission === "read_only" ? "Read-Only" : "Editor"}
                            </span>
                          </div>
                          <div className="text-[11px] font-mono text-ink-tertiary mt-0.5">
                            {inv.email} • Role: {inv.role || "investigator"}
                            {inv.investigatorId
                              ? ` • ID: ${inv.investigatorId}`
                              : ""}
                          </div>
                        </div>

                        {participants.canManageParticipants && (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() =>
                              handleRemoveParticipant(inv.id, inv.name)
                            }
                            className="h-7 text-[11px] font-mono text-rose-300 hover:text-rose-200 gap-1"
                          >
                            <UserMinus className="w-3 h-3" />
                            Remove
                          </Button>
                        )}
                      </div>
                    ))
                )}
              </div>
            </div>

            {/* Assign Investigator Form */}
            <div className="p-3.5 rounded-lg border border-line bg-bg-deepest space-y-3">
              <div className="text-xs font-mono font-semibold text-gold uppercase flex items-center gap-1.5">
                <UserPlus className="w-3.5 h-3.5" />
                Assign Collaborating Investigator
              </div>

              {!participants.canManageParticipants ? (
                <div className="p-3 rounded border border-line bg-bg-inset text-xs font-mono text-ink-tertiary">
                  Only the Case Creator, Supervisors, or Administrators can
                  modify participant assignments on this case.
                </div>
              ) : (
                <div className="space-y-2.5 text-xs">
                  <div>
                    <label className="block text-[11px] font-mono text-ink-secondary mb-1">
                      Select Active Agency Investigator
                    </label>
                    <select
                      value={selectedUserId}
                      onChange={(e) => {
                        setSelectedUserId(e.target.value);
                        if (e.target.value) setManualEmailOrId("");
                      }}
                      className="w-full h-8 rounded border border-line bg-bg-inset px-2.5 text-xs text-ink-primary font-mono"
                    >
                      <option value="">
                        -- Choose from active investigators ({assignableUsers.length}{" "}
                        available) --
                      </option>
                      {assignableUsers.map((u) => (
                        <option key={u.id} value={u.id}>
                          {u.name} ({u.email}) — {u.role}
                          {u.investigatorId ? ` [${u.investigatorId}]` : ""}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-[11px] font-mono text-ink-secondary mb-1">
                      Or Assign by Email / Investigator ID
                    </label>
                    <Input
                      placeholder="e.g., colleague@agency.gov or INV-2026-002"
                      value={manualEmailOrId}
                      onChange={(e) => {
                        setManualEmailOrId(e.target.value);
                        if (e.target.value) setSelectedUserId("");
                      }}
                      className="h-8 text-xs font-mono"
                    />
                  </div>

                  <div>
                    <label className="block text-[11px] font-mono text-ink-secondary mb-1">
                      Case Access Permission Level
                    </label>
                    <select
                      value={assignPermission}
                      onChange={(e) => setAssignPermission(e.target.value as "editor" | "read_only")}
                      className="w-full h-8 rounded border border-line bg-bg-inset px-2.5 text-xs text-ink-primary font-mono"
                    >
                      <option value="editor">Editor — Can add evidence, generate composites & review</option>
                      <option value="read_only">Read-Only — Can inspect evidence & add timeline comments</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-[11px] font-mono text-ink-secondary mb-1">
                      Assignment Collaboration Note (Optional)
                    </label>
                    <Input
                      placeholder="e.g., Assigned for secondary witness interview & composite review"
                      value={assignmentNotes}
                      onChange={(e) => setAssignmentNotes(e.target.value)}
                      className="h-8 text-xs"
                    />
                  </div>

                  <Button
                    size="sm"
                    variant="primary"
                    disabled={
                      assigning ||
                      (!selectedUserId && !manualEmailOrId.trim())
                    }
                    onClick={handleAddParticipant}
                    className="h-8 text-xs font-mono gap-1.5"
                  >
                    <UserPlus className="w-3.5 h-3.5" />
                    {assigning
                      ? "Assigning Investigator..."
                      : "Assign to Case Roster"}
                  </Button>
                </div>
              )}
            </div>
          </div>

          {/* Participant Assignment Audit Log */}
          {participants.assignmentHistory.length > 0 && (
            <div className="p-3 rounded-lg border border-line bg-bg-deepest space-y-2">
              <div className="text-[11px] font-mono font-semibold text-ink-secondary uppercase flex items-center gap-1.5">
                <History className="w-3.5 h-3.5 text-gold" />
                Participant Assignment Audit History (SecurityAudit)
              </div>
              <div className="space-y-1.5 max-h-40 overflow-y-auto">
                {participants.assignmentHistory.map((item) => (
                  <div
                    key={item.auditId}
                    className="p-2 rounded border border-line bg-bg-inset flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono"
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className={`px-1.5 py-0.5 rounded text-[9px] ${
                          item.event === "CASE_PARTICIPANT_ADDED"
                            ? "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30"
                            : "bg-rose-500/15 text-rose-300 border border-rose-500/30"
                        }`}
                      >
                        {item.event}
                      </span>
                      <span className="text-ink-primary">
                        {item.targetParticipantName ||
                          item.targetParticipantEmail ||
                          item.targetParticipantId}
                      </span>
                      {item.notes && (
                        <span className="text-ink-tertiary">
                          ({item.notes})
                        </span>
                      )}
                    </div>
                    <div className="text-ink-disabled">
                      By {item.actor?.name || "Authorized User"} •{" "}
                      {new Date(item.timestamp).toLocaleString()}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 3: CASE COMMENTS & COLLABORATION NOTES */}
      {activeTab === "comments" && commentsData && (
        <div className="space-y-4">
          {/* Forensic Evidence Isolation Banner */}
          <div className="p-2.5 rounded border border-gold/30 bg-gold/5 flex items-center gap-2 text-[11px] font-mono text-ink-secondary">
            <ShieldCheck className="w-4 h-4 text-gold shrink-0" />
            <span>
              <strong>Forensic Evidence Isolation Guarantee:</strong> Case
              collaboration comments and notes are stored in an isolated
              collection (`CaseComment`) and never modify raw witness statements,
              BiLSTM-CRF attribute extractions, or 20-D GCIT conditioning
              vectors.
            </span>
          </div>

          {/* Post New Comment Form */}
          <div className="p-3.5 rounded-lg border border-line bg-bg-deepest space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-xs font-mono font-semibold text-gold uppercase flex items-center gap-1.5">
                <MessageSquare className="w-3.5 h-3.5" />
                Add Investigator Case Note / Comment
              </span>

              <div className="flex flex-wrap items-center gap-2">
                <select
                  value={newCommentType}
                  onChange={(e) =>
                    setNewCommentType(e.target.value as CaseCommentNoteType)
                  }
                  className="h-7 rounded border border-line bg-bg-inset px-2 text-[11px] font-mono text-ink-primary"
                >
                  <option value="general">Type: General Note</option>
                  <option value="lead_follow_up">Type: Lead Follow-Up</option>
                  <option value="supervisory_review">
                    Type: Supervisory Review
                  </option>
                  <option value="quality_assurance">
                    Type: Quality Assurance
                  </option>
                </select>

                {compositesList.length > 0 && (
                  <select
                    value={linkedCompositeId}
                    onChange={(e) => setLinkedCompositeId(e.target.value)}
                    className="h-7 rounded border border-line bg-bg-inset px-2 text-[11px] font-mono text-ink-primary"
                  >
                    <option value="">Link Resource: Case-Level (None)</option>
                    {compositesList.map((comp: any) => (
                      <option key={comp._id} value={comp._id}>
                        Link: Composite v{comp.version} ({comp.status})
                      </option>
                    ))}
                  </select>
                )}
              </div>
            </div>

            <textarea
              rows={3}
              placeholder="Enter investigative note, composite comparison rationale, or supervisory feedback..."
              value={newCommentContent}
              onChange={(e) => setNewCommentContent(e.target.value)}
              className="w-full rounded border border-line bg-bg-inset p-2.5 text-xs text-ink-primary focus:border-gold outline-none"
            />

            <div className="flex justify-end">
              <Button
                size="sm"
                variant="primary"
                disabled={postingComment || !newCommentContent.trim()}
                onClick={handlePostComment}
                className="h-8 text-xs font-mono gap-1.5"
              >
                <Send className="w-3.5 h-3.5" />
                {postingComment ? "Posting Note..." : "Post Case Note"}
              </Button>
            </div>
          </div>

          {/* Chronological Comments List */}
          {commentsData.comments.length === 0 ? (
            <div className="p-4 rounded border border-line bg-bg-deepest text-xs font-mono text-ink-tertiary text-center">
              No investigator comments or collaboration notes recorded for{" "}
              {caseNumber} yet.
            </div>
          ) : (
            <div className="space-y-2.5">
              {commentsData.comments.map((comment) => {
                const noteMeta =
                  NOTE_TYPE_LABELS[comment.noteType] ||
                  NOTE_TYPE_LABELS.general;
                const isEditing = editingCommentId === comment.id;
                const showHistory = expandedHistoryCommentId === comment.id;

                return (
                  <div
                    key={comment.id}
                    className={`p-3.5 rounded-lg border space-y-2 text-xs ${
                      comment.isDeleted
                        ? "border-rose-500/30 bg-rose-950/10 opacity-80"
                        : "border-line bg-bg-deepest"
                    }`}
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-semibold text-ink-primary">
                          {comment.author?.name || "Investigator"}
                        </span>
                        {comment.author?.role && (
                          <span className="text-[10px] font-mono text-ink-tertiary">
                            ({comment.author.role})
                          </span>
                        )}
                        <span
                          className={`px-1.5 py-0.5 rounded text-[9px] font-mono border ${noteMeta.badgeClass}`}
                        >
                          {noteMeta.label}
                        </span>
                        {comment.linkedResource && (
                          <span className="px-1.5 py-0.5 rounded text-[9px] font-mono bg-gold/15 text-gold border border-gold/30 flex items-center gap-1">
                            <LinkIcon className="w-2.5 h-2.5" />
                            {comment.linkedResource.label}
                          </span>
                        )}
                        {comment.isEdited && (
                          <span className="px-1.5 py-0.5 rounded text-[9px] font-mono bg-amber-500/15 text-amber-300 border border-amber-500/30">
                            Edited ({comment.editHistory.length}x)
                          </span>
                        )}
                        {comment.isDeleted && (
                          <span className="px-1.5 py-0.5 rounded text-[9px] font-mono bg-rose-500/20 text-rose-300 border border-rose-500/40">
                            Soft-Deleted (Audit Retained)
                          </span>
                        )}
                      </div>

                      <div className="flex items-center gap-2 text-[10px] font-mono text-ink-tertiary">
                        <span>
                          {new Date(comment.createdAt).toLocaleString()}
                        </span>

                        {comment.canModify && !isEditing && (
                          <div className="flex items-center gap-1.5 ml-2">
                            <button
                              type="button"
                              onClick={() => {
                                setEditingCommentId(comment.id);
                                setEditContent(comment.content);
                                setEditNoteType(comment.noteType);
                                setEditReason("");
                              }}
                              className="text-gold hover:underline flex items-center gap-0.5"
                            >
                              <Edit3 className="w-3 h-3" />
                              Edit
                            </button>
                            <button
                              type="button"
                              onClick={() => handleDeleteComment(comment.id)}
                              className="text-rose-400 hover:underline flex items-center gap-0.5 ml-1"
                            >
                              <Trash2 className="w-3 h-3" />
                              Delete
                            </button>
                          </div>
                        )}
                      </div>
                    </div>

                    {isEditing ? (
                      <div className="space-y-2 pt-1">
                        <textarea
                          rows={3}
                          value={editContent}
                          onChange={(e) => setEditContent(e.target.value)}
                          className="w-full rounded border border-line bg-bg-inset p-2 text-xs text-ink-primary"
                        />
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <Input
                            placeholder="Reason for edit (recorded in audit history)..."
                            value={editReason}
                            onChange={(e) => setEditReason(e.target.value)}
                            className="h-7 text-xs flex-1"
                          />
                          <div className="flex items-center gap-1.5">
                            <Button
                              size="sm"
                              variant="primary"
                              disabled={savingEdit}
                              onClick={() => handleSaveCommentEdit(comment.id)}
                              className="h-7 text-xs font-mono"
                            >
                              {savingEdit ? "Saving..." : "Save Edit"}
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => setEditingCommentId(null)}
                              className="h-7 text-xs font-mono"
                            >
                              Cancel
                            </Button>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <p className="text-ink-primary whitespace-pre-wrap leading-relaxed">
                        {comment.content}
                      </p>
                    )}

                    {/* Audit History Drawer Toggle for Edited or Deleted Comments */}
                    {(comment.editHistory.length > 0 || comment.isDeleted) && (
                      <div className="pt-1 border-t border-line/50">
                        <button
                          type="button"
                          onClick={() =>
                            setExpandedHistoryCommentId(
                              showHistory ? null : comment.id
                            )
                          }
                          className="text-[10px] font-mono text-gold hover:underline flex items-center gap-1"
                        >
                          <History className="w-3 h-3" />
                          {showHistory
                            ? "Hide Comment Modification & Deletion Audit History"
                            : `View Comment Audit History (${
                                comment.editHistory.length
                              } edit(s)${
                                comment.isDeleted ? ", soft-deleted" : ""
                              })`}
                        </button>

                        {showHistory && (
                          <div className="mt-2 p-2.5 rounded bg-bg-inset border border-line space-y-2 text-[11px] font-mono">
                            {comment.editHistory.map((h, idx) => (
                              <div
                                key={idx}
                                className="pb-1.5 border-b border-line/40 last:border-none"
                              >
                                <div className="text-ink-tertiary">
                                  Revision #{idx + 1} • Edited by{" "}
                                  {h.editedBy?.name || "Investigator"} at{" "}
                                  {new Date(h.editedAt).toLocaleString()}
                                  {h.editReason
                                    ? ` • Reason: ${h.editReason}`
                                    : ""}
                                </div>
                                <div className="text-ink-secondary mt-0.5">
                                  Prior Content: "{h.previousContent}"
                                </div>
                              </div>
                            ))}

                            {comment.isDeleted && (
                              <div className="text-rose-300">
                                Soft-Deleted by{" "}
                                {comment.deletedBy?.name || "Authorized User"}{" "}
                                at{" "}
                                {comment.deletedAt
                                  ? new Date(comment.deletedAt).toLocaleString()
                                  : "N/A"}
                                {comment.deleteReason
                                  ? ` • Reason: ${comment.deleteReason}`
                                  : ""}
                                {comment.originalContentPreservedInAudit && (
                                  <div className="text-ink-secondary mt-0.5">
                                    Preserved Original Content: "
                                    {comment.originalContentPreservedInAudit}"
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

