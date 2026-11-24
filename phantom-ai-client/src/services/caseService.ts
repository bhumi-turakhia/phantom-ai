import api from "./api";

export interface TimelineActor {
  id: string;
  name: string;
  email?: string;
  role?: string;
  investigatorId?: string;
}

export interface TimelineResourceRef {
  type:
    | "case"
    | "user"
    | "witness_description"
    | "composite"
    | "report"
    | "case_comment";
  id: string;
  label: string;
}

export type TimelineCategory =
  | "case"
  | "collaboration"
  | "evidence"
  | "nlp"
  | "composite"
  | "comparison"
  | "report"
  | "comment";

export type RecordProvenanceType =
  | "security_audit"
  | "persisted_record"
  | "merged_audit_and_record"
  | "historical_derived_state";

export interface TimelineEventItem {
  eventId: string;
  eventType: string;
  category: TimelineCategory;
  timestamp: string;
  actor: TimelineActor | null;
  caseId: string;
  caseNumber: string;
  resourceRef: TimelineResourceRef | null;
  description: string;
  metadata: Record<string, unknown>;
  recordProvenance: RecordProvenanceType;
  provenanceNote: string;
}

export type WorkflowStageState =
  | "completed"
  | "pending"
  | "blocked"
  | "not_started";

export interface WorkflowStageProgress {
  stageKey:
    | "case_creation"
    | "witness_evidence"
    | "nlp_analysis"
    | "evidence_approval"
    | "composite_generation"
    | "variation_selection"
    | "report_generation";
  stageNumber: number;
  label: string;
  status: WorkflowStageState;
  summary: string;
  completedAt: string | null;
  actor: TimelineActor | null;
  metrics: Record<string, unknown>;
}

export interface CaseProgressData {
  caseId: string;
  caseNumber: string;
  caseStatus: string;
  completedStagesCount: number;
  totalStagesCount: number;
  completionPercentage: number;
  hasBlockingIssues: boolean;
  currentStage: {
    stageKey: string;
    stageNumber: number;
    label: string;
    status: WorkflowStageState;
  };
  stages: WorkflowStageProgress[];
}

export interface CaseParticipantItem {
  id: string;
  name: string;
  email?: string;
  role?: string;
  investigatorId?: string;
  badgeNumber?: string;
  department?: string;
  participantRole: "case_creator" | "assigned_investigator";
}

export interface AssignmentAuditEntry {
  auditId: string;
  event: "CASE_PARTICIPANT_ADDED" | "CASE_PARTICIPANT_REMOVED";
  timestamp: string;
  actor: TimelineActor | null;
  targetParticipantId: string | null;
  targetParticipantName: string | null;
  targetParticipantEmail: string | null;
  targetInvestigatorId: string | null;
  notes: string | null;
}

export interface CaseParticipantsData {
  caseId: string;
  caseNumber: string;
  canManageParticipants: boolean;
  creator: CaseParticipantItem | null;
  assignedInvestigators: CaseParticipantItem[];
  totalParticipants: number;
  assignmentHistory: AssignmentAuditEntry[];
}

export interface EligibleInvestigator {
  id: string;
  name: string;
  email: string;
  role: string;
  investigatorId?: string;
  badgeNumber?: string;
  department?: string;
}

export type CaseCommentNoteType =
  | "general"
  | "lead_follow_up"
  | "supervisory_review"
  | "quality_assurance";

export interface CaseCommentEditEntry {
  previousContent: string;
  previousNoteType: CaseCommentNoteType;
  editedBy: TimelineActor | null;
  editedAt: string;
  editReason: string | null;
}

export interface CaseCommentItem {
  id: string;
  caseId: string;
  author: TimelineActor | null;
  content: string;
  originalContentPreservedInAudit?: string;
  noteType: CaseCommentNoteType;
  linkedResource: {
    resourceType: "witness_description" | "composite" | "report";
    resourceId: string;
    label: string;
  } | null;
  isEdited: boolean;
  editedAt: string | null;
  editHistory: CaseCommentEditEntry[];
  isDeleted: boolean;
  deletedAt: string | null;
  deletedBy: TimelineActor | null;
  deleteReason: string | null;
  canModify: boolean;
  evidenceIsolationNotice: string;
  createdAt: string;
  updatedAt: string;
}

export interface CaseCommentsData {
  caseId: string;
  totalComments: number;
  includeDeleted: boolean;
  comments: CaseCommentItem[];
}

export interface CaseTimelineData {
  caseId: string;
  caseNumber: string;
  order: "asc" | "desc";
  categoryFilter: string;
  totalEvents: number;
  provenanceSummary: {
    security_audit: number;
    persisted_record: number;
    merged_audit_and_record: number;
    historical_derived_state: number;
  };
  events: TimelineEventItem[];
}

export interface CaseWorkspaceData {
  caseId: string;
  caseNumber: string;
  progress: CaseProgressData;
  participants: CaseParticipantsData;
  comments: CaseCommentsData;
  timeline: CaseTimelineData;
}

export const createCase = async (data: {
  caseNumber: string;
  title: string;
  description: string;
}) => {
  const response = await api.post("/cases", data);
  return response.data;
};

export const getAllCases = async (params?: {
  view?: "active" | "my_cases" | "assigned" | "under_review" | "resolved" | "archived" | "all";
  status?: string;
  priority?: string;
  search?: string;
  page?: number;
  limit?: number;
}) => {
  const response = await api.get("/cases", { params });
  return response.data;
};

export const getCaseById = async (id: string) => {
  const response = await api.get(`/cases/${id}`);
  return response.data;
};

export const updateCase = async (
  id: string,
  data: Record<string, unknown>
) => {
  const response = await api.patch(`/cases/${id}`, data);
  return response.data;
};

export const resolveCase = async (
  id: string,
  data?: {
    summary?: string;
    reason?: string;
    resolutionSummary?: string;
    resolutionReason?: string;
  }
) => {
  const payload = {
    resolutionSummary: data?.resolutionSummary || data?.summary || "",
    resolutionReason: data?.resolutionReason || data?.reason || undefined,
  };
  const response = await api.post(`/cases/${id}/resolve`, payload);
  return response.data;
};

export const reopenCase = async (
  id: string,
  data: { reason?: string; reopenReason?: string }
) => {
  const payload = {
    reopenReason: data.reopenReason || data.reason || "",
  };
  const response = await api.post(`/cases/${id}/reopen`, payload);
  return response.data;
};

export const archiveCase = async (
  id: string,
  data?: { reason?: string; archiveReason?: string }
) => {
  const payload = {
    archiveReason: data?.archiveReason || data?.reason || undefined,
  };
  const response = await api.post(`/cases/${id}/archive`, payload);
  return response.data;
};

export const restoreCase = async (
  id: string,
  data?: { reason?: string; restoreReason?: string }
) => {
  const payload = {
    restoreReason: data?.restoreReason || data?.reason || undefined,
  };
  const response = await api.post(`/cases/${id}/restore`, payload);
  return response.data;
};

export const deleteCase = async (
  id: string,
  data?: { reason?: string; deletionReason?: string }
) => {
  const payload = {
    deletionReason: data?.deletionReason || data?.reason || undefined,
  };
  const response = await api.delete(`/cases/${id}`, {
    data: payload,
  });
  return response.data;
};

export const getCaseWorkspace = async (caseId: string) => {
  const response = await api.get(`/cases/${caseId}/workspace`);
  return response.data;
};

export const getCaseTimeline = async (
  caseId: string,
  params?: {
    order?: "asc" | "desc";
    category?: string;
    includeDeletedComments?: boolean;
  }
) => {
  const response = await api.get(`/cases/${caseId}/timeline`, {
    params: {
      order: params?.order,
      category: params?.category,
      includeDeletedComments:
        params?.includeDeletedComments !== undefined
          ? String(params.includeDeletedComments)
          : undefined,
    },
  });
  return response.data;
};

export const getCaseProgress = async (caseId: string) => {
  const response = await api.get(`/cases/${caseId}/progress`);
  return response.data;
};

export const getCaseParticipants = async (caseId: string) => {
  const response = await api.get(`/cases/${caseId}/participants`);
  return response.data;
};

export const getEligibleInvestigators = async (caseId: string) => {
  const response = await api.get(`/cases/${caseId}/eligible-investigators`);
  return response.data;
};

export const addCaseParticipant = async (
  caseId: string,
  data: {
    userId?: string;
    investigatorId?: string;
    email?: string;
    notes?: string;
  }
) => {
  const response = await api.post(`/cases/${caseId}/participants`, data);
  return response.data;
};

export const removeCaseParticipant = async (
  caseId: string,
  participantId: string,
  reason?: string
) => {
  const response = await api.delete(
    `/cases/${caseId}/participants/${participantId}`,
    {
      data: reason ? { reason } : undefined,
    }
  );
  return response.data;
};

export const getCaseComments = async (
  caseId: string,
  includeDeleted = true
) => {
  const response = await api.get(`/cases/${caseId}/comments`, {
    params: { includeDeleted: String(includeDeleted) },
  });
  return response.data;
};

export const addCaseComment = async (
  caseId: string,
  data: {
    content: string;
    noteType?: CaseCommentNoteType;
    linkedResource?: {
      resourceType: "witness_description" | "composite" | "report";
      resourceId: string;
      label?: string;
    };
  }
) => {
  const response = await api.post(`/cases/${caseId}/comments`, data);
  return response.data;
};

export const updateCaseComment = async (
  caseId: string,
  commentId: string,
  data: {
    content: string;
    noteType?: CaseCommentNoteType;
    editReason?: string;
  }
) => {
  const response = await api.patch(
    `/cases/${caseId}/comments/${commentId}`,
    data
  );
  return response.data;
};

export const deleteCaseComment = async (
  caseId: string,
  commentId: string,
  deleteReason?: string
) => {
  const response = await api.delete(
    `/cases/${caseId}/comments/${commentId}`,
    {
      data: deleteReason ? { deleteReason } : undefined,
    }
  );
  return response.data;
};
