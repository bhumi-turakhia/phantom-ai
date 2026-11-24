import { Types } from 'mongoose';
import { Case } from '../models/Case.model';
import { User } from '../models/User.model';
import { WitnessDescription } from '../models/WitnessDescription.model';
import { Composite } from '../models/Composite.model';
import { Report } from '../models/Report.model';
import { SecurityAudit } from '../models/SecurityAudit.model';
import {
  CaseComment,
  CaseCommentNoteType,
} from '../models/CaseComment.model';
import { ApiError } from '../utils/apiResponse';
import { assertCaseAccess } from './witnessDescription.service';
import type {
  AddParticipantInput,
  CreateCommentInput,
  UpdateCommentInput,
} from '../validations/caseCollaboration.validation';

export type TimelineCategory =
  | 'case'
  | 'collaboration'
  | 'evidence'
  | 'nlp'
  | 'composite'
  | 'comparison'
  | 'report'
  | 'comment';

export type RecordProvenanceType =
  | 'security_audit'
  | 'persisted_record'
  | 'merged_audit_and_record'
  | 'historical_derived_state';

export interface TimelineActor {
  id: string;
  name: string;
  email?: string;
  role?: string;
  investigatorId?: string;
}

export interface TimelineResourceRef {
  type:
    | 'case'
    | 'user'
    | 'witness_description'
    | 'composite'
    | 'report'
    | 'case_comment';
  id: string;
  label: string;
}

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

export type WorkflowStageState = 'completed' | 'pending' | 'blocked' | 'not_started';

export interface WorkflowStageProgress {
  stageKey:
    | 'case_creation'
    | 'witness_evidence'
    | 'nlp_analysis'
    | 'evidence_approval'
    | 'composite_generation'
    | 'variation_selection'
    | 'report_generation';
  stageNumber: number;
  label: string;
  status: WorkflowStageState;
  summary: string;
  completedAt: string | null;
  actor: TimelineActor | null;
  metrics: Record<string, unknown>;
}

const validateObjectId = (id: string, label = 'ID'): void => {
  if (!Types.ObjectId.isValid(id)) {
    throw new ApiError(`Invalid ${label}`, 400);
  }
};

const formatActor = (userObj: any): TimelineActor | null => {
  if (!userObj) return null;
  if (typeof userObj === 'string' || userObj instanceof Types.ObjectId) {
    const idStr = userObj.toString();
    return {
      id: idStr,
      name: `Investigator (${idStr.slice(-6)})`,
    };
  }
  const idStr = (userObj._id || userObj.id || '').toString();
  return {
    id: idStr,
    name: userObj.name || userObj.email || `Investigator (${idStr.slice(-6)})`,
    email: userObj.email,
    role: userObj.role,
    investigatorId: userObj.investigatorId,
  };
};

const extractIdString = (ref: any): string => {
  if (!ref) return '';
  if (typeof ref === 'string') return ref;
  if (ref instanceof Types.ObjectId) return ref.toString();
  if (ref._id) return ref._id.toString();
  return String(ref);
};

const assertCanManageParticipants = (
  caseDoc: any,
  userId: string,
  userRole?: string
): void => {
  if (userRole === 'admin' || userRole === 'supervisor') {
    return;
  }
  const creatorId = extractIdString(caseDoc.createdBy);
  if (creatorId === userId.toString()) {
    return;
  }
  throw new ApiError(
    'Forbidden: Only the case creator, supervisors, or administrators can modify case participant assignments.',
    403
  );
};

/**
 * Returns active agency investigators who can be assigned to the case.
 */
export const getEligibleInvestigators = async (
  caseId: string,
  userId: string,
  userRole?: string
) => {
  validateObjectId(caseId, 'Case ID');
  await assertCaseAccess(caseId, userId, userRole);

  const users = await User.find({ isActive: true })
    .select('name email role investigatorId badgeNumber department')
    .sort({ name: 1 })
    .lean();

  return users.map((u: any) => ({
    id: u._id.toString(),
    name: u.name,
    email: u.email,
    role: u.role,
    investigatorId: u.investigatorId,
    badgeNumber: u.badgeNumber,
    department: u.department,
  }));
};

/**
 * Returns case creator, assigned investigators, and participant assignment audit history.
 */
export const getCaseParticipants = async (
  caseId: string,
  userId: string,
  userRole?: string
) => {
  validateObjectId(caseId, 'Case ID');
  await assertCaseAccess(caseId, userId, userRole);

  const caseDoc = await Case.findOne({ _id: caseId, isDeleted: false })
    .populate('createdBy', 'name email role investigatorId badgeNumber department isActive')
    .populate('assignedTo', 'name email role investigatorId badgeNumber department isActive')
    .lean();

  if (!caseDoc) {
    throw new ApiError('Case not found', 404);
  }

  const creator = caseDoc.createdBy as any;
  const creatorId = extractIdString(creator);
  const assignedList = Array.isArray(caseDoc.assignedTo) ? (caseDoc.assignedTo as any[]) : [];

  const auditLogs = await SecurityAudit.find({
    event: { $in: ['CASE_PARTICIPANT_ADDED', 'CASE_PARTICIPANT_REMOVED'] },
    'details.caseId': caseId.toString(),
  })
    .populate('userId', 'name email role investigatorId')
    .sort({ createdAt: -1 })
    .lean();

  const canManageParticipants =
    userRole === 'admin' ||
    userRole === 'supervisor' ||
    creatorId === userId.toString();

  return {
    caseId: caseDoc._id.toString(),
    caseNumber: caseDoc.caseNumber,
    canManageParticipants,
    creator: creator
      ? {
          id: creatorId,
          name: creator.name || 'Lead Investigator',
          email: creator.email,
          role: creator.role,
          investigatorId: creator.investigatorId,
          badgeNumber: creator.badgeNumber,
          department: creator.department,
          participantRole: 'case_creator',
        }
      : null,
    assignedInvestigators: assignedList.map((u: any) => {
      const uId = extractIdString(u);
      const collab = ((caseDoc as any).collaborators || []).find(
        (c: any) => extractIdString(c.user) === uId
      );
      return {
        id: uId,
        name: u.name || 'Assigned Investigator',
        email: u.email,
        role: u.role,
        investigatorId: u.investigatorId,
        badgeNumber: u.badgeNumber,
        department: u.department,
        permission: collab?.permission || 'editor',
        participantRole:
          uId === creatorId ? 'case_creator' : 'assigned_investigator',
      };
    }),
    totalParticipants:
      1 +
      assignedList.filter((u: any) => extractIdString(u) !== creatorId).length,
    assignmentHistory: auditLogs.map((log: any) => ({
      auditId: log._id.toString(),
      event: log.event,
      timestamp: new Date(log.createdAt).toISOString(),
      actor: formatActor(log.userId),
      targetParticipantId: log.details?.participantId || null,
      targetParticipantName: log.details?.participantName || null,
      targetParticipantEmail: log.details?.participantEmail || null,
      targetInvestigatorId: log.details?.participantInvestigatorId || null,
      notes: log.details?.notes || log.details?.reason || null,
    })),
  };
};

/**
 * Assigns an active investigator to a case and logs CASE_PARTICIPANT_ADDED.
 */
export const addCaseParticipant = async (
  caseId: string,
  userId: string,
  userRole: string | undefined,
  input: AddParticipantInput,
  auditMeta?: { ipAddress?: string; userAgent?: string }
) => {
  validateObjectId(caseId, 'Case ID');
  await assertCaseAccess(caseId, userId, userRole);

  const caseDoc = await Case.findOne({ _id: caseId, isDeleted: false });
  if (!caseDoc) {
    throw new ApiError('Case not found', 404);
  }

  assertCanManageParticipants(caseDoc, userId, userRole);

  let targetUser: any = null;
  if (input.userId) {
    validateObjectId(input.userId, 'Participant User ID');
    targetUser = await User.findById(input.userId);
  } else if (input.investigatorId) {
    targetUser = await User.findOne({
      investigatorId: input.investigatorId.trim().toUpperCase(),
    });
  } else if (input.email) {
    targetUser = await User.findOne({
      email: input.email.trim().toLowerCase(),
    });
  }

  if (!targetUser) {
    throw new ApiError('Target investigator account not found', 404);
  }

  if (!targetUser.isActive) {
    throw new ApiError('Cannot assign an inactive user account to a case', 400);
  }

  const targetIdStr = targetUser._id.toString();
  const creatorIdStr = extractIdString(caseDoc.createdBy);

  if (targetIdStr === creatorIdStr) {
    throw new ApiError(
      'Investigator is already the case creator and primary owner of this case',
      409
    );
  }

  const alreadyAssigned = (caseDoc.assignedTo || []).some(
    (id) => id.toString() === targetIdStr
  );

  if (alreadyAssigned) {
    throw new ApiError('Investigator is already assigned to this case', 409);
  }

  caseDoc.assignedTo.push(new Types.ObjectId(targetIdStr));
  if (!caseDoc.collaborators) {
    caseDoc.collaborators = [];
  }
  const existingCollab = caseDoc.collaborators.find(
    (c) => c.user.toString() === targetIdStr
  );
  if (!existingCollab) {
    caseDoc.collaborators.push({
      user: new Types.ObjectId(targetIdStr),
      permission: (input as any).permission === 'read_only' ? 'read_only' : 'editor',
      addedAt: new Date(),
      addedBy: new Types.ObjectId(userId),
    });
  }
  await caseDoc.save();

  await SecurityAudit.create({
    userId: new Types.ObjectId(userId),
    event: 'CASE_PARTICIPANT_ADDED',
    status: 'success',
    ipAddress: auditMeta?.ipAddress,
    userAgent: auditMeta?.userAgent,
    details: {
      caseId: caseDoc._id.toString(),
      caseNumber: caseDoc.caseNumber,
      participantId: targetIdStr,
      participantName: targetUser.name,
      participantEmail: targetUser.email,
      participantRole: targetUser.role,
      participantInvestigatorId: targetUser.investigatorId,
      notes: input.notes || null,
    },
  });

  return getCaseParticipants(caseId, userId, userRole);
};

/**
 * Removes an assigned investigator from a case and logs CASE_PARTICIPANT_REMOVED.
 */
export const removeCaseParticipant = async (
  caseId: string,
  participantId: string,
  userId: string,
  userRole: string | undefined,
  reason?: string,
  auditMeta?: { ipAddress?: string; userAgent?: string }
) => {
  validateObjectId(caseId, 'Case ID');
  validateObjectId(participantId, 'Participant User ID');
  await assertCaseAccess(caseId, userId, userRole);

  const caseDoc = await Case.findOne({ _id: caseId, isDeleted: false });
  if (!caseDoc) {
    throw new ApiError('Case not found', 404);
  }

  assertCanManageParticipants(caseDoc, userId, userRole);

  const creatorIdStr = extractIdString(caseDoc.createdBy);
  if (participantId === creatorIdStr) {
    throw new ApiError(
      'Cannot remove the primary case creator from case ownership',
      400
    );
  }

  const assignedIndex = (caseDoc.assignedTo || []).findIndex(
    (id) => id.toString() === participantId
  );

  if (assignedIndex === -1) {
    throw new ApiError('Investigator is not currently assigned to this case', 404);
  }

  const targetUser = await User.findById(participantId)
    .select('name email role investigatorId')
    .lean();

  caseDoc.assignedTo.splice(assignedIndex, 1);
  if (caseDoc.collaborators) {
    caseDoc.collaborators = caseDoc.collaborators.filter(
      (c) => c.user.toString() !== participantId
    );
  }
  await caseDoc.save();

  await SecurityAudit.create({
    userId: new Types.ObjectId(userId),
    event: 'CASE_PARTICIPANT_REMOVED',
    status: 'success',
    ipAddress: auditMeta?.ipAddress,
    userAgent: auditMeta?.userAgent,
    details: {
      caseId: caseDoc._id.toString(),
      caseNumber: caseDoc.caseNumber,
      participantId,
      participantName: (targetUser as any)?.name || null,
      participantEmail: (targetUser as any)?.email || null,
      participantRole: (targetUser as any)?.role || null,
      participantInvestigatorId: (targetUser as any)?.investigatorId || null,
      reason: reason || null,
    },
  });

  return getCaseParticipants(caseId, userId, userRole);
};

/**
 * Validates that a linked resource belongs to the specified case.
 */
const verifyLinkedResourceOwnership = async (
  caseId: string,
  linkedResource: {
    resourceType: 'witness_description' | 'composite' | 'report';
    resourceId: string;
    label?: string;
  }
): Promise<{
  resourceType: 'witness_description' | 'composite' | 'report';
  resourceId: Types.ObjectId;
  label: string;
}> => {
  validateObjectId(linkedResource.resourceId, 'Linked Resource ID');

  if (linkedResource.resourceType === 'witness_description') {
    const wd = await WitnessDescription.findById(linkedResource.resourceId)
      .select('case witnessName isAnonymous')
      .lean();
    if (!wd || extractIdString(wd.case) !== caseId.toString()) {
      throw new ApiError(
        'Linked witness description does not exist or does not belong to this case',
        404
      );
    }
    const defaultLabel = wd.isAnonymous
      ? 'Anonymous Witness Statement'
      : `Witness Statement (${wd.witnessName || wd._id.toString().slice(-6)})`;
    return {
      resourceType: 'witness_description',
      resourceId: new Types.ObjectId(linkedResource.resourceId),
      label: linkedResource.label || defaultLabel,
    };
  }

  if (linkedResource.resourceType === 'composite') {
    const comp = await Composite.findById(linkedResource.resourceId)
      .select('case version editingType')
      .lean();
    if (!comp || extractIdString(comp.case) !== caseId.toString()) {
      throw new ApiError(
        'Linked composite does not exist or does not belong to this case',
        404
      );
    }
    return {
      resourceType: 'composite',
      resourceId: new Types.ObjectId(linkedResource.resourceId),
      label:
        linkedResource.label ||
        `Composite v${comp.version} (${comp.editingType || 'initial_generation'})`,
    };
  }

  if (linkedResource.resourceType === 'report') {
    const rep = await Report.findById(linkedResource.resourceId)
      .select('case reportNumber title')
      .lean();
    if (!rep || extractIdString(rep.case) !== caseId.toString()) {
      throw new ApiError(
        'Linked forensic report does not exist or does not belong to this case',
        404
      );
    }
    return {
      resourceType: 'report',
      resourceId: new Types.ObjectId(linkedResource.resourceId),
      label: linkedResource.label || `Report ${rep.reportNumber}`,
    };
  }

  throw new ApiError('Unsupported linked resource type', 400);
};

const serializeComment = (comment: any, currentUserId: string, currentUserRole?: string) => {
  const authorObj = comment.author;
  const authorIdStr = extractIdString(authorObj);
  const canModify =
    !comment.isDeleted &&
    (authorIdStr === currentUserId.toString() ||
      currentUserRole === 'admin' ||
      currentUserRole === 'supervisor');

  return {
    id: comment._id.toString(),
    caseId: extractIdString(comment.case),
    author: formatActor(authorObj),
    content: comment.isDeleted
      ? '[Comment removed — retained in forensic audit history]'
      : comment.content,
    originalContentPreservedInAudit: comment.isDeleted ? comment.content : undefined,
    noteType: comment.noteType as CaseCommentNoteType,
    linkedResource: comment.linkedResource
      ? {
          resourceType: comment.linkedResource.resourceType,
          resourceId: extractIdString(comment.linkedResource.resourceId),
          label: comment.linkedResource.label || '',
        }
      : null,
    isEdited: Boolean(comment.isEdited),
    editedAt: comment.editedAt ? new Date(comment.editedAt).toISOString() : null,
    editHistory: (comment.editHistory || []).map((entry: any) => ({
      previousContent: entry.previousContent,
      previousNoteType: entry.previousNoteType || 'general',
      editedBy: formatActor(entry.editedBy),
      editedAt: new Date(entry.editedAt).toISOString(),
      editReason: entry.editReason || null,
    })),
    isDeleted: Boolean(comment.isDeleted),
    deletedAt: comment.deletedAt ? new Date(comment.deletedAt).toISOString() : null,
    deletedBy: comment.deletedBy ? formatActor(comment.deletedBy) : null,
    deleteReason: comment.deleteReason || null,
    canModify,
    evidenceIsolationNotice:
      'Investigator collaboration note only — strictly isolated from witness testimony and GCIT forensic conditioning.',
    createdAt: new Date(comment.createdAt).toISOString(),
    updatedAt: new Date(comment.updatedAt).toISOString(),
  };
};

/**
 * Lists chronological investigator comments for a case.
 */
export const getCaseComments = async (
  caseId: string,
  userId: string,
  userRole?: string,
  includeDeleted = false
) => {
  validateObjectId(caseId, 'Case ID');
  await assertCaseAccess(caseId, userId, userRole);

  const query: Record<string, unknown> = {
    case: new Types.ObjectId(caseId),
  };
  if (!includeDeleted) {
    query.isDeleted = false;
  }

  const comments = await CaseComment.find(query)
    .populate('author', 'name email role investigatorId')
    .populate('editHistory.editedBy', 'name email role investigatorId')
    .populate('deletedBy', 'name email role investigatorId')
    .sort({ createdAt: 1 })
    .lean();

  return {
    caseId,
    totalComments: comments.length,
    includeDeleted,
    comments: comments.map((c) => serializeComment(c, userId, userRole)),
  };
};

/**
 * Creates a new investigator comment/note on a case.
 * Never modifies WitnessDescription or Composite records.
 */
export const addCaseComment = async (
  caseId: string,
  userId: string,
  userRole: string | undefined,
  input: CreateCommentInput,
  auditMeta?: { ipAddress?: string; userAgent?: string }
) => {
  validateObjectId(caseId, 'Case ID');
  await assertCaseAccess(caseId, userId, userRole);

  const caseDoc = await Case.findOne({ _id: caseId, isDeleted: false })
    .select('caseNumber title')
    .lean();
  if (!caseDoc) {
    throw new ApiError('Case not found', 404);
  }

  let verifiedResource:
    | {
        resourceType: 'witness_description' | 'composite' | 'report';
        resourceId: Types.ObjectId;
        label: string;
      }
    | undefined;

  if (input.linkedResource) {
    verifiedResource = await verifyLinkedResourceOwnership(caseId, input.linkedResource);
  }

  const created = await CaseComment.create({
    case: new Types.ObjectId(caseId),
    author: new Types.ObjectId(userId),
    content: input.content.trim(),
    noteType: input.noteType || 'general',
    linkedResource: verifiedResource,
    isEdited: false,
    editHistory: [],
    isDeleted: false,
  });

  const populated = await CaseComment.findById(created._id)
    .populate('author', 'name email role investigatorId')
    .lean();

  await SecurityAudit.create({
    userId: new Types.ObjectId(userId),
    event: 'CASE_COMMENT_ADDED',
    status: 'success',
    ipAddress: auditMeta?.ipAddress,
    userAgent: auditMeta?.userAgent,
    details: {
      caseId: caseId.toString(),
      caseNumber: caseDoc.caseNumber,
      commentId: created._id.toString(),
      noteType: created.noteType,
      linkedResource: verifiedResource
        ? {
            resourceType: verifiedResource.resourceType,
            resourceId: verifiedResource.resourceId.toString(),
            label: verifiedResource.label,
          }
        : null,
    },
  });

  return serializeComment(populated || created, userId, userRole);
};

/**
 * Updates an existing investigator comment while preserving prior content in editHistory.
 * Permitted only for the original comment author, supervisors, or admins.
 */
export const updateCaseComment = async (
  caseId: string,
  commentId: string,
  userId: string,
  userRole: string | undefined,
  input: UpdateCommentInput,
  auditMeta?: { ipAddress?: string; userAgent?: string }
) => {
  validateObjectId(caseId, 'Case ID');
  validateObjectId(commentId, 'Comment ID');
  await assertCaseAccess(caseId, userId, userRole);

  const comment = await CaseComment.findById(commentId);
  if (!comment || comment.case.toString() !== caseId.toString()) {
    throw new ApiError('Case comment not found for this case', 404);
  }

  if (comment.isDeleted) {
    throw new ApiError('Cannot edit a deleted case comment', 400);
  }

  const isAuthor = comment.author.toString() === userId.toString();
  const isPrivileged = userRole === 'admin' || userRole === 'supervisor';
  if (!isAuthor && !isPrivileged) {
    throw new ApiError(
      'Forbidden: Only the comment author, supervisors, or administrators can edit this comment',
      403
    );
  }

  const previousContent = comment.content;
  const previousNoteType = comment.noteType;

  comment.editHistory.push({
    previousContent,
    previousNoteType,
    editedBy: new Types.ObjectId(userId),
    editedAt: new Date(),
    editReason: input.editReason?.trim() || undefined,
  });

  comment.content = input.content.trim();
  if (input.noteType) {
    comment.noteType = input.noteType;
  }
  comment.isEdited = true;
  comment.editedAt = new Date();

  await comment.save();

  const populated = await CaseComment.findById(comment._id)
    .populate('author', 'name email role investigatorId')
    .populate('editHistory.editedBy', 'name email role investigatorId')
    .populate('deletedBy', 'name email role investigatorId')
    .lean();

  await SecurityAudit.create({
    userId: new Types.ObjectId(userId),
    event: 'CASE_COMMENT_EDITED',
    status: 'success',
    ipAddress: auditMeta?.ipAddress,
    userAgent: auditMeta?.userAgent,
    details: {
      caseId: caseId.toString(),
      commentId: comment._id.toString(),
      authorId: comment.author.toString(),
      editCount: comment.editHistory.length,
      editReason: input.editReason?.trim() || null,
    },
  });

  return serializeComment(populated || comment, userId, userRole);
};

/**
 * Soft-deletes a case comment while preserving full content and edit history for forensic audit.
 * Permitted only for the original comment author, supervisors, or admins.
 */
export const deleteCaseComment = async (
  caseId: string,
  commentId: string,
  userId: string,
  userRole: string | undefined,
  deleteReason?: string,
  auditMeta?: { ipAddress?: string; userAgent?: string }
) => {
  validateObjectId(caseId, 'Case ID');
  validateObjectId(commentId, 'Comment ID');
  await assertCaseAccess(caseId, userId, userRole);

  const comment = await CaseComment.findById(commentId);
  if (!comment || comment.case.toString() !== caseId.toString()) {
    throw new ApiError('Case comment not found for this case', 404);
  }

  if (comment.isDeleted) {
    throw new ApiError('Case comment is already deleted', 400);
  }

  const isAuthor = comment.author.toString() === userId.toString();
  const isPrivileged = userRole === 'admin' || userRole === 'supervisor';
  if (!isAuthor && !isPrivileged) {
    throw new ApiError(
      'Forbidden: Only the comment author, supervisors, or administrators can delete this comment',
      403
    );
  }

  comment.isDeleted = true;
  comment.deletedAt = new Date();
  comment.deletedBy = new Types.ObjectId(userId);
  comment.deleteReason = deleteReason?.trim() || undefined;

  await comment.save();

  const populated = await CaseComment.findById(comment._id)
    .populate('author', 'name email role investigatorId')
    .populate('editHistory.editedBy', 'name email role investigatorId')
    .populate('deletedBy', 'name email role investigatorId')
    .lean();

  await SecurityAudit.create({
    userId: new Types.ObjectId(userId),
    event: 'CASE_COMMENT_DELETED',
    status: 'success',
    ipAddress: auditMeta?.ipAddress,
    userAgent: auditMeta?.userAgent,
    details: {
      caseId: caseId.toString(),
      commentId: comment._id.toString(),
      authorId: comment.author.toString(),
      deleteReason: deleteReason?.trim() || null,
    },
  });

  return serializeComment(populated || comment, userId, userRole);
};

/**
 * Computes actual 7-stage investigative workflow progress from persisted database records.
 */
export const getCaseProgress = async (
  caseId: string,
  userId: string,
  userRole?: string
) => {
  validateObjectId(caseId, 'Case ID');
  await assertCaseAccess(caseId, userId, userRole);

  const [caseDoc, witnessDocs, compositeDocs, reportDocs, comparisonAudits] =
    await Promise.all([
      Case.findOne({ _id: caseId, isDeleted: false })
        .populate('createdBy', 'name email role investigatorId')
        .lean(),
      WitnessDescription.find({ case: new Types.ObjectId(caseId) })
        .populate('recordedBy', 'name email role investigatorId')
        .populate('reviewedBy', 'name email role investigatorId')
        .sort({ createdAt: 1 })
        .lean(),
      Composite.find({ case: new Types.ObjectId(caseId) })
        .populate('createdBy', 'name email role investigatorId')
        .populate('reviewedBy', 'name email role investigatorId')
        .populate('selectionHistory.selectedBy', 'name email role investigatorId')
        .sort({ version: 1 })
        .lean(),
      Report.find({ case: new Types.ObjectId(caseId) })
        .populate('generatedBy', 'name email role investigatorId')
        .sort({ createdAt: 1 })
        .lean(),
      SecurityAudit.find({
        event: 'COMPOSITE_VARIATIONS_COMPARED',
        'details.caseId': caseId.toString(),
      })
        .populate('userId', 'name email role investigatorId')
        .sort({ createdAt: -1 })
        .lean(),
    ]);

  if (!caseDoc) {
    throw new ApiError('Case not found', 404);
  }

  // Stage 1: Case Creation
  const stage1: WorkflowStageProgress = {
    stageKey: 'case_creation',
    stageNumber: 1,
    label: 'Case Creation',
    status: 'completed',
    summary: `Case ${caseDoc.caseNumber} initialized (${caseDoc.status}, ${caseDoc.priority} priority)`,
    completedAt: new Date(caseDoc.createdAt).toISOString(),
    actor: formatActor(caseDoc.createdBy),
    metrics: {
      caseNumber: caseDoc.caseNumber,
      caseStatus: caseDoc.status,
      priority: caseDoc.priority,
    },
  };

  // Stage 2: Witness Evidence Collection
  const validWitnessDocs = witnessDocs.filter(
    (w: any) => Boolean(w.rawStatement && w.rawStatement.trim())
  );
  const latestWitness =
    validWitnessDocs.length > 0 ? validWitnessDocs[validWitnessDocs.length - 1] : null;
  const stage2: WorkflowStageProgress = {
    stageKey: 'witness_evidence',
    stageNumber: 2,
    label: 'Witness Evidence',
    status: validWitnessDocs.length > 0 ? 'completed' : 'not_started',
    summary:
      validWitnessDocs.length > 0
        ? `${validWitnessDocs.length} witness statement(s) recorded`
        : 'No witness statements submitted yet',
    completedAt: latestWitness ? new Date(latestWitness.createdAt).toISOString() : null,
    actor: latestWitness ? formatActor(latestWitness.recordedBy) : null,
    metrics: {
      totalStatements: witnessDocs.length,
      validStatements: validWitnessDocs.length,
    },
  };

  // Stage 3: NLP Analysis
  const analyzedWitnessDocs = witnessDocs.filter(
    (w: any) => Array.isArray(w.extractedAttributes) && w.extractedAttributes.length > 0
  );
  const latestAnalyzed =
    analyzedWitnessDocs.length > 0
      ? analyzedWitnessDocs[analyzedWitnessDocs.length - 1]
      : null;
  const totalExtractedAttributes = analyzedWitnessDocs.reduce(
    (acc: number, w: any) => acc + (w.extractedAttributes?.length || 0),
    0
  );
  let stage3Status: WorkflowStageState = 'not_started';
  if (analyzedWitnessDocs.length > 0) {
    stage3Status = 'completed';
  } else if (validWitnessDocs.length > 0) {
    stage3Status = 'pending';
  }
  const stage3: WorkflowStageProgress = {
    stageKey: 'nlp_analysis',
    stageNumber: 3,
    label: 'NLP Analysis',
    status: stage3Status,
    summary:
      analyzedWitnessDocs.length > 0
        ? `BiLSTM-CRF extracted ${totalExtractedAttributes} attribute(s) across ${analyzedWitnessDocs.length} statement(s)`
        : validWitnessDocs.length > 0
        ? 'Witness statement awaiting BiLSTM-CRF attribute extraction'
        : 'Awaiting witness statement submission',
    completedAt: latestAnalyzed ? new Date(latestAnalyzed.updatedAt).toISOString() : null,
    actor: latestAnalyzed ? formatActor(latestAnalyzed.recordedBy) : null,
    metrics: {
      analyzedStatementsCount: analyzedWitnessDocs.length,
      totalExtractedAttributes,
    },
  };

  // Stage 4: Evidence Review & Approval
  const approvedWitnessDocs = witnessDocs.filter(
    (w: any) => w.reviewStatus === 'approved'
  );
  const rejectedWitnessDocs = witnessDocs.filter(
    (w: any) => w.reviewStatus === 'rejected'
  );
  const inReviewWitnessDocs = witnessDocs.filter(
    (w: any) => w.reviewStatus === 'in_review' || w.reviewStatus === 'pending'
  );
  const unresolvedContradictionsCount = analyzedWitnessDocs.reduce(
    (acc: number, w: any) =>
      acc +
      (w.extractedAttributes || []).filter(
        (attr: any) => attr.evidenceState === 'contradictory'
      ).length,
    0
  );
  const totalCorrectionsCount = witnessDocs.reduce(
    (acc: number, w: any) => acc + (w.corrections?.length || 0),
    0
  );

  let stage4Status: WorkflowStageState = 'not_started';
  let stage4Summary = 'Awaiting NLP attribute extraction';
  if (approvedWitnessDocs.length > 0) {
    stage4Status = 'completed';
    stage4Summary = `${approvedWitnessDocs.length} statement(s) approved (${totalCorrectionsCount} correction(s) logged)`;
  } else if (
    analyzedWitnessDocs.length > 0 &&
    (rejectedWitnessDocs.length === analyzedWitnessDocs.length ||
      unresolvedContradictionsCount > 0)
  ) {
    stage4Status = 'blocked';
    stage4Summary =
      rejectedWitnessDocs.length === analyzedWitnessDocs.length
        ? 'All analyzed witness statements are currently rejected'
        : `${unresolvedContradictionsCount} contradictory attribute(s) require investigator resolution before approval`;
  } else if (analyzedWitnessDocs.length > 0 && inReviewWitnessDocs.length > 0) {
    stage4Status = 'pending';
    stage4Summary = `${inReviewWitnessDocs.length} analyzed statement(s) pending investigator review and approval`;
  }

  const latestApproved =
    approvedWitnessDocs.length > 0
      ? approvedWitnessDocs[approvedWitnessDocs.length - 1]
      : null;

  const stage4: WorkflowStageProgress = {
    stageKey: 'evidence_approval',
    stageNumber: 4,
    label: 'Evidence Approval',
    status: stage4Status,
    summary: stage4Summary,
    completedAt:
      latestApproved && latestApproved.reviewedAt
        ? new Date(latestApproved.reviewedAt).toISOString()
        : latestApproved
        ? new Date(latestApproved.updatedAt).toISOString()
        : null,
    actor: latestApproved
      ? formatActor(latestApproved.reviewedBy || latestApproved.recordedBy)
      : null,
    metrics: {
      approvedCount: approvedWitnessDocs.length,
      rejectedCount: rejectedWitnessDocs.length,
      inReviewCount: inReviewWitnessDocs.length,
      totalCorrectionsCount,
      unresolvedContradictionsCount,
    },
  };

  // Stage 5: Composite Generation
  const latestComposite =
    compositeDocs.length > 0 ? compositeDocs[compositeDocs.length - 1] : null;
  const totalVariationsCount = compositeDocs.reduce(
    (acc: number, c: any) => acc + (c.variations?.length || 0),
    0
  );
  let stage5Status: WorkflowStageState = 'not_started';
  let stage5Summary = 'No facial composites generated yet';
  if (compositeDocs.length > 0) {
    stage5Status = 'completed';
    stage5Summary = `${compositeDocs.length} composite version(s) generated (${totalVariationsCount} total variations)`;
  } else if (stage4Status === 'blocked') {
    stage5Status = 'blocked';
    stage5Summary = 'Blocked until witness evidence review is resolved and approved';
  } else if (stage4Status === 'completed') {
    stage5Status = 'pending';
    stage5Summary = 'Approved witness evidence ready for conditional WGAN-GP generation';
  }

  const stage5: WorkflowStageProgress = {
    stageKey: 'composite_generation',
    stageNumber: 5,
    label: 'Composite Generation',
    status: stage5Status,
    summary: stage5Summary,
    completedAt: latestComposite ? new Date(latestComposite.createdAt).toISOString() : null,
    actor: latestComposite ? formatActor(latestComposite.createdBy) : null,
    metrics: {
      totalCompositesCount: compositeDocs.length,
      totalVariationsCount,
      latestVersion: latestComposite ? latestComposite.version : null,
    },
  };

  // Stage 6: Comparison & Selection
  const compositesWithSelection = compositeDocs.filter(
    (c: any) =>
      c.status === 'approved' ||
      (Array.isArray(c.selectionHistory) && c.selectionHistory.length > 0) ||
      (Array.isArray(c.variations) &&
        c.variations.some((v: any) => Boolean(v.approvedAt)))
  );
  const latestSelectedComp =
    compositesWithSelection.length > 0
      ? compositesWithSelection[compositesWithSelection.length - 1]
      : null;

  let latestSelectionTimestamp: string | null = null;
  let latestSelectionActor: TimelineActor | null = null;
  if (latestSelectedComp) {
    const selHist = (latestSelectedComp as any).selectionHistory || [];
    if (selHist.length > 0) {
      const lastEntry = selHist[selHist.length - 1];
      latestSelectionTimestamp = new Date(lastEntry.selectedAt).toISOString();
      latestSelectionActor = formatActor(lastEntry.selectedBy);
    } else if ((latestSelectedComp as any).reviewedAt) {
      latestSelectionTimestamp = new Date(
        (latestSelectedComp as any).reviewedAt
      ).toISOString();
      latestSelectionActor = formatActor(
        (latestSelectedComp as any).reviewedBy || (latestSelectedComp as any).createdBy
      );
    } else {
      latestSelectionTimestamp = new Date(
        (latestSelectedComp as any).createdAt
      ).toISOString();
      latestSelectionActor = formatActor((latestSelectedComp as any).createdBy);
    }
  }

  let stage6Status: WorkflowStageState = 'not_started';
  let stage6Summary = 'Awaiting composite generation';
  if (compositesWithSelection.length > 0) {
    stage6Status = 'completed';
    stage6Summary = `Preferred variation selected on ${compositesWithSelection.length} composite(s) (${comparisonAudits.length} comparison session(s) logged)`;
  } else if (compositeDocs.length > 0) {
    stage6Status = 'pending';
    stage6Summary =
      comparisonAudits.length > 0
        ? `Variations compared (${comparisonAudits.length} session(s)); awaiting preferred variation selection`
        : `${compositeDocs.length} composite(s) generated; awaiting variation comparison & selection`;
  }

  const stage6: WorkflowStageProgress = {
    stageKey: 'variation_selection',
    stageNumber: 6,
    label: 'Comparison & Selection',
    status: stage6Status,
    summary: stage6Summary,
    completedAt: latestSelectionTimestamp,
    actor: latestSelectionActor,
    metrics: {
      comparedSessionsCount: comparisonAudits.length,
      selectedCompositesCount: compositesWithSelection.length,
      approvedCompositesCount: compositeDocs.filter((c: any) => c.status === 'approved')
        .length,
    },
  };

  // Stage 7: Report Generation
  const latestReport = reportDocs.length > 0 ? reportDocs[reportDocs.length - 1] : null;
  let stage7Status: WorkflowStageState = 'not_started';
  let stage7Summary = 'No forensic dossier exported yet';
  if (reportDocs.length > 0 && latestReport) {
    stage7Status = 'completed';
    stage7Summary = `${reportDocs.length} forensic PDF dossier(s) generated (latest: ${latestReport.reportNumber})`;
  } else if (compositesWithSelection.length > 0) {
    stage7Status = 'pending';
    stage7Summary = 'Selected composite ready for forensic PDF dossier generation';
  }

  const stage7: WorkflowStageProgress = {
    stageKey: 'report_generation',
    stageNumber: 7,
    label: 'Report Generation',
    status: stage7Status,
    summary: stage7Summary,
    completedAt: latestReport ? new Date(latestReport.createdAt).toISOString() : null,
    actor: latestReport ? formatActor(latestReport.generatedBy) : null,
    metrics: {
      totalReportsCount: reportDocs.length,
      latestReportNumber: latestReport ? latestReport.reportNumber : null,
      latestReportSha256: latestReport ? latestReport.sha256 : null,
    },
  };

  const stages: WorkflowStageProgress[] = [
    stage1,
    stage2,
    stage3,
    stage4,
    stage5,
    stage6,
    stage7,
  ];

  const completedStagesCount = stages.filter((s) => s.status === 'completed').length;
  const blockedStage = stages.find((s) => s.status === 'blocked');
  const pendingStage = stages.find((s) => s.status === 'pending');
  const notStartedStage = stages.find((s) => s.status === 'not_started');

  const currentStage =
    blockedStage || pendingStage || notStartedStage || stages[stages.length - 1];

  return {
    caseId: caseDoc._id.toString(),
    caseNumber: caseDoc.caseNumber,
    caseStatus: caseDoc.status,
    completedStagesCount,
    totalStagesCount: stages.length,
    completionPercentage: Math.round((completedStagesCount / stages.length) * 100),
    hasBlockingIssues: Boolean(blockedStage),
    currentStage: {
      stageKey: currentStage.stageKey,
      stageNumber: currentStage.stageNumber,
      label: currentStage.label,
      status: currentStage.status,
    },
    stages,
  };
};

/**
 * Synthesizes a chronological case activity timeline from actual persisted database documents
 * and SecurityAudit logs without duplicating already-captured records.
 * Clearly distinguishes exact timestamped/audited events from historical state reconstructions.
 */
export const getCaseTimeline = async (
  caseId: string,
  userId: string,
  userRole?: string,
  options?: {
    order?: 'asc' | 'desc';
    category?: string;
    includeDeletedComments?: boolean;
  }
) => {
  validateObjectId(caseId, 'Case ID');
  await assertCaseAccess(caseId, userId, userRole);

  const order = options?.order === 'asc' ? 'asc' : 'desc';
  const categoryFilter =
    options?.category && options.category !== 'all' ? options.category : null;
  const includeDeletedComments = options?.includeDeletedComments ?? true;

  const [caseDoc, witnessDocs, compositeDocs, reportDocs, commentDocs] =
    await Promise.all([
      Case.findOne({ _id: caseId, isDeleted: false })
        .populate('createdBy', 'name email role investigatorId')
        .populate('assignedTo', 'name email role investigatorId')
        .lean(),
      WitnessDescription.find({ case: new Types.ObjectId(caseId) })
        .populate('recordedBy', 'name email role investigatorId')
        .populate('reviewedBy', 'name email role investigatorId')
        .populate('corrections.correctedBy', 'name email role investigatorId')
        .sort({ createdAt: 1 })
        .lean(),
      Composite.find({ case: new Types.ObjectId(caseId) })
        .populate('createdBy', 'name email role investigatorId')
        .populate('reviewedBy', 'name email role investigatorId')
        .populate('selectionHistory.selectedBy', 'name email role investigatorId')
        .sort({ version: 1 })
        .lean(),
      Report.find({ case: new Types.ObjectId(caseId) })
        .populate('generatedBy', 'name email role investigatorId')
        .sort({ createdAt: 1 })
        .lean(),
      CaseComment.find(
        includeDeletedComments
          ? { case: new Types.ObjectId(caseId) }
          : { case: new Types.ObjectId(caseId), isDeleted: false }
      )
        .populate('author', 'name email role investigatorId')
        .populate('editHistory.editedBy', 'name email role investigatorId')
        .populate('deletedBy', 'name email role investigatorId')
        .sort({ createdAt: 1 })
        .lean(),
    ]);

  if (!caseDoc) {
    throw new ApiError('Case not found', 404);
  }

  const caseIdStr = caseDoc._id.toString();
  const caseNumber = caseDoc.caseNumber;
  const compositeIds = compositeDocs.map((c: any) => c._id.toString());
  const reportIds = reportDocs.map((r: any) => r._id.toString());
  const witnessIds = witnessDocs.map((w: any) => w._id.toString());

  const auditOrClauses: Record<string, unknown>[] = [{ 'details.caseId': caseIdStr }];
  if (compositeIds.length > 0) {
    auditOrClauses.push({ 'details.compositeId': { $in: compositeIds } });
  }
  if (reportIds.length > 0) {
    auditOrClauses.push({ 'details.reportId': { $in: reportIds } });
  }
  if (witnessIds.length > 0) {
    auditOrClauses.push({ 'details.witnessDescriptionId': { $in: witnessIds } });
  }

  const auditLogs = await SecurityAudit.find({
    $or: auditOrClauses,
  })
    .populate('userId', 'name email role investigatorId')
    .sort({ createdAt: 1 })
    .lean();

  const events: TimelineEventItem[] = [];
  const consumedAuditIds = new Set<string>();

  // 1. Case Creation & Updates
  const caseCreatedAudit = auditLogs.find((a: any) => a.event === 'CASE_CREATED');
  if (caseCreatedAudit) {
    consumedAuditIds.add(caseCreatedAudit._id.toString());
  }
  events.push({
    eventId: `case-created-${caseIdStr}`,
    eventType: 'CASE_CREATED',
    category: 'case',
    timestamp: new Date(caseDoc.createdAt).toISOString(),
    actor: formatActor(caseDoc.createdBy || caseCreatedAudit?.userId),
    caseId: caseIdStr,
    caseNumber,
    resourceRef: {
      type: 'case',
      id: caseIdStr,
      label: `Case ${caseNumber}: ${caseDoc.title}`,
    },
    description: `Case ${caseNumber} ("${caseDoc.title}") initialized with ${caseDoc.priority} priority and ${caseDoc.status} status.`,
    metadata: {
      status: caseDoc.status,
      priority: caseDoc.priority,
      location: caseDoc.location || null,
      auditId: caseCreatedAudit ? caseCreatedAudit._id.toString() : null,
    },
    recordProvenance: caseCreatedAudit ? 'merged_audit_and_record' : 'persisted_record',
    provenanceNote: caseCreatedAudit
      ? 'Verified from persisted Case record and CASE_CREATED SecurityAudit log.'
      : 'Verified from persisted Case.createdAt and Case.createdBy fields.',
  });

  const caseUpdateAudits = auditLogs.filter((a: any) => a.event === 'CASE_UPDATED');
  for (const audit of caseUpdateAudits) {
    consumedAuditIds.add(audit._id.toString());
    const updatedFields = Array.isArray((audit.details as any)?.updatedFields)
      ? ((audit.details as any).updatedFields as string[]).join(', ')
      : 'metadata';
    events.push({
      eventId: `case-updated-${audit._id.toString()}`,
      eventType: 'CASE_UPDATED',
      category: 'case',
      timestamp: new Date(audit.createdAt).toISOString(),
      actor: formatActor(audit.userId),
      caseId: caseIdStr,
      caseNumber,
      resourceRef: {
        type: 'case',
        id: caseIdStr,
        label: `Case ${caseNumber}`,
      },
      description: `Case ${caseNumber} updated (${updatedFields}).`,
      metadata: {
        ...(audit.details || {}),
        auditId: audit._id.toString(),
      },
      recordProvenance: 'security_audit',
      provenanceNote: 'Verified from CASE_UPDATED SecurityAudit record.',
    });
  }

  // If case.updatedAt > case.createdAt and no CASE_UPDATED or participant audit explains it, record historical state note
  const participantAudits = auditLogs.filter(
    (a: any) =>
      a.event === 'CASE_PARTICIPANT_ADDED' || a.event === 'CASE_PARTICIPANT_REMOVED'
  );
  if (
    caseUpdateAudits.length === 0 &&
    participantAudits.length === 0 &&
    caseDoc.updatedAt &&
    new Date(caseDoc.updatedAt).getTime() - new Date(caseDoc.createdAt).getTime() > 2000
  ) {
    events.push({
      eventId: `case-historical-update-${caseIdStr}`,
      eventType: 'CASE_UPDATED_HISTORICAL',
      category: 'case',
      timestamp: new Date(caseDoc.updatedAt).toISOString(),
      actor: null,
      caseId: caseIdStr,
      caseNumber,
      resourceRef: {
        type: 'case',
        id: caseIdStr,
        label: `Case ${caseNumber}`,
      },
      description: `Case ${caseNumber} metadata modified (current status: ${caseDoc.status}, priority: ${caseDoc.priority}).`,
      metadata: {
        currentStatus: caseDoc.status,
        currentPriority: caseDoc.priority,
      },
      recordProvenance: 'historical_derived_state',
      provenanceNote:
        'Derived from Case.updatedAt timestamp on a pre-Phase 7.7 update; individual field diffs prior to CASE_UPDATED audit logging cannot be reconstructed.',
    });
  }

  // 2. Investigator Assignment Changes
  const auditedAddedParticipantIds = new Set<string>();
  for (const audit of participantAudits) {
    consumedAuditIds.add(audit._id.toString());
    const details = (audit.details || {}) as Record<string, any>;
    if (audit.event === 'CASE_PARTICIPANT_ADDED' && details.participantId) {
      auditedAddedParticipantIds.add(String(details.participantId));
    }
    const isAdd = audit.event === 'CASE_PARTICIPANT_ADDED';
    const pName =
      details.participantName ||
      details.participantEmail ||
      details.participantId ||
      'Investigator';
    events.push({
      eventId: `participant-audit-${audit._id.toString()}`,
      eventType: audit.event,
      category: 'collaboration',
      timestamp: new Date(audit.createdAt).toISOString(),
      actor: formatActor(audit.userId),
      caseId: caseIdStr,
      caseNumber,
      resourceRef: details.participantId
        ? {
            type: 'user',
            id: String(details.participantId),
            label: String(pName),
          }
        : null,
      description: isAdd
        ? `Investigator ${pName} assigned to Case ${caseNumber}${
            details.notes ? ` (${details.notes})` : ''
          }.`
        : `Investigator ${pName} removed from Case ${caseNumber}${
            details.reason ? ` (Reason: ${details.reason})` : ''
          }.`,
      metadata: {
        ...details,
        auditId: audit._id.toString(),
      },
      recordProvenance: 'security_audit',
      provenanceNote: `Verified from ${audit.event} SecurityAudit event.`,
    });
  }

  // Distinguish pre-existing assigned investigators without individual assignment audit logs
  const creatorIdStr = extractIdString(caseDoc.createdBy);
  for (const assigned of (caseDoc.assignedTo || []) as any[]) {
    const assignedIdStr = extractIdString(assigned);
    if (
      assignedIdStr &&
      assignedIdStr !== creatorIdStr &&
      !auditedAddedParticipantIds.has(assignedIdStr)
    ) {
      const actorInfo = formatActor(assigned);
      events.push({
        eventId: `participant-baseline-${caseIdStr}-${assignedIdStr}`,
        eventType: 'CASE_PARTICIPANT_ASSIGNED_INITIAL',
        category: 'collaboration',
        timestamp: new Date(caseDoc.createdAt).toISOString(),
        actor: formatActor(caseDoc.createdBy),
        caseId: caseIdStr,
        caseNumber,
        resourceRef: {
          type: 'user',
          id: assignedIdStr,
          label: actorInfo?.name || assignedIdStr,
        },
        description: `Investigator ${
          actorInfo?.name || assignedIdStr
        } listed in case participant roster.`,
        metadata: {
          participantId: assignedIdStr,
          participantName: actorInfo?.name,
          participantRole: actorInfo?.role,
        },
        recordProvenance: 'historical_derived_state',
        provenanceNote:
          'Derived from persisted Case.assignedTo array; assigned at case creation or prior to per-assignment audit logging.',
      });
    }
  }

  // 3. Witness Statement Submissions, NLP Analysis, Corrections, and Approvals
  for (const wd of witnessDocs as any[]) {
    const wdIdStr = wd._id.toString();
    const witnessLabel = wd.isAnonymous
      ? 'Anonymous Witness Statement'
      : `Witness Statement (${wd.witnessName || wdIdStr.slice(-6)})`;

    const statementAudit = auditLogs.find(
      (a: any) =>
        a.event === 'WITNESS_STATEMENT_CREATED' &&
        (a.details as any)?.witnessDescriptionId === wdIdStr
    );
    if (statementAudit) {
      consumedAuditIds.add(statementAudit._id.toString());
    }

    events.push({
      eventId: `witness-created-${wdIdStr}`,
      eventType: 'WITNESS_STATEMENT_SUBMITTED',
      category: 'evidence',
      timestamp: new Date(wd.createdAt).toISOString(),
      actor: formatActor(wd.recordedBy || statementAudit?.userId),
      caseId: caseIdStr,
      caseNumber,
      resourceRef: {
        type: 'witness_description',
        id: wdIdStr,
        label: witnessLabel,
      },
      description: `${witnessLabel} recorded (${
        (wd.rawStatement || '').length
      } chars).`,
      metadata: {
        witnessDescriptionId: wdIdStr,
        witnessName: wd.isAnonymous ? 'Anonymous' : wd.witnessName || null,
        statementLength: (wd.rawStatement || '').length,
        auditId: statementAudit ? statementAudit._id.toString() : null,
      },
      recordProvenance: statementAudit ? 'merged_audit_and_record' : 'persisted_record',
      provenanceNote: statementAudit
        ? 'Verified from persisted WitnessDescription document and WITNESS_STATEMENT_CREATED audit log.'
        : 'Verified from persisted WitnessDescription.createdAt and recordedBy fields.',
    });

    // NLP Analysis
    const nlpAudits = auditLogs.filter(
      (a: any) =>
        a.event === 'WITNESS_NLP_ANALYZED' &&
        (a.details as any)?.witnessDescriptionId === wdIdStr
    );
    if (nlpAudits.length > 0) {
      for (const nlpAudit of nlpAudits) {
        consumedAuditIds.add(nlpAudit._id.toString());
        const d = (nlpAudit.details || {}) as Record<string, any>;
        events.push({
          eventId: `witness-nlp-audit-${nlpAudit._id.toString()}`,
          eventType: 'WITNESS_NLP_ANALYZED',
          category: 'nlp',
          timestamp: new Date(nlpAudit.createdAt).toISOString(),
          actor: formatActor(nlpAudit.userId || wd.recordedBy),
          caseId: caseIdStr,
          caseNumber,
          resourceRef: {
            type: 'witness_description',
            id: wdIdStr,
            label: witnessLabel,
          },
          description: `BiLSTM-CRF NLP analysis completed on ${witnessLabel}: ${
            d.extractedAttributesCount ?? wd.extractedAttributes?.length ?? 0
          } attribute(s) extracted.`,
          metadata: {
            ...d,
            auditId: nlpAudit._id.toString(),
          },
          recordProvenance: 'merged_audit_and_record',
          provenanceNote:
            'Verified from WITNESS_NLP_ANALYZED SecurityAudit event and persisted WitnessDescription.extractedAttributes.',
        });
      }
    } else if (
      Array.isArray(wd.extractedAttributes) &&
      wd.extractedAttributes.length > 0
    ) {
      // Derived from persisted extractedAttributes + modelMetadata on historical record
      const nlpTimestamp = new Date(
        Math.min(
          new Date(wd.createdAt).getTime() + 1000,
          new Date(wd.updatedAt || wd.createdAt).getTime()
        )
      ).toISOString();
      events.push({
        eventId: `witness-nlp-persisted-${wdIdStr}`,
        eventType: 'WITNESS_NLP_ANALYZED',
        category: 'nlp',
        timestamp: nlpTimestamp,
        actor: formatActor(wd.recordedBy),
        caseId: caseIdStr,
        caseNumber,
        resourceRef: {
          type: 'witness_description',
          id: wdIdStr,
          label: witnessLabel,
        },
        description: `BiLSTM-CRF NLP analysis recorded on ${witnessLabel}: ${
          wd.extractedAttributes.length
        } attribute(s) extracted (${wd.modelMetadata?.architecture || 'CharAwareBiLSTM-CRF'}).`,
        metadata: {
          witnessDescriptionId: wdIdStr,
          extractedAttributesCount: wd.extractedAttributes.length,
          modelMetadata: wd.modelMetadata || null,
        },
        recordProvenance: 'historical_derived_state',
        provenanceNote:
          'Verified from persisted WitnessDescription.extractedAttributes and modelMetadata; exact NLP execution timestamp reconstructed relative to statement creation.',
      });
    }

    // Corrections on WitnessDescription
    const correctionAudits = auditLogs.filter(
      (a: any) =>
        a.event === 'WITNESS_EVIDENCE_CORRECTED' &&
        (a.details as any)?.witnessDescriptionId === wdIdStr
    );
    for (const ca of correctionAudits) {
      consumedAuditIds.add(ca._id.toString());
    }

    for (let idx = 0; idx < (wd.corrections || []).length; idx++) {
      const corr = wd.corrections[idx];
      const corrTs = corr.timestamp
        ? new Date(corr.timestamp).toISOString()
        : new Date(wd.updatedAt).toISOString();
      events.push({
        eventId: `witness-correction-${wdIdStr}-${idx}`,
        eventType: 'WITNESS_EVIDENCE_CORRECTED',
        category: 'evidence',
        timestamp: corrTs,
        actor: formatActor(corr.correctedBy),
        caseId: caseIdStr,
        caseNumber,
        resourceRef: {
          type: 'witness_description',
          id: wdIdStr,
          label: witnessLabel,
        },
        description: corr.rejected
          ? `Investigator rejected attribute "${corr.attributeCategory}" on ${witnessLabel} (Reason: ${corr.reason}).`
          : `Investigator corrected attribute "${corr.attributeCategory}" from "${
              corr.previousValue ?? 'none'
            }" (${corr.previousEvidenceState || 'unknown'}) to "${
              corr.correctedValue ?? 'none'
            }" (${corr.correctedEvidenceState || 'explicit'}) — Reason: ${corr.reason}.`,
        metadata: {
          witnessDescriptionId: wdIdStr,
          attributeCategory: corr.attributeCategory,
          previousValue: corr.previousValue ?? null,
          correctedValue: corr.correctedValue ?? null,
          previousEvidenceState: corr.previousEvidenceState,
          correctedEvidenceState: corr.correctedEvidenceState,
          reason: corr.reason,
          rejected: Boolean(corr.rejected),
        },
        recordProvenance: 'persisted_record',
        provenanceNote:
          'Verified from persisted WitnessDescription.corrections subdocument with exact investigator timestamp.',
      });
    }

    // Approvals / Rejections on WitnessDescription
    const approvalAudits = auditLogs.filter(
      (a: any) =>
        a.event === 'WITNESS_EVIDENCE_APPROVED' &&
        (a.details as any)?.witnessDescriptionId === wdIdStr
    );
    if (approvalAudits.length > 0) {
      for (const appAudit of approvalAudits) {
        consumedAuditIds.add(appAudit._id.toString());
        const statusVal = (appAudit.details as any)?.reviewStatus || wd.reviewStatus;
        events.push({
          eventId: `witness-approval-audit-${appAudit._id.toString()}`,
          eventType:
            statusVal === 'rejected'
              ? 'WITNESS_EVIDENCE_REJECTED'
              : 'WITNESS_EVIDENCE_APPROVED',
          category: 'evidence',
          timestamp: new Date(appAudit.createdAt).toISOString(),
          actor: formatActor(appAudit.userId || wd.reviewedBy),
          caseId: caseIdStr,
          caseNumber,
          resourceRef: {
            type: 'witness_description',
            id: wdIdStr,
            label: witnessLabel,
          },
          description: `${witnessLabel} review status set to "${statusVal}" by investigator.`,
          metadata: {
            witnessDescriptionId: wdIdStr,
            reviewStatus: statusVal,
            auditId: appAudit._id.toString(),
          },
          recordProvenance: 'merged_audit_and_record',
          provenanceNote:
            'Verified from WITNESS_EVIDENCE_APPROVED SecurityAudit event and persisted WitnessDescription.reviewStatus.',
        });
      }
    } else if (
      (wd.reviewStatus === 'approved' || wd.reviewStatus === 'rejected') &&
      wd.reviewedAt
    ) {
      events.push({
        eventId: `witness-approval-${wdIdStr}`,
        eventType:
          wd.reviewStatus === 'rejected'
            ? 'WITNESS_EVIDENCE_REJECTED'
            : 'WITNESS_EVIDENCE_APPROVED',
        category: 'evidence',
        timestamp: new Date(wd.reviewedAt).toISOString(),
        actor: formatActor(wd.reviewedBy || wd.recordedBy),
        caseId: caseIdStr,
        caseNumber,
        resourceRef: {
          type: 'witness_description',
          id: wdIdStr,
          label: witnessLabel,
        },
        description: `${witnessLabel} marked as "${wd.reviewStatus}" for forensic composite conditioning.`,
        metadata: {
          witnessDescriptionId: wdIdStr,
          reviewStatus: wd.reviewStatus,
        },
        recordProvenance: 'persisted_record',
        provenanceNote:
          'Verified from persisted WitnessDescription.reviewedAt and reviewedBy fields.',
      });
    }
  }

  // 4. Composite Generation, Semantic Editing, Comparison, and Selection
  for (const comp of compositeDocs as any[]) {
    const compIdStr = comp._id.toString();
    const isSemanticEdit = comp.editingType === 'semantic_latent_traversal';
    const compLabel = `Composite v${comp.version} (${
      isSemanticEdit ? 'Semantic Edit' : 'Initial Generation'
    })`;

    const genAudit = auditLogs.find(
      (a: any) =>
        (a.event === 'COMPOSITE_GENERATED_SUCCESS' ||
          a.event === 'COMPOSITE_SEMANTIC_EDIT') &&
        (a.details as any)?.compositeId === compIdStr
    );
    if (genAudit) {
      consumedAuditIds.add(genAudit._id.toString());
    }

    // Also consume corresponding COMPOSITE_GENERATION_INITIATED if paired with this success
    const initAudit = auditLogs.find(
      (a: any) =>
        a.event === 'COMPOSITE_GENERATION_INITIATED' &&
        Math.abs(
          new Date(a.createdAt).getTime() - new Date(comp.createdAt).getTime()
        ) < 15000
    );
    if (initAudit) {
      consumedAuditIds.add(initAudit._id.toString());
    }

    events.push({
      eventId: `composite-created-${compIdStr}`,
      eventType: isSemanticEdit
        ? 'COMPOSITE_SEMANTIC_EDIT'
        : 'COMPOSITE_GENERATED_SUCCESS',
      category: 'composite',
      timestamp: new Date(comp.createdAt).toISOString(),
      actor: formatActor(comp.createdBy || genAudit?.userId),
      caseId: caseIdStr,
      caseNumber,
      resourceRef: {
        type: 'composite',
        id: compIdStr,
        label: compLabel,
      },
      description: isSemanticEdit
        ? `Generated ${compLabel} via semantic attribute traversal (${
            comp.variations?.length || 1
          } variation(s)).`
        : `Generated ${compLabel} with ${
            comp.variations?.length || 1
          } multi-seed variation(s) from 20-D GCIT conditioning.`,
      metadata: {
        compositeId: compIdStr,
        version: comp.version,
        editingType: comp.editingType || 'initial_generation',
        variationsCount: comp.variations?.length || 1,
        parentCompositeId: comp.parentComposite
          ? comp.parentComposite.toString()
          : null,
        traversalParameters: comp.traversalParameters || null,
        auditId: genAudit ? genAudit._id.toString() : null,
      },
      recordProvenance: genAudit ? 'merged_audit_and_record' : 'persisted_record',
      provenanceNote: genAudit
        ? `Verified from persisted Composite v${comp.version} record and ${genAudit.event} SecurityAudit log.`
        : `Verified from persisted Composite v${comp.version} document.`,
    });

    // Selection History on Composite
    const selectionAudits = auditLogs.filter(
      (a: any) =>
        (a.event === 'COMPOSITE_VARIATION_SELECTED' ||
          a.event === 'COMPOSITE_APPROVED') &&
        (a.details as any)?.compositeId === compIdStr
    );
    for (const sa of selectionAudits) {
      consumedAuditIds.add(sa._id.toString());
    }

    if (Array.isArray(comp.selectionHistory) && comp.selectionHistory.length > 0) {
      for (let sIdx = 0; sIdx < comp.selectionHistory.length; sIdx++) {
        const sel = comp.selectionHistory[sIdx];
        const matchingAudit = selectionAudits[sIdx] || null;
        events.push({
          eventId: `composite-selection-${compIdStr}-${sIdx}`,
          eventType: 'COMPOSITE_VARIATION_SELECTED',
          category: 'comparison',
          timestamp: new Date(sel.selectedAt).toISOString(),
          actor: formatActor(sel.selectedBy || matchingAudit?.userId),
          caseId: caseIdStr,
          caseNumber,
          resourceRef: {
            type: 'composite',
            id: compIdStr,
            label: `${compLabel} — Variation #${sel.variationIndex + 1}`,
          },
          description: `Selected Variation #${sel.variationIndex + 1} (seed ${
            sel.seed ?? 'N/A'
          }) on Composite v${comp.version} with status "${sel.status}"${
            sel.notes ? ` — Notes: ${sel.notes}` : ''
          }.`,
          metadata: {
            compositeId: compIdStr,
            version: comp.version,
            variationIndex: sel.variationIndex,
            previousVariationIndex: sel.previousVariationIndex ?? null,
            seed: sel.seed ?? null,
            status: sel.status,
            notes: sel.notes || null,
            auditId: matchingAudit ? matchingAudit._id.toString() : null,
          },
          recordProvenance: matchingAudit
            ? 'merged_audit_and_record'
            : 'persisted_record',
          provenanceNote: matchingAudit
            ? 'Verified from Composite.selectionHistory and COMPOSITE_VARIATION_SELECTED SecurityAudit log.'
            : 'Verified from persisted Composite.selectionHistory entry.',
        });
      }
    } else if (selectionAudits.length > 0) {
      for (const sa of selectionAudits) {
        const d = (sa.details || {}) as Record<string, any>;
        events.push({
          eventId: `composite-selection-audit-${sa._id.toString()}`,
          eventType: sa.event,
          category: 'comparison',
          timestamp: new Date(sa.createdAt).toISOString(),
          actor: formatActor(sa.userId || comp.reviewedBy),
          caseId: caseIdStr,
          caseNumber,
          resourceRef: {
            type: 'composite',
            id: compIdStr,
            label: compLabel,
          },
          description: `Selected preferred variation #${
            typeof d.variationIndex === 'number' ? d.variationIndex + 1 : 1
          } on Composite v${comp.version}.`,
          metadata: {
            ...d,
            auditId: sa._id.toString(),
          },
          recordProvenance: 'security_audit',
          provenanceNote: `Verified from ${sa.event} SecurityAudit log.`,
        });
      }
    } else if (comp.status === 'approved' && comp.reviewedAt) {
      events.push({
        eventId: `composite-approved-${compIdStr}`,
        eventType: 'COMPOSITE_APPROVED',
        category: 'comparison',
        timestamp: new Date(comp.reviewedAt).toISOString(),
        actor: formatActor(comp.reviewedBy || comp.createdBy),
        caseId: caseIdStr,
        caseNumber,
        resourceRef: {
          type: 'composite',
          id: compIdStr,
          label: compLabel,
        },
        description: `Composite v${comp.version} approved (selected variation #${
          (comp.selectedVariationIndex ?? 0) + 1
        }).`,
        metadata: {
          compositeId: compIdStr,
          version: comp.version,
          selectedVariationIndex: comp.selectedVariationIndex ?? 0,
        },
        recordProvenance: 'persisted_record',
        provenanceNote:
          'Verified from persisted Composite.reviewedAt and reviewedBy fields.',
      });
    }
  }

  // 5. Report Generation & Downloads
  for (const rep of reportDocs as any[]) {
    const repIdStr = rep._id.toString();
    const repGenAudit = auditLogs.find(
      (a: any) =>
        a.event === 'REPORT_GENERATED_SUCCESS' &&
        (a.details as any)?.reportId === repIdStr
    );
    if (repGenAudit) {
      consumedAuditIds.add(repGenAudit._id.toString());
    }

    events.push({
      eventId: `report-generated-${repIdStr}`,
      eventType: 'REPORT_GENERATED_SUCCESS',
      category: 'report',
      timestamp: new Date(rep.createdAt).toISOString(),
      actor: formatActor(rep.generatedBy || repGenAudit?.userId),
      caseId: caseIdStr,
      caseNumber,
      resourceRef: {
        type: 'report',
        id: repIdStr,
        label: `Report ${rep.reportNumber}`,
      },
      description: `Forensic Dossier ${rep.reportNumber} ("${rep.title}") generated for Composite v${rep.compositeVersion} (SHA-256: ${rep.sha256.slice(
        0,
        12
      )}...).`,
      metadata: {
        reportId: repIdStr,
        reportNumber: rep.reportNumber,
        title: rep.title,
        compositeVersion: rep.compositeVersion,
        sha256: rep.sha256,
        auditId: repGenAudit ? repGenAudit._id.toString() : null,
      },
      recordProvenance: repGenAudit ? 'merged_audit_and_record' : 'persisted_record',
      provenanceNote: repGenAudit
        ? 'Verified from persisted Report record and REPORT_GENERATED_SUCCESS SecurityAudit log.'
        : 'Verified from persisted Report document.',
    });
  }

  // 6. Case Comments (Creation, Edits, Deletions)
  const commentAudits = auditLogs.filter(
    (a: any) =>
      a.event === 'CASE_COMMENT_ADDED' ||
      a.event === 'CASE_COMMENT_EDITED' ||
      a.event === 'CASE_COMMENT_DELETED'
  );
  for (const ca of commentAudits) {
    consumedAuditIds.add(ca._id.toString());
  }

  for (const cDoc of commentDocs as any[]) {
    const cIdStr = cDoc._id.toString();
    const addAudit = commentAudits.find(
      (a: any) =>
        a.event === 'CASE_COMMENT_ADDED' &&
        (a.details as any)?.commentId === cIdStr
    );

    const previewText = cDoc.isDeleted
      ? '[Comment subsequently soft-deleted — preserved in audit history]'
      : cDoc.content.length > 120
      ? `${cDoc.content.slice(0, 120)}...`
      : cDoc.content;

    events.push({
      eventId: `comment-added-${cIdStr}`,
      eventType: 'CASE_COMMENT_ADDED',
      category: 'comment',
      timestamp: new Date(cDoc.createdAt).toISOString(),
      actor: formatActor(cDoc.author),
      caseId: caseIdStr,
      caseNumber,
      resourceRef: {
        type: 'case_comment',
        id: cIdStr,
        label: `Investigator Note (${cDoc.noteType || 'general'})`,
      },
      description: `Investigator note posted [${cDoc.noteType || 'general'}]${
        cDoc.linkedResource?.label ? ` linked to ${cDoc.linkedResource.label}` : ''
      }: "${previewText}"`,
      metadata: {
        commentId: cIdStr,
        noteType: cDoc.noteType || 'general',
        linkedResource: cDoc.linkedResource
          ? {
              resourceType: cDoc.linkedResource.resourceType,
              resourceId: extractIdString(cDoc.linkedResource.resourceId),
              label: cDoc.linkedResource.label,
            }
          : null,
        isDeleted: Boolean(cDoc.isDeleted),
        auditId: addAudit ? addAudit._id.toString() : null,
      },
      recordProvenance: addAudit ? 'merged_audit_and_record' : 'persisted_record',
      provenanceNote: addAudit
        ? 'Verified from persisted CaseComment document and CASE_COMMENT_ADDED SecurityAudit log.'
        : 'Verified from persisted CaseComment document.',
    });

    // Comment edits
    for (let eIdx = 0; eIdx < (cDoc.editHistory || []).length; eIdx++) {
      const editEntry = cDoc.editHistory[eIdx];
      events.push({
        eventId: `comment-edited-${cIdStr}-${eIdx}`,
        eventType: 'CASE_COMMENT_EDITED',
        category: 'comment',
        timestamp: new Date(editEntry.editedAt).toISOString(),
        actor: formatActor(editEntry.editedBy || cDoc.author),
        caseId: caseIdStr,
        caseNumber,
        resourceRef: {
          type: 'case_comment',
          id: cIdStr,
          label: `Investigator Note (${cDoc.noteType || 'general'})`,
        },
        description: `Investigator note edited${
          editEntry.editReason ? ` (Reason: ${editEntry.editReason})` : ''
        }. Prior version preserved in edit history.`,
        metadata: {
          commentId: cIdStr,
          editIndex: eIdx + 1,
          previousNoteType: editEntry.previousNoteType || 'general',
          editReason: editEntry.editReason || null,
        },
        recordProvenance: 'persisted_record',
        provenanceNote:
          'Verified from persisted CaseComment.editHistory entry and CASE_COMMENT_EDITED audit trail.',
      });
    }

    // Comment soft-deletion
    if (cDoc.isDeleted && cDoc.deletedAt) {
      events.push({
        eventId: `comment-deleted-${cIdStr}`,
        eventType: 'CASE_COMMENT_DELETED',
        category: 'comment',
        timestamp: new Date(cDoc.deletedAt).toISOString(),
        actor: formatActor(cDoc.deletedBy || cDoc.author),
        caseId: caseIdStr,
        caseNumber,
        resourceRef: {
          type: 'case_comment',
          id: cIdStr,
          label: `Investigator Note (${cDoc.noteType || 'general'})`,
        },
        description: `Investigator note soft-deleted${
          cDoc.deleteReason ? ` (Reason: ${cDoc.deleteReason})` : ''
        }; original text preserved in forensic audit log.`,
        metadata: {
          commentId: cIdStr,
          deleteReason: cDoc.deleteReason || null,
        },
        recordProvenance: 'persisted_record',
        provenanceNote:
          'Verified from persisted CaseComment soft-delete metadata and CASE_COMMENT_DELETED audit log.',
      });
    }
  }

  // 7. Remaining Unconsumed SecurityAudit Events (e.g., COMPOSITE_VARIATIONS_COMPARED, REPORT_DOWNLOADED, COMPOSITE_GENERATION_FAILURE, REPORT_GENERATION_FAILURE)
  for (const audit of auditLogs as any[]) {
    const auditIdStr = audit._id.toString();
    if (consumedAuditIds.has(auditIdStr)) {
      continue;
    }

    const d = (audit.details || {}) as Record<string, any>;
    let category: TimelineCategory = 'case';
    let description = `Audit event ${audit.event} recorded.`;
    let resourceRef: TimelineResourceRef | null = null;

    if (audit.event === 'COMPOSITE_VARIATIONS_COMPARED') {
      category = 'comparison';
      description = `Multi-variation side-by-side comparison inspected for Composite v${
        d.version ?? '?'
      } (${d.variationCount ?? '?'} variations).`;
      if (d.compositeId) {
        resourceRef = {
          type: 'composite',
          id: String(d.compositeId),
          label: `Composite v${d.version ?? '?'}`,
        };
      }
    } else if (audit.event === 'REPORT_DOWNLOADED') {
      category = 'report';
      description = `Forensic PDF Dossier ${
        d.reportNumber || ''
      } downloaded and integrity-verified.`;
      if (d.reportId) {
        resourceRef = {
          type: 'report',
          id: String(d.reportId),
          label: `Report ${d.reportNumber || String(d.reportId).slice(-6)}`,
        };
      }
    } else if (audit.event === 'COMPOSITE_GENERATION_FAILURE') {
      category = 'composite';
      description = `Composite generation attempt failed: ${
        d.reason || 'validation or inference error'
      }.`;
    } else if (audit.event === 'REPORT_GENERATION_FAILURE') {
      category = 'report';
      description = `Forensic report export failed: ${d.error || 'generation error'}.`;
    } else if (audit.event === 'COMPOSITE_VIEWED') {
      // Skip noisy passive view events from cluttering the primary investigative action timeline
      continue;
    } else {
      description = `Recorded ${audit.event} (${audit.status}).`;
    }

    events.push({
      eventId: `audit-${auditIdStr}`,
      eventType: audit.event,
      category,
      timestamp: new Date(audit.createdAt).toISOString(),
      actor: formatActor(audit.userId),
      caseId: caseIdStr,
      caseNumber,
      resourceRef,
      description,
      metadata: {
        ...d,
        auditId: auditIdStr,
        status: audit.status,
      },
      recordProvenance: 'security_audit',
      provenanceNote: `Verified directly from ${audit.event} SecurityAudit record.`,
    });
  }

  // Filter by category if requested
  const filteredEvents = categoryFilter
    ? events.filter((ev) => ev.category === categoryFilter)
    : events;

  // Deterministic chronological sort
  filteredEvents.sort((a, b) => {
    const diff = new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
    if (diff !== 0) {
      return order === 'asc' ? diff : -diff;
    }
    return order === 'asc'
      ? a.eventId.localeCompare(b.eventId)
      : b.eventId.localeCompare(a.eventId);
  });

  const provenanceCounts = {
    security_audit: filteredEvents.filter(
      (e) => e.recordProvenance === 'security_audit'
    ).length,
    persisted_record: filteredEvents.filter(
      (e) => e.recordProvenance === 'persisted_record'
    ).length,
    merged_audit_and_record: filteredEvents.filter(
      (e) => e.recordProvenance === 'merged_audit_and_record'
    ).length,
    historical_derived_state: filteredEvents.filter(
      (e) => e.recordProvenance === 'historical_derived_state'
    ).length,
  };

  return {
    caseId: caseIdStr,
    caseNumber,
    order,
    categoryFilter: categoryFilter || 'all',
    totalEvents: filteredEvents.length,
    provenanceSummary: provenanceCounts,
    events: filteredEvents,
  };
};

/**
 * Convenience aggregator returning progress, participants, comments, and timeline in a single authorized call.
 */
export const getCaseCollaborationWorkspace = async (
  caseId: string,
  userId: string,
  userRole?: string
) => {
  validateObjectId(caseId, 'Case ID');
  await assertCaseAccess(caseId, userId, userRole);

  const [progress, participants, commentsData, timeline] = await Promise.all([
    getCaseProgress(caseId, userId, userRole),
    getCaseParticipants(caseId, userId, userRole),
    getCaseComments(caseId, userId, userRole, true),
    getCaseTimeline(caseId, userId, userRole, {
      order: 'desc',
      includeDeletedComments: true,
    }),
  ]);

  return {
    caseId,
    caseNumber: progress.caseNumber,
    progress,
    participants,
    comments: commentsData,
    timeline,
  };
};
