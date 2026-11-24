import { Router } from 'express';
import {
  createCase,
  getAllCases,
  getCaseById,
  updateCase,
  resolveCase,
  reopenCase,
  archiveCase,
  restoreCase,
  deleteCase,
} from '../controllers/case.controller';
import {
  getCaseWorkspace,
  getCaseTimeline,
  getCaseProgress,
  getEligibleInvestigators,
  getCaseParticipants,
  addCaseParticipant,
  removeCaseParticipant,
  getCaseComments,
  addCaseComment,
  updateCaseComment,
  deleteCaseComment,
} from '../controllers/caseCollaboration.controller';
import { authenticate } from '../middlewares/auth.middleware';
import { validate } from '../middlewares/validate.middleware';
import {
  createCaseSchema,
  updateCaseSchema,
  caseIdParamSchema,
  resolveCaseSchema,
  reopenCaseSchema,
  archiveCaseSchema,
  restoreCaseSchema,
  deleteCaseSchema,
} from '../validations/case.validation';
import {
  caseIdOnlyParamSchema,
  timelineQuerySchema,
  addParticipantSchema,
  removeParticipantSchema,
  listCommentsQuerySchema,
  createCommentSchema,
  updateCommentSchema,
  deleteCommentSchema,
} from '../validations/caseCollaboration.validation';
import compositeRoutes from './composite.routes';
import { caseReportRouter } from './report.routes';

const router = Router();

router.use(authenticate);

/**
 * @route   POST /api/v1/cases
 * @desc    Create a new case
 * @access  Private
 */
router.post('/', validate(createCaseSchema), createCase);

/**
 * @route   GET /api/v1/cases
 * @desc    Get all cases with optional filters and pagination
 * @access  Private
 */
router.get('/', getAllCases);

/**
 * @route   GET /api/v1/cases/:id
 * @desc    Get a single case by ID
 * @access  Private
 */
router.get('/:id', validate(caseIdParamSchema), getCaseById);

/**
 * @route   PATCH /api/v1/cases/:id
 * @desc    Update a case by ID
 * @access  Private
 */
router.patch('/:id', validate(updateCaseSchema), updateCase);

/**
 * @route   POST /api/v1/cases/:id/resolve
 * @desc    Mark an active case as resolved with summary and reason
 * @access  Private (Owner, Supervisor, Admin)
 */
router.post('/:id/resolve', validate(resolveCaseSchema), resolveCase);

/**
 * @route   POST /api/v1/cases/:id/reopen
 * @desc    Reopen a resolved case back to active status
 * @access  Private (Owner, Supervisor, Admin)
 */
router.post('/:id/reopen', validate(reopenCaseSchema), reopenCase);

/**
 * @route   POST /api/v1/cases/:id/archive
 * @desc    Archive a resolved case for historical retention (read-only)
 * @access  Private (Owner, Supervisor, Admin)
 */
router.post('/:id/archive', validate(archiveCaseSchema), archiveCase);

/**
 * @route   POST /api/v1/cases/:id/restore
 * @desc    Restore an archived case back to accessible status
 * @access  Private (Owner, Supervisor, Admin)
 */
router.post('/:id/restore', validate(restoreCaseSchema), restoreCase);

/**
 * @route   DELETE /api/v1/cases/:id
 * @desc    Soft delete a case by ID (Owner, Supervisor, Admin)
 * @access  Private
 */
router.delete('/:id', validate(deleteCaseSchema), deleteCase);

/**
 * @route   GET /api/v1/cases/:id/workspace
 * @desc    Get aggregated case collaboration workspace (progress, participants, comments, timeline)
 * @access  Private
 */
router.get('/:id/workspace', validate(caseIdOnlyParamSchema), getCaseWorkspace);

/**
 * @route   GET /api/v1/cases/:id/timeline
 * @desc    Get chronological case activity timeline synthesized from persisted records and audit logs
 * @access  Private
 */
router.get('/:id/timeline', validate(timelineQuerySchema), getCaseTimeline);

/**
 * @route   GET /api/v1/cases/:id/progress
 * @desc    Get 7-stage investigative workflow progress for a case
 * @access  Private
 */
router.get('/:id/progress', validate(caseIdOnlyParamSchema), getCaseProgress);

/**
 * @route   GET /api/v1/cases/:id/eligible-investigators
 * @desc    Get active agency investigators available for assignment
 * @access  Private
 */
router.get(
  '/:id/eligible-investigators',
  validate(caseIdOnlyParamSchema),
  getEligibleInvestigators
);

/**
 * @route   GET /api/v1/cases/:id/participants
 * @desc    Get case creator, assigned investigators, and assignment audit history
 * @access  Private
 */
router.get('/:id/participants', validate(caseIdOnlyParamSchema), getCaseParticipants);

/**
 * @route   POST /api/v1/cases/:id/participants
 * @desc    Assign an active investigator to a case
 * @access  Private (Case Creator, Supervisor, or Admin)
 */
router.post('/:id/participants', validate(addParticipantSchema), addCaseParticipant);

/**
 * @route   DELETE /api/v1/cases/:id/participants/:participantId
 * @desc    Remove an assigned investigator from a case
 * @access  Private (Case Creator, Supervisor, or Admin)
 */
router.delete(
  '/:id/participants/:participantId',
  validate(removeParticipantSchema),
  removeCaseParticipant
);

/**
 * @route   GET /api/v1/cases/:id/comments
 * @desc    Get chronological investigator collaboration comments for a case
 * @access  Private
 */
router.get('/:id/comments', validate(listCommentsQuerySchema), getCaseComments);

/**
 * @route   POST /api/v1/cases/:id/comments
 * @desc    Post a new investigator comment/note on a case (strictly isolated from witness evidence)
 * @access  Private
 */
router.post('/:id/comments', validate(createCommentSchema), addCaseComment);

/**
 * @route   PATCH /api/v1/cases/:id/comments/:commentId
 * @desc    Edit an existing investigator comment while preserving prior content in editHistory
 * @access  Private (Comment Author, Supervisor, or Admin)
 */
router.patch(
  '/:id/comments/:commentId',
  validate(updateCommentSchema),
  updateCaseComment
);

/**
 * @route   DELETE /api/v1/cases/:id/comments/:commentId
 * @desc    Soft-delete an investigator comment while preserving content in forensic audit history
 * @access  Private (Comment Author, Supervisor, or Admin)
 */
router.delete(
  '/:id/comments/:commentId',
  validate(deleteCommentSchema),
  deleteCaseComment
);

/**
 * @route   /api/v1/cases/:caseId/composites
 * @desc    Forensic facial composite management endpoints
 */
router.use('/:caseId/composites', compositeRoutes);

/**
 * @route   /api/v1/cases/:caseId/reports
 * @desc    Forensic investigation PDF report management endpoints
 */
router.use('/:caseId/reports', caseReportRouter);

export default router;