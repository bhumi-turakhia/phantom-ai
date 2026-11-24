import { Request, Response } from 'express';
import { asyncHandler } from '../utils/asyncHandler';
import { ApiResponse } from '../utils/apiResponse';
import * as caseCollaborationService from '../services/caseCollaboration.service';

export const getCaseWorkspace = asyncHandler(async (req: Request, res: Response) => {
  const { id } = req.params;
  const userId = (req as any).user?.id as string;
  const userRole = (req as any).user?.role as string | undefined;

  const workspace = await caseCollaborationService.getCaseCollaborationWorkspace(
    id,
    userId,
    userRole
  );

  return ApiResponse.success(
    res,
    'Case collaboration workspace retrieved successfully',
    workspace,
    200
  );
});

export const getCaseTimeline = asyncHandler(async (req: Request, res: Response) => {
  const { id } = req.params;
  const userId = (req as any).user?.id as string;
  const userRole = (req as any).user?.role as string | undefined;
  const { order, category, includeDeletedComments } = req.query;

  const timeline = await caseCollaborationService.getCaseTimeline(id, userId, userRole, {
    order: order === 'asc' ? 'asc' : 'desc',
    category: category as string | undefined,
    includeDeletedComments: includeDeletedComments !== 'false',
  });

  return ApiResponse.success(
    res,
    'Case activity timeline retrieved successfully',
    timeline,
    200
  );
});

export const getCaseProgress = asyncHandler(async (req: Request, res: Response) => {
  const { id } = req.params;
  const userId = (req as any).user?.id as string;
  const userRole = (req as any).user?.role as string | undefined;

  const progress = await caseCollaborationService.getCaseProgress(id, userId, userRole);

  return ApiResponse.success(
    res,
    'Case workflow progress retrieved successfully',
    progress,
    200
  );
});

export const getEligibleInvestigators = asyncHandler(
  async (req: Request, res: Response) => {
    const { id } = req.params;
    const userId = (req as any).user?.id as string;
    const userRole = (req as any).user?.role as string | undefined;

    const investigators = await caseCollaborationService.getEligibleInvestigators(
      id,
      userId,
      userRole
    );

    return ApiResponse.success(
      res,
      'Eligible active investigators retrieved successfully',
      investigators,
      200
    );
  }
);

export const getCaseParticipants = asyncHandler(async (req: Request, res: Response) => {
  const { id } = req.params;
  const userId = (req as any).user?.id as string;
  const userRole = (req as any).user?.role as string | undefined;

  const participants = await caseCollaborationService.getCaseParticipants(
    id,
    userId,
    userRole
  );

  return ApiResponse.success(
    res,
    'Case participants retrieved successfully',
    participants,
    200
  );
});

export const addCaseParticipant = asyncHandler(async (req: Request, res: Response) => {
  const { id } = req.params;
  const userId = (req as any).user?.id as string;
  const userRole = (req as any).user?.role as string | undefined;

  const participants = await caseCollaborationService.addCaseParticipant(
    id,
    userId,
    userRole,
    req.body,
    {
      ipAddress: req.ip,
      userAgent: req.headers['user-agent'],
    }
  );

  return ApiResponse.success(
    res,
    'Investigator assigned to case successfully',
    participants,
    201
  );
});

export const removeCaseParticipant = asyncHandler(
  async (req: Request, res: Response) => {
    const { id, participantId } = req.params;
    const userId = (req as any).user?.id as string;
    const userRole = (req as any).user?.role as string | undefined;
    const reason = req.body?.reason as string | undefined;

    const participants = await caseCollaborationService.removeCaseParticipant(
      id,
      participantId,
      userId,
      userRole,
      reason,
      {
        ipAddress: req.ip,
        userAgent: req.headers['user-agent'],
      }
    );

    return ApiResponse.success(
      res,
      'Investigator removed from case successfully',
      participants,
      200
    );
  }
);

export const getCaseComments = asyncHandler(async (req: Request, res: Response) => {
  const { id } = req.params;
  const userId = (req as any).user?.id as string;
  const userRole = (req as any).user?.role as string | undefined;
  const includeDeleted = req.query.includeDeleted === 'true';

  const comments = await caseCollaborationService.getCaseComments(
    id,
    userId,
    userRole,
    includeDeleted
  );

  return ApiResponse.success(
    res,
    'Case comments retrieved successfully',
    comments,
    200
  );
});

export const addCaseComment = asyncHandler(async (req: Request, res: Response) => {
  const { id } = req.params;
  const userId = (req as any).user?.id as string;
  const userRole = (req as any).user?.role as string | undefined;

  const comment = await caseCollaborationService.addCaseComment(
    id,
    userId,
    userRole,
    req.body,
    {
      ipAddress: req.ip,
      userAgent: req.headers['user-agent'],
    }
  );

  return ApiResponse.success(
    res,
    'Case comment posted successfully',
    comment,
    201
  );
});

export const updateCaseComment = asyncHandler(async (req: Request, res: Response) => {
  const { id, commentId } = req.params;
  const userId = (req as any).user?.id as string;
  const userRole = (req as any).user?.role as string | undefined;

  const comment = await caseCollaborationService.updateCaseComment(
    id,
    commentId,
    userId,
    userRole,
    req.body,
    {
      ipAddress: req.ip,
      userAgent: req.headers['user-agent'],
    }
  );

  return ApiResponse.success(
    res,
    'Case comment updated successfully',
    comment,
    200
  );
});

export const deleteCaseComment = asyncHandler(async (req: Request, res: Response) => {
  const { id, commentId } = req.params;
  const userId = (req as any).user?.id as string;
  const userRole = (req as any).user?.role as string | undefined;
  const deleteReason = req.body?.deleteReason as string | undefined;

  const comment = await caseCollaborationService.deleteCaseComment(
    id,
    commentId,
    userId,
    userRole,
    deleteReason,
    {
      ipAddress: req.ip,
      userAgent: req.headers['user-agent'],
    }
  );

  return ApiResponse.success(
    res,
    'Case comment removed and archived in audit log successfully',
    comment,
    200
  );
});

