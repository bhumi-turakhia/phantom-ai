import { Request, Response } from 'express';
import { asyncHandler } from '../utils/asyncHandler';
import { ApiResponse } from '../utils/apiResponse';
import * as caseService from '../services/case.service';
import { CaseStatus, CasePriority } from '../models/Case.model';

export const createCase = asyncHandler(async (req: Request, res: Response) => {
  const userId = (req as any).user?.id as string;

  const newCase = await caseService.createCase(req.body, userId);

  return ApiResponse.success(res, 'Case created successfully', newCase, 201);
});

export const getAllCases = asyncHandler(async (req: Request, res: Response) => {
  const { status, priority, search, page, limit, view } = req.query;
  const userId = (req as any).user?.id as string | undefined;
  const userRole = (req as any).user?.role as string | undefined;

  const result = await caseService.getAllCases(
    {
      status: status as CaseStatus | undefined,
      priority: priority as CasePriority | undefined,
      search: search as string | undefined,
      view: view as any,
      page: page ? parseInt(page as string, 10) : undefined,
      limit: limit ? parseInt(limit as string, 10) : undefined,
    },
    userId,
    userRole
  );

  return ApiResponse.success(res, 'Cases fetched successfully', result, 200);
});

export const getCaseById = asyncHandler(async (req: Request, res: Response) => {
  const { id } = req.params;
  const userId = (req as any).user?.id as string | undefined;
  const userRole = (req as any).user?.role as string | undefined;

  const caseDoc = await caseService.getCaseById(id, userId, userRole);

  return ApiResponse.success(res, 'Case fetched successfully', caseDoc, 200);
});

export const updateCase = asyncHandler(async (req: Request, res: Response) => {
  const { id } = req.params;
  const userId = (req as any).user?.id as string;
  const userRole = (req as any).user?.role as string | undefined;

  const updated = await caseService.updateCase(id, req.body, userId, userRole);

  return ApiResponse.success(res, 'Case updated successfully', updated, 200);
});

export const resolveCase = asyncHandler(async (req: Request, res: Response) => {
  const { id } = req.params;
  const userId = (req as any).user?.id as string;
  const userRole = (req as any).user?.role as string | undefined;
  const ipAddress = req.ip || req.socket.remoteAddress;
  const userAgent = req.headers['user-agent'];

  const resolved = await caseService.resolveCase(id, req.body, userId, userRole, {
    ipAddress,
    userAgent,
  });

  return ApiResponse.success(res, 'Case marked as resolved successfully', resolved, 200);
});

export const reopenCase = asyncHandler(async (req: Request, res: Response) => {
  const { id } = req.params;
  const userId = (req as any).user?.id as string;
  const userRole = (req as any).user?.role as string | undefined;
  const ipAddress = req.ip || req.socket.remoteAddress;
  const userAgent = req.headers['user-agent'];

  const reopened = await caseService.reopenCase(id, req.body, userId, userRole, {
    ipAddress,
    userAgent,
  });

  return ApiResponse.success(res, 'Case reopened successfully', reopened, 200);
});

export const archiveCase = asyncHandler(async (req: Request, res: Response) => {
  const { id } = req.params;
  const userId = (req as any).user?.id as string;
  const userRole = (req as any).user?.role as string | undefined;
  const ipAddress = req.ip || req.socket.remoteAddress;
  const userAgent = req.headers['user-agent'];

  const archived = await caseService.archiveCase(id, req.body, userId, userRole, {
    ipAddress,
    userAgent,
  });

  return ApiResponse.success(res, 'Case archived successfully', archived, 200);
});

export const restoreCase = asyncHandler(async (req: Request, res: Response) => {
  const { id } = req.params;
  const userId = (req as any).user?.id as string;
  const userRole = (req as any).user?.role as string | undefined;
  const ipAddress = req.ip || req.socket.remoteAddress;
  const userAgent = req.headers['user-agent'];

  const restored = await caseService.restoreCase(id, req.body, userId, userRole, {
    ipAddress,
    userAgent,
  });

  return ApiResponse.success(res, 'Case restored from archive successfully', restored, 200);
});

export const deleteCase = asyncHandler(async (req: Request, res: Response) => {
  const { id } = req.params;
  const userId = (req as any).user?.id as string;
  const userRole = (req as any).user?.role as string | undefined;
  const ipAddress = req.ip || req.socket.remoteAddress;
  const userAgent = req.headers['user-agent'];

  await caseService.deleteCase(id, userId, userRole, req.body, { ipAddress, userAgent });

  return ApiResponse.success(res, 'Case safely soft-deleted from active workspace', null, 200);
});
