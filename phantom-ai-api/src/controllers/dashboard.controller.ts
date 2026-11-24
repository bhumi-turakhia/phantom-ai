import { Request, Response } from 'express';
import { asyncHandler } from '../utils/asyncHandler';
import { ApiResponse } from '../utils/apiResponse';
import * as dashboardService from '../services/dashboard.service';

export const getDashboardOverview = asyncHandler(async (req: Request, res: Response) => {
  const userId = (req as any).user?.id as string;
  const userRole = (req as any).user?.role as string | undefined;

  const overview = await dashboardService.getDashboardOverview(userId, userRole);

  return ApiResponse.success(res, 'Dashboard telemetry retrieved successfully', overview, 200);
});

