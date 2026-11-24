import { Request, Response } from 'express';
import { asyncHandler } from '../utils/asyncHandler';
import { ApiResponse } from '../utils/apiResponse';
import * as witnessDescriptionService from '../services/witnessDescription.service';

export const createWitnessDescription = asyncHandler(
  async (req: Request, res: Response) => {
    const userId = (req as any).user?.id as string;
    const userRole = (req as any).user?.role as string | undefined;

    const witnessDescription =
      await witnessDescriptionService.createWitnessDescription(
        req.body,
        userId,
        userRole
      );

    return ApiResponse.success(
      res,
      'Witness description created successfully',
      witnessDescription,
      201
    );
  }
);

export const getWitnessDescriptionsByCase = asyncHandler(
  async (req: Request, res: Response) => {
    const { caseId } = req.params;
    const userId = (req as any).user?.id as string | undefined;
    const userRole = (req as any).user?.role as string | undefined;
    const { page, limit, gender, build, skinTone } = req.query;

    const result =
      await witnessDescriptionService.getWitnessDescriptionsByCase(
        caseId,
        {
          page: page ? parseInt(page as string, 10) : undefined,
          limit: limit ? parseInt(limit as string, 10) : undefined,
          gender: gender as string | undefined,
          build: build as string | undefined,
          skinTone: skinTone as string | undefined,
        },
        userId,
        userRole
      );

    return ApiResponse.success(
      res,
      'Witness descriptions fetched successfully',
      result,
      200
    );
  }
);

export const getWitnessDescriptionById = asyncHandler(
  async (req: Request, res: Response) => {
    const { id } = req.params;
    const userId = (req as any).user?.id as string | undefined;
    const userRole = (req as any).user?.role as string | undefined;

    const witnessDescription =
      await witnessDescriptionService.getWitnessDescriptionById(
        id,
        userId,
        userRole
      );

    return ApiResponse.success(
      res,
      'Witness description fetched successfully',
      witnessDescription,
      200
    );
  }
);

export const updateWitnessDescription = asyncHandler(
  async (req: Request, res: Response) => {
    const { id } = req.params;
    const userId = (req as any).user?.id as string;
    const userRole = (req as any).user?.role as string | undefined;

    const updated =
      await witnessDescriptionService.updateWitnessDescription(
        id,
        req.body,
        userId,
        userRole
      );

    return ApiResponse.success(
      res,
      'Witness description updated successfully',
      updated,
      200
    );
  }
);

export const deleteWitnessDescription = asyncHandler(
  async (req: Request, res: Response) => {
    const { id } = req.params;
    const userId = (req as any).user?.id as string;
    const userRole = (req as any).user?.role as string | undefined;

    await witnessDescriptionService.deleteWitnessDescription(id, userId, userRole);

    return ApiResponse.success(
      res,
      'Witness description deleted successfully',
      null,
      200
    );
  }
);

export const addCorrection = asyncHandler(
  async (req: Request, res: Response) => {
    const { id } = req.params;
    const userId = (req as any).user?.id as string;
    const userRole = (req as any).user?.role as string | undefined;

    const updated = await witnessDescriptionService.addCorrection(
      id,
      req.body,
      userId,
      userRole
    );

    return ApiResponse.success(
      res,
      'Investigator correction recorded successfully',
      updated,
      200
    );
  }
);

export const updateApprovalStatus = asyncHandler(
  async (req: Request, res: Response) => {
    const { id } = req.params;
    const userId = (req as any).user?.id as string;
    const userRole = (req as any).user?.role as string | undefined;
    const { status } = req.body;

    const updated = await witnessDescriptionService.updateApprovalStatus(
      id,
      status || 'approved',
      userId,
      userRole
    );

    return ApiResponse.success(
      res,
      `Witness analysis ${status || 'approved'} successfully`,
      updated,
      200
    );
  }
);

export const getWitnessDescriptionTrace = asyncHandler(
  async (req: Request, res: Response) => {
    const { id } = req.params;
    const userId = (req as any).user?.id as string;
    const userRole = (req as any).user?.role as string | undefined;

    const trace = await witnessDescriptionService.getWitnessDescriptionTrace(
      id,
      userId,
      userRole
    );

    return ApiResponse.success(
      res,
      'Witness description evidence trace retrieved successfully',
      trace,
      200
    );
  }
);
