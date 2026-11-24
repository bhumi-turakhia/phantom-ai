import { Request, Response } from 'express';
import { asyncHandler } from '../utils/asyncHandler';
import { ApiResponse } from '../utils/apiResponse';
import * as aiWitnessAnalysisService from '../services/aiWitnessAnalysis.service';

export const analyzeWitnessStatement = asyncHandler(
  async (req: Request, res: Response) => {
    const { witnessDescriptionId } = req.params;

    const userId = (req as any).user?.id as string;
    const userRole = (req as any).user?.role as string | undefined;

    const result =
      await aiWitnessAnalysisService.analyzeWitnessStatement(
        witnessDescriptionId,
        userId,
        userRole
      );

    return ApiResponse.success(
      res,
      'Witness statement analyzed and attributes extracted successfully',
      result,
      200
    );
  }
);