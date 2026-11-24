import { Request, Response } from 'express';
import { asyncHandler } from '../utils/asyncHandler';
import { ApiResponse } from '../utils/apiResponse';
import * as imageGenerationService from '../services/imageGeneration.service';

export const generateImage = asyncHandler(
  async (req: Request, res: Response) => {
    const { jobId } = req.params;
    const userId = (req as any).user?.id as string;

    const updatedJob = await imageGenerationService.generateImageForJob(
      jobId,
      userId
    );

    return ApiResponse.success(
      res,
      'Image generated successfully',
      updatedJob,
      200
    );
  }
);

export const getJobById = asyncHandler(
  async (req: Request, res: Response) => {
    const { jobId } = req.params;
    const userId = (req as any).user?.id as string;

    const job = await imageGenerationService.getJobById(jobId, userId);

    return ApiResponse.success(
      res,
      'Generation job fetched successfully',
      job,
      200
    );
  }
);

export const getJobsByWitnessDescription = asyncHandler(
  async (req: Request, res: Response) => {
    const { witnessDescriptionId } = req.params;
    const userId = (req as any).user?.id as string;

    const jobs = await imageGenerationService.getJobsByWitnessDescription(
      witnessDescriptionId,
      userId
    );

    return ApiResponse.success(
      res,
      'Generation jobs fetched successfully',
      jobs,
      200
    );
  }
);

export const cancelJob = asyncHandler(
  async (req: Request, res: Response) => {
    const { jobId } = req.params;
    const userId = (req as any).user?.id as string;

    const cancelled = await imageGenerationService.cancelJob(jobId, userId);

    return ApiResponse.success(
      res,
      'Generation job cancelled successfully',
      cancelled,
      200
    );
  }
);