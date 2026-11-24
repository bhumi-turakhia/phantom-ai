import { Request, Response } from 'express';
import { asyncHandler } from '../utils/asyncHandler';
import { ApiResponse, ApiError } from '../utils/apiResponse';
import * as promptGenerationService from '../services/promptGeneration.service';
import { Types } from 'mongoose';
import { WitnessDescription } from '../models/WitnessDescription.model';

export const generatePrompt = asyncHandler(
  async (req: Request, res: Response) => {
    const { witnessDescriptionId } = req.params;
    const userId = (req as any).user?.id as string;

    const result =
      await promptGenerationService.generatePromptForWitnessDescription(
        witnessDescriptionId,
        userId
      );

    return ApiResponse.success(
      res,
      'Prompt generated successfully',
      result,
      201
    );
  }
);

export const getPromptByJobId = asyncHandler(
  async (req: Request, res: Response) => {
    const { jobId } = req.params;

    const result = await promptGenerationService.getPromptByJobId(jobId);

    return ApiResponse.success(
      res,
      'Generation job fetched successfully',
      result,
      200
    );
  }
);

export const previewPrompt = asyncHandler(
  async (req: Request, res: Response) => {
    const { witnessDescriptionId } = req.params;

    if (!Types.ObjectId.isValid(witnessDescriptionId)) {
      throw new ApiError('Invalid Witness Description ID', 400);
    }

    const witnessDescription = await WitnessDescription.findById(
      witnessDescriptionId
    );

    if (!witnessDescription) {
      throw new ApiError('Witness description not found', 404);
    }

    const prompt =
      promptGenerationService.buildPromptFromWitnessDescription(
        witnessDescription
      );

    return ApiResponse.success(
      res,
      'Prompt preview generated successfully',
      {
        prompt,
        witnessDescriptionId,
        note: 'This is a preview only. No generation job was created.',
      },
      200
    );
  }
);