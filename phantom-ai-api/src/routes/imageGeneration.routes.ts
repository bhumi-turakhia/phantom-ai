import { Router } from 'express';
import { z } from 'zod';
import {
  generateImage,
  getJobById,
  getJobsByWitnessDescription,
  cancelJob,
} from '../controllers/imageGeneration.controller';
import { authenticate } from '../middlewares/auth.middleware';
import { validate } from '../middlewares/validate.middleware';

const jobIdSchema = z.object({
  params: z.object({
    jobId: z.string().min(1, 'Job ID is required'),
  }),
});

const witnessDescriptionIdSchema = z.object({
  params: z.object({
    witnessDescriptionId: z
      .string()
      .min(1, 'Witness Description ID is required'),
  }),
});

const router = Router();

router.use(authenticate);

/**
 * @route   POST /api/v1/image-generation/:jobId/generate
 * @desc    Generate image using Hugging Face SDXL for an existing GenerationJob
 * @access  Private
 */
router.post(
  '/:jobId/generate',
  validate(jobIdSchema),
  generateImage
);

/**
 * @route   GET /api/v1/image-generation/job/:jobId
 * @desc    Get a GenerationJob by ID
 * @access  Private
 */
router.get(
  '/job/:jobId',
  validate(jobIdSchema),
  getJobById
);

/**
 * @route   GET /api/v1/image-generation/witness/:witnessDescriptionId
 * @desc    Get all GenerationJobs for a WitnessDescription
 * @access  Private
 */
router.get(
  '/witness/:witnessDescriptionId',
  validate(witnessDescriptionIdSchema),
  getJobsByWitnessDescription
);

/**
 * @route   PATCH /api/v1/image-generation/:jobId/cancel
 * @desc    Cancel a pending GenerationJob
 * @access  Private
 */
router.patch(
  '/:jobId/cancel',
  validate(jobIdSchema),
  cancelJob
);

export default router;