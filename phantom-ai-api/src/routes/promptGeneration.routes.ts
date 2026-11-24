import { Router } from 'express';
import {
  generatePrompt,
  getPromptByJobId,
  previewPrompt,
} from '../controllers/promptGeneration.controller';
import { authenticate } from '../middlewares/auth.middleware';
import { validate } from '../middlewares/validate.middleware';
import { z } from 'zod';

const witnessDescriptionIdSchema = z.object({
  params: z.object({
    witnessDescriptionId: z
      .string()
      .min(1, 'Witness Description ID is required'),
  }),
});

const jobIdSchema = z.object({
  params: z.object({
    jobId: z
      .string()
      .min(1, 'Job ID is required'),
  }),
});

const router = Router();

router.use(authenticate);

/**
 * @route   POST /api/v1/prompt-generation/:witnessDescriptionId
 * @desc    Generate an AI image prompt from a witness description and create a GenerationJob
 * @access  Private
 */
router.post(
  '/:witnessDescriptionId',
  validate(witnessDescriptionIdSchema),
  generatePrompt
);

/**
 * @route   GET /api/v1/prompt-generation/preview/:witnessDescriptionId
 * @desc    Preview the generated prompt without creating a GenerationJob
 * @access  Private
 */
router.get(
  '/preview/:witnessDescriptionId',
  validate(witnessDescriptionIdSchema),
  previewPrompt
);

/**
 * @route   GET /api/v1/prompt-generation/job/:jobId
 * @desc    Fetch a GenerationJob and its prompt by job ID
 * @access  Private
 */
router.get(
  '/job/:jobId',
  validate(jobIdSchema),
  getPromptByJobId
);

export default router;