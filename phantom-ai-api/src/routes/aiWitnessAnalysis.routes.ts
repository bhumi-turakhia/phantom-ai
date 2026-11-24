import { Router } from 'express';
import { z } from 'zod';
import { analyzeWitnessStatement } from '../controllers/aiWitnessAnalysis.controller';
import { authenticate } from '../middlewares/auth.middleware';
import { validate } from '../middlewares/validate.middleware';

const witnessDescriptionIdSchema = z.object({
  params: z.object({
    witnessDescriptionId: z
      .string()
      .min(1, 'Witness Description ID is required'),
  }),
});

const router = Router();

router.use(authenticate);

router.post(
  '/:witnessDescriptionId',
  validate(witnessDescriptionIdSchema),
  analyzeWitnessStatement
);

export default router;