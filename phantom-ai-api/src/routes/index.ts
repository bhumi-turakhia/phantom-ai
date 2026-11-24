import { Router, Request, Response } from 'express';
import authRoutes from './auth.routes';
import caseRoutes from './case.routes';
import witnessDescriptionRoutes from './witnessDescription.routes';
import promptGenerationRoutes from './promptGeneration.routes';
import aiWitnessAnalysisRoutes from './aiWitnessAnalysis.routes';
import imageGenerationRoutes from './imageGeneration.routes';
import { globalReportRouter } from './report.routes';
import dashboardRoutes from './dashboard.routes';

const router = Router();

router.get('/health', (_req: Request, res: Response) => {
  res.status(200).json({ status: 'ok' });
});

router.use('/auth', authRoutes);
router.use('/cases', caseRoutes);
router.use('/witness-descriptions', witnessDescriptionRoutes);
router.use('/prompt-generation', promptGenerationRoutes);
router.use('/ai-analysis', aiWitnessAnalysisRoutes);
router.use('/image-generation', imageGenerationRoutes);
router.use('/reports', globalReportRouter);
router.use('/dashboard', dashboardRoutes);

export default router;