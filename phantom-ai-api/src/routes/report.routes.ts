import { Router } from 'express';
import {
  generateReport,
  getCaseReports,
  getReportById,
  downloadReportFile,
  getAllReports,
} from '../controllers/report.controller';
import { authenticate } from '../middlewares/auth.middleware';
import { validate } from '../middlewares/validate.middleware';
import {
  generateReportSchema,
  reportIdParamsSchema,
} from '../validations/report.validation';

// Case-scoped router: /api/v1/cases/:caseId/reports
export const caseReportRouter = Router({ mergeParams: true });
caseReportRouter.use(authenticate);

caseReportRouter.post('/', validate(generateReportSchema), generateReport);
caseReportRouter.get('/', getCaseReports);
caseReportRouter.get('/:reportId', validate(reportIdParamsSchema), getReportById);
caseReportRouter.get('/:reportId/download', validate(reportIdParamsSchema), downloadReportFile);

// Global user-scoped router: /api/v1/reports
export const globalReportRouter = Router();
globalReportRouter.use(authenticate);

globalReportRouter.get('/', getAllReports);

export default caseReportRouter;

