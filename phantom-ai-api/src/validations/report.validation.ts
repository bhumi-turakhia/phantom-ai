import { z } from 'zod';

export const generateReportSchema = z.object({
  params: z.object({
    caseId: z.string({ required_error: 'Case ID is required' }).min(1),
  }),
  body: z.object({
    witnessDescriptionId: z.string().optional(),
    compositeId: z.string().optional(),
    compositeVersion: z.number().int().min(1).optional(),
    title: z.string().max(255).optional(),
    notes: z.string().max(4000).optional(),
  }),
});

export const reportIdParamsSchema = z.object({
  params: z.object({
    caseId: z.string({ required_error: 'Case ID is required' }).min(1),
    reportId: z.string({ required_error: 'Report ID is required' }).min(1),
  }),
});

export const reportListQuerySchema = z.object({
  params: z.object({
    caseId: z.string().optional(),
  }),
  query: z.object({
    page: z.string().optional().transform((val) => (val ? parseInt(val, 10) : 1)),
    limit: z.string().optional().transform((val) => (val ? parseInt(val, 10) : 20)),
  }),
});

export type GenerateReportBodyInput = z.infer<typeof generateReportSchema>['body'];

