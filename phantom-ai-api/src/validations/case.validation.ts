import { z } from 'zod';

export const createCaseSchema = z.object({
  body: z.object({
    caseNumber: z
      .string({ required_error: 'Case number is required' })
      .trim()
      .toUpperCase()
      .min(1, 'Case number is required'),
    title: z
      .string({ required_error: 'Case title is required' })
      .trim()
      .min(1, 'Case title is required')
      .max(200, 'Title must not exceed 200 characters'),
    description: z
      .string()
      .trim()
      .max(5000, 'Description must not exceed 5000 characters')
      .optional(),
    status: z
      .enum(['open', 'in_progress', 'under_review', 'resolved', 'archived', 'closed'], {
        errorMap: () => ({
          message: 'Status must be open, in_progress, under_review, resolved, archived, or closed',
        }),
      })
      .optional(),
    priority: z
      .enum(['low', 'medium', 'high', 'critical'], {
        errorMap: () => ({ message: 'Priority must be low, medium, high, or critical' }),
      })
      .optional(),
    assignedTo: z.array(z.string().trim()).optional(),
    location: z
      .object({
        address: z.string().trim().optional(),
        city: z.string().trim().optional(),
        state: z.string().trim().optional(),
        coordinates: z
          .object({
            lat: z.number().min(-90).max(90),
            lng: z.number().min(-180).max(180),
          })
          .optional(),
      })
      .optional(),
    incidentDate: z
      .string()
      .datetime({ message: 'Incident date must be a valid ISO datetime string' })
      .optional(),
    tags: z.array(z.string().trim().toLowerCase()).optional(),
  }),
});

export const updateCaseSchema = z.object({
  body: z.object({
    title: z
      .string()
      .trim()
      .min(1, 'Case title is required')
      .max(200, 'Title must not exceed 200 characters')
      .optional(),
    description: z
      .string()
      .trim()
      .max(5000, 'Description must not exceed 5000 characters')
      .optional(),
    status: z
      .enum(['open', 'in_progress', 'under_review', 'resolved', 'archived', 'closed'], {
        errorMap: () => ({
          message: 'Status must be open, in_progress, under_review, resolved, archived, or closed',
        }),
      })
      .optional(),
    priority: z
      .enum(['low', 'medium', 'high', 'critical'], {
        errorMap: () => ({ message: 'Priority must be low, medium, high, or critical' }),
      })
      .optional(),
    assignedTo: z.array(z.string().trim()).optional(),
    location: z
      .object({
        address: z.string().trim().optional(),
        city: z.string().trim().optional(),
        state: z.string().trim().optional(),
        coordinates: z
          .object({
            lat: z.number().min(-90).max(90),
            lng: z.number().min(-180).max(180),
          })
          .optional(),
      })
      .optional(),
    incidentDate: z
      .string()
      .datetime({ message: 'Incident date must be a valid ISO datetime string' })
      .optional(),
    tags: z.array(z.string().trim().toLowerCase()).optional(),
  }),
  params: z.object({
    id: z.string().min(1, 'Case ID is required'),
  }),
});

export const caseIdParamSchema = z.object({
  params: z.object({
    id: z.string().min(1, 'Case ID is required'),
  }),
});

export const resolveCaseSchema = z.object({
  params: z.object({
    id: z.string().min(1, 'Case ID is required'),
  }),
  body: z
    .object({
      summary: z.string().trim().max(2000, 'Summary cannot exceed 2000 characters').optional(),
      reason: z.string().trim().max(1000, 'Reason cannot exceed 1000 characters').optional(),
    })
    .optional(),
});

export const reopenCaseSchema = z.object({
  params: z.object({
    id: z.string().min(1, 'Case ID is required'),
  }),
  body: z.object({
    reason: z
      .string({ required_error: 'Reopen reason is required' })
      .trim()
      .min(1, 'Reopen reason is required')
      .max(1000, 'Reason cannot exceed 1000 characters'),
  }),
});

export const archiveCaseSchema = z.object({
  params: z.object({
    id: z.string().min(1, 'Case ID is required'),
  }),
  body: z
    .object({
      reason: z.string().trim().max(1000, 'Reason cannot exceed 1000 characters').optional(),
    })
    .optional(),
});

export const restoreCaseSchema = z.object({
  params: z.object({
    id: z.string().min(1, 'Case ID is required'),
  }),
  body: z
    .object({
      reason: z.string().trim().max(1000, 'Reason cannot exceed 1000 characters').optional(),
    })
    .optional(),
});

export const deleteCaseSchema = z.object({
  params: z.object({
    id: z.string().min(1, 'Case ID is required'),
  }),
  body: z
    .object({
      reason: z.string().trim().max(1000, 'Reason cannot exceed 1000 characters').optional(),
    })
    .optional(),
});

export type CreateCaseInput = z.infer<typeof createCaseSchema>['body'];
export type UpdateCaseInput = z.infer<typeof updateCaseSchema>['body'];
export type ResolveCaseInput = z.infer<typeof resolveCaseSchema>['body'];
export type ReopenCaseInput = z.infer<typeof reopenCaseSchema>['body'];
export type ArchiveCaseInput = z.infer<typeof archiveCaseSchema>['body'];
export type RestoreCaseInput = z.infer<typeof restoreCaseSchema>['body'];
export type DeleteCaseInput = z.infer<typeof deleteCaseSchema>['body'];
