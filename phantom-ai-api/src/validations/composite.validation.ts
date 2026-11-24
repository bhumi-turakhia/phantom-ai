import { z } from 'zod';

export const generateCompositeSchema = z.object({
  params: z.object({
    caseId: z.string({ required_error: 'Case ID is required' }).min(1),
  }),
  body: z.object({
    witnessDescriptionId: z
      .string({ required_error: 'Witness description ID is required' })
      .min(1, 'Witness description ID must not be empty'),
    seed: z.number().int().optional(),
    numVariations: z.number().int().min(1).max(4).default(1),
    notes: z.string().max(2000).optional(),
  }),
});

export const caseIdParamsSchema = z.object({
  params: z.object({
    caseId: z.string({ required_error: 'Case ID is required' }).min(1),
  }),
});

export const compositeIdParamsSchema = z.object({
  params: z.object({
    caseId: z.string({ required_error: 'Case ID is required' }).min(1),
    compositeId: z.string({ required_error: 'Composite ID is required' }).min(1),
  }),
});

export const compositeImageParamsSchema = z.object({
  params: z.object({
    caseId: z.string({ required_error: 'Case ID is required' }).min(1),
    compositeId: z.string({ required_error: 'Composite ID is required' }).min(1),
    variationIndex: z.string({ required_error: 'Variation index is required' }).min(1),
  }),
});

export const selectVariationSchema = z.object({
  params: z.object({
    caseId: z.string({ required_error: 'Case ID is required' }).min(1),
    compositeId: z.string({ required_error: 'Composite ID is required' }).min(1),
  }),
  body: z.object({
    variationIndex: z.number().int().min(1),
    status: z.enum(['draft', 'under_review', 'approved', 'superseded']).default('approved'),
    notes: z.string().max(2000).optional(),
  }),
});

export const compareSingleCompositeSchema = z.object({
  params: z.object({
    caseId: z.string({ required_error: 'Case ID is required' }).min(1),
    compositeId: z.string({ required_error: 'Composite ID is required' }).min(1),
  }),
  query: z
    .object({
      variations: z.string().optional(),
    })
    .optional(),
});

export const compareCaseVariationsSchema = z.object({
  params: z.object({
    caseId: z.string({ required_error: 'Case ID is required' }).min(1),
  }),
  body: z
    .object({
      compositeId: z.string().min(1).optional(),
      variationIndices: z.array(z.number().int().min(1)).optional(),
      targets: z
        .array(
          z.object({
            compositeId: z.string().min(1),
            variationIndex: z.number().int().min(1),
          })
        )
        .optional(),
    })
    .refine(
      (data) =>
        (Array.isArray(data.targets) && data.targets.length >= 1) ||
        (typeof data.compositeId === 'string' && data.compositeId.trim().length > 0),
      {
        message: 'Either compositeId or targets array must be provided for variation comparison.',
        path: ['targets'],
      }
    ),
});

export type GenerateCompositeBodyInput = z.infer<typeof generateCompositeSchema>['body'];

export const editCompositeSchema = z.object({
  params: z.object({
    caseId: z.string({ required_error: 'Case ID is required' }).min(1),
    compositeId: z.string({ required_error: 'Composite ID is required' }).min(1),
  }),
  body: z
    .object({
      seed: z.number().int().optional(),
      selectedSeed: z.number().int().optional(),
      continuousAdjustments: z.record(z.string(), z.number().min(-1.0).max(1.0)).optional().default({}),
      latentTraversals: z
        .union([
          z.array(
            z.object({
              attribute: z.string().min(1),
              alpha: z.number().min(-3.0).max(3.0),
            })
          ),
          z.record(z.string(), z.number().min(-3.0).max(3.0)),
        ])
        .optional()
        .default([]),
      notes: z.string().max(2000).optional(),
    })
    .refine((data) => data.seed !== undefined || data.selectedSeed !== undefined, {
      message: 'Seed is required (either seed or selectedSeed)',
      path: ['seed'],
    }),
});

export type EditCompositeBodyInput = z.infer<typeof editCompositeSchema>['body'];
export type SelectVariationBodyInput = z.infer<typeof selectVariationSchema>['body'];

