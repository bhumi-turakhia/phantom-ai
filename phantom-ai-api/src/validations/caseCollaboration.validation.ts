import { z } from 'zod';

const objectIdRegex = /^[a-fA-F0-9]{24}$/;

export const caseIdOnlyParamSchema = z.object({
  params: z.object({
    id: z.string().regex(objectIdRegex, 'Invalid Case ID'),
  }),
});

export const timelineQuerySchema = z.object({
  params: z.object({
    id: z.string().regex(objectIdRegex, 'Invalid Case ID'),
  }),
  query: z.object({
    order: z.enum(['asc', 'desc']).optional(),
    category: z
      .enum([
        'all',
        'case',
        'collaboration',
        'evidence',
        'nlp',
        'composite',
        'comparison',
        'report',
        'comment',
      ])
      .optional(),
    includeDeletedComments: z.enum(['true', 'false']).optional(),
  }),
});

export const addParticipantSchema = z.object({
  params: z.object({
    id: z.string().regex(objectIdRegex, 'Invalid Case ID'),
  }),
  body: z
    .object({
      userId: z
        .string()
        .regex(objectIdRegex, 'Invalid User ID')
        .optional(),
      investigatorId: z.string().trim().min(1).max(60).optional(),
      email: z.string().trim().email('Invalid email address').optional(),
      notes: z.string().trim().max(500, 'Assignment notes cannot exceed 500 characters').optional(),
      permission: z.enum(['read_only', 'editor']).optional(),
    })
    .refine(
      (data) => Boolean(data.userId || data.investigatorId || data.email),
      {
        message: 'At least one identifier (userId, investigatorId, or email) is required to assign a participant',
      }
    ),
});

export const removeParticipantSchema = z.object({
  params: z.object({
    id: z.string().regex(objectIdRegex, 'Invalid Case ID'),
    participantId: z.string().regex(objectIdRegex, 'Invalid Participant User ID'),
  }),
  body: z
    .object({
      reason: z.string().trim().max(500, 'Removal reason cannot exceed 500 characters').optional(),
    })
    .optional(),
});

export const listCommentsQuerySchema = z.object({
  params: z.object({
    id: z.string().regex(objectIdRegex, 'Invalid Case ID'),
  }),
  query: z.object({
    includeDeleted: z.enum(['true', 'false']).optional(),
  }),
});

export const createCommentSchema = z.object({
  params: z.object({
    id: z.string().regex(objectIdRegex, 'Invalid Case ID'),
  }),
  body: z.object({
    content: z
      .string()
      .trim()
      .min(1, 'Comment content cannot be empty')
      .max(4000, 'Comment content cannot exceed 4000 characters'),
    noteType: z
      .enum(['general', 'lead_follow_up', 'supervisory_review', 'quality_assurance'])
      .optional()
      .default('general'),
    linkedResource: z
      .object({
        resourceType: z.enum(['witness_description', 'composite', 'report']),
        resourceId: z.string().regex(objectIdRegex, 'Invalid linked resource ID'),
        label: z.string().trim().max(200).optional(),
      })
      .optional(),
  }),
});

export const updateCommentSchema = z.object({
  params: z.object({
    id: z.string().regex(objectIdRegex, 'Invalid Case ID'),
    commentId: z.string().regex(objectIdRegex, 'Invalid Comment ID'),
  }),
  body: z.object({
    content: z
      .string()
      .trim()
      .min(1, 'Comment content cannot be empty')
      .max(4000, 'Comment content cannot exceed 4000 characters'),
    noteType: z
      .enum(['general', 'lead_follow_up', 'supervisory_review', 'quality_assurance'])
      .optional(),
    editReason: z
      .string()
      .trim()
      .max(500, 'Edit reason cannot exceed 500 characters')
      .optional(),
  }),
});

export const deleteCommentSchema = z.object({
  params: z.object({
    id: z.string().regex(objectIdRegex, 'Invalid Case ID'),
    commentId: z.string().regex(objectIdRegex, 'Invalid Comment ID'),
  }),
  body: z
    .object({
      deleteReason: z
        .string()
        .trim()
        .max(500, 'Delete reason cannot exceed 500 characters')
        .optional(),
    })
    .optional(),
});

export type AddParticipantInput = z.infer<typeof addParticipantSchema>['body'];
export type CreateCommentInput = z.infer<typeof createCommentSchema>['body'];
export type UpdateCommentInput = z.infer<typeof updateCommentSchema>['body'];

