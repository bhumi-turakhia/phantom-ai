import { z } from 'zod';

const facialFeaturesSchema = z.object({
  faceShape: z.string().trim().optional(),
  hairColor: z.string().trim().optional(),
  hairStyle: z.string().trim().optional(),
  hairLength: z.string().trim().optional(),
  eyeColor: z.string().trim().optional(),
  eyeShape: z.string().trim().optional(),
  noseShape: z.string().trim().optional(),
  lipShape: z.string().trim().optional(),
  eyebrowShape: z.string().trim().optional(),
  facialHair: z.string().trim().optional(),
  ears: z.string().trim().optional(),
  jawline: z.string().trim().optional(),
  cheekbones: z.string().trim().optional(),
  forehead: z.string().trim().optional(),
});

const distinguishingFeatureSchema = z.object({
  type: z
    .string({ required_error: 'Distinguishing feature type is required' })
    .trim()
    .min(1, 'Type is required'),
  location: z
    .string({ required_error: 'Distinguishing feature location is required' })
    .trim()
    .min(1, 'Location is required'),
  description: z
    .string({ required_error: 'Distinguishing feature description is required' })
    .trim()
    .min(1, 'Description is required'),
});

export const createWitnessDescriptionSchema = z.object({
  body: z.object({
    case: z
      .string({ required_error: 'Case ID is required' })
      .trim()
      .min(1, 'Case ID is required'),
    witnessName: z
      .string()
      .trim()
      .max(100, 'Witness name must not exceed 100 characters')
      .optional(),
    witnessContact: z
      .string()
      .trim()
      .optional(),
    isAnonymous: z
      .boolean()
      .optional()
      .default(false),
    rawStatement: z
      .string({ required_error: 'Raw witness statement is required' })
      .trim()
      .min(1, 'Raw witness statement is required')
      .max(10000, 'Statement must not exceed 10000 characters'),
    estimatedAge: z
      .enum(['child', 'teen', '18-25', '26-35', '36-45', '46-55', '56-65', '65+'], {
        errorMap: () => ({ message: 'Invalid age range' }),
      })
      .optional(),
    gender: z
      .enum(['male', 'female', 'other', 'unknown'], {
        errorMap: () => ({ message: 'Gender must be male, female, other, or unknown' }),
      })
      .optional(),
    ethnicity: z
      .string()
      .trim()
      .optional(),
    height: z
      .string()
      .trim()
      .optional(),
    build: z
      .enum(['slim', 'average', 'athletic', 'heavy', 'unknown'], {
        errorMap: () => ({ message: 'Build must be slim, average, athletic, heavy, or unknown' }),
      })
      .optional(),
    skinTone: z
      .enum(['very_light', 'light', 'medium', 'tan', 'dark', 'very_dark'], {
        errorMap: () => ({ message: 'Invalid skin tone value' }),
      })
      .optional(),
    facialFeatures: facialFeaturesSchema.optional(),
    distinguishingFeatures: z
      .array(distinguishingFeatureSchema)
      .optional()
      .default([]),
    clothingDescription: z
      .string()
      .trim()
      .max(2000, 'Clothing description must not exceed 2000 characters')
      .optional(),
    additionalNotes: z
      .string()
      .trim()
      .max(2000, 'Additional notes must not exceed 2000 characters')
      .optional(),
    confidenceLevel: z
      .number()
      .min(0, 'Confidence level must be at least 0')
      .max(100, 'Confidence level must not exceed 100')
      .optional()
      .default(50),
  }),
});

export const updateWitnessDescriptionSchema = z.object({
  body: z.object({
    witnessName: z
      .string()
      .trim()
      .max(100, 'Witness name must not exceed 100 characters')
      .optional(),
    witnessContact: z
      .string()
      .trim()
      .optional(),
    isAnonymous: z
      .boolean()
      .optional(),
    rawStatement: z
      .string()
      .trim()
      .min(1, 'Raw witness statement cannot be empty')
      .max(10000, 'Statement must not exceed 10000 characters')
      .optional(),
    estimatedAge: z
      .enum(['child', 'teen', '18-25', '26-35', '36-45', '46-55', '56-65', '65+'], {
        errorMap: () => ({ message: 'Invalid age range' }),
      })
      .optional(),
    gender: z
      .enum(['male', 'female', 'other', 'unknown'], {
        errorMap: () => ({ message: 'Gender must be male, female, other, or unknown' }),
      })
      .optional(),
    ethnicity: z
      .string()
      .trim()
      .optional(),
    height: z
      .string()
      .trim()
      .optional(),
    build: z
      .enum(['slim', 'average', 'athletic', 'heavy', 'unknown'], {
        errorMap: () => ({ message: 'Build must be slim, average, athletic, heavy, or unknown' }),
      })
      .optional(),
    skinTone: z
      .enum(['very_light', 'light', 'medium', 'tan', 'dark', 'very_dark'], {
        errorMap: () => ({ message: 'Invalid skin tone value' }),
      })
      .optional(),
    facialFeatures: facialFeaturesSchema.optional(),
    distinguishingFeatures: z
      .array(distinguishingFeatureSchema)
      .optional(),
    clothingDescription: z
      .string()
      .trim()
      .max(2000, 'Clothing description must not exceed 2000 characters')
      .optional(),
    additionalNotes: z
      .string()
      .trim()
      .max(2000, 'Additional notes must not exceed 2000 characters')
      .optional(),
    confidenceLevel: z
      .number()
      .min(0, 'Confidence level must be at least 0')
      .max(100, 'Confidence level must not exceed 100')
      .optional(),
  }),
  params: z.object({
    id: z.string().min(1, 'Witness description ID is required'),
  }),
});

export const witnessDescriptionIdParamSchema = z.object({
  params: z.object({
    id: z.string().min(1, 'Witness description ID is required'),
  }),
});

export const listByCaseParamSchema = z.object({
  params: z.object({
    caseId: z.string().min(1, 'Case ID is required'),
  }),
});

export type CreateWitnessDescriptionInput = z.infer<typeof createWitnessDescriptionSchema>['body'];
export type UpdateWitnessDescriptionInput = z.infer<typeof updateWitnessDescriptionSchema>['body'];