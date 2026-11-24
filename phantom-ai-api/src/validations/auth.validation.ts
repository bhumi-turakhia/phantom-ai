import { z } from 'zod';

export const registerSchema = z.object({
  body: z.object({
    name: z
      .string({ required_error: 'Name is required' })
      .min(2, 'Name must be at least 2 characters')
      .max(100, 'Name must not exceed 100 characters')
      .trim(),
    email: z
      .string({ required_error: 'Email is required' })
      .email('Please provide a valid email address')
      .toLowerCase()
      .trim(),
    password: z
      .string({ required_error: 'Passcode is required' })
      .min(8, 'Passcode must be at least 8 characters')
      .regex(
        /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/,
        'Passcode must contain at least one uppercase letter, one lowercase letter, and one number'
      ),
    role: z
      .enum(['admin', 'investigator', 'supervisor', 'analyst'], {
        errorMap: () => ({ message: 'Role must be admin, investigator, supervisor, or analyst' }),
      })
      .optional(),
    badgeNumber: z.string().trim().optional(),
    department: z.string().trim().optional(),
  }),
});

export const loginSchema = z.object({
  body: z
    .object({
      identifier: z.string().trim().optional(),
      email: z.string().trim().optional(),
      password: z.string({ required_error: 'Passcode is required' }).min(1, 'Passcode is required'),
    })
    .refine((data) => data.identifier || data.email, {
      message: 'Investigator ID or Email is required',
      path: ['identifier'],
    }),
});

export const passkeyVerifyLoginSchema = z.object({
  body: z.object({
    preAuthToken: z.string({ required_error: 'Pre-auth token is required' }),
    response: z.record(z.any(), { required_error: 'WebAuthn assertion response is required' }),
  }),
});

export const passkeyVerifyRegisterSchema = z.object({
  body: z.object({
    response: z.record(z.any(), { required_error: 'WebAuthn registration response is required' }),
    deviceName: z.string().trim().max(100).optional(),
  }),
});

export const recoverySchema = z.object({
  body: z
    .object({
      identifier: z.string().trim().optional(),
      email: z.string().trim().optional(),
      password: z.string({ required_error: 'Passcode is required' }).min(1),
      recoveryCode: z
        .string({ required_error: 'Recovery code is required' })
        .min(8, 'Recovery code must be valid')
        .trim(),
    })
    .refine((data) => data.identifier || data.email, {
      message: 'Investigator ID or Email is required',
      path: ['identifier'],
    }),
});

export const removePasskeySchema = z.object({
  body: z.object({
    password: z.string({ required_error: 'Passcode confirmation is required' }),
  }),
  params: z.object({
    credentialId: z.string({ required_error: 'Credential ID is required' }),
  }),
});

export const refreshTokenSchema = z.object({
  body: z.object({
    refreshToken: z.string().min(1).optional(),
  }),
});

export const updateProfileSchema = z.object({
  body: z.object({
    name: z.string().trim().min(2, 'Name must be at least 2 characters').max(100).optional(),
    badgeNumber: z.string().trim().max(50).optional(),
    department: z.string().trim().max(100).optional(),
    currentPassword: z.string().optional(),
    newPassword: z
      .string()
      .min(8, 'New password must be at least 8 characters')
      .regex(
        /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/,
        'Password must contain at least one uppercase letter, one lowercase letter, and one number'
      )
      .optional(),
  }),
});

export type RegisterInput = z.infer<typeof registerSchema>['body'];
export type LoginInput = z.infer<typeof loginSchema>['body'];
export type RecoveryInput = z.infer<typeof recoverySchema>['body'];
export type UpdateProfileInput = z.infer<typeof updateProfileSchema>['body'];
