import { Schema, model, Document, Types } from 'mongoose';

export type SecurityEventType =
  | 'LOGIN_PASSCODE_SUCCESS'
  | 'LOGIN_PASSCODE_FAILURE'
  | 'LOGIN_PASSKEY_SUCCESS'
  | 'LOGIN_PASSKEY_FAILURE'
  | 'LOGIN_LOCKED_OUT'
  | 'PASSKEY_REGISTER_INITIATED'
  | 'PASSKEY_REGISTER_SUCCESS'
  | 'PASSKEY_DELETED'
  | 'RECOVERY_CODE_USED'
  | 'RECOVERY_CODE_FAILED'
  | 'LOGOUT'
  | 'TOKEN_REFRESH'
  | 'SESSION_REVOKED'
  | 'ADMIN_RECOVERY_TRIGGERED'
  | 'UNAUTHORIZED_ACCESS_ATTEMPT'
  | 'COMPOSITE_GENERATION_INITIATED'
  | 'COMPOSITE_GENERATED_SUCCESS'
  | 'COMPOSITE_GENERATION_FAILURE'
  | 'COMPOSITE_VIEWED'
  | 'COMPOSITE_APPROVED'
  | 'COMPOSITE_VARIATION_SELECTED'
  | 'COMPOSITE_VARIATIONS_COMPARED'
  | 'COMPOSITE_REVISION_CREATED'
  | 'COMPOSITE_SEMANTIC_EDIT'
  | 'REPORT_GENERATED_SUCCESS'
  | 'REPORT_GENERATION_FAILURE'
  | 'REPORT_DOWNLOADED'
  | 'CASE_CREATED'
  | 'CASE_UPDATED'
  | 'CASE_PARTICIPANT_ADDED'
  | 'CASE_PARTICIPANT_REMOVED'
  | 'CASE_COMMENT_ADDED'
  | 'CASE_COMMENT_EDITED'
  | 'CASE_COMMENT_DELETED'
  | 'WITNESS_STATEMENT_CREATED'
  | 'WITNESS_NLP_ANALYZED'
  | 'WITNESS_EVIDENCE_CORRECTED'
  | 'WITNESS_EVIDENCE_APPROVED';

export interface ISecurityAudit extends Document {
  _id: Types.ObjectId;
  userId?: Types.ObjectId;
  investigatorId?: string;
  email?: string;
  event: SecurityEventType;
  status: 'success' | 'failure' | 'warning';
  ipAddress?: string;
  userAgent?: string;
  details?: Record<string, unknown>;
  createdAt: Date;
}

const securityAuditSchema = new Schema<ISecurityAudit>(
  {
    userId: {
      type: Schema.Types.ObjectId,
      ref: 'User',
      index: true,
    },
    investigatorId: {
      type: String,
      index: true,
    },
    email: {
      type: String,
      index: true,
    },
    event: {
      type: String,
      required: true,
      index: true,
    },
    status: {
      type: String,
      enum: ['success', 'failure', 'warning'],
      required: true,
    },
    ipAddress: {
      type: String,
      trim: true,
    },
    userAgent: {
      type: String,
      trim: true,
    },
    details: {
      type: Schema.Types.Mixed,
    },
  },
  {
    timestamps: { createdAt: true, updatedAt: false },
  }
);

securityAuditSchema.index({ createdAt: -1 });

export const SecurityAudit = model<ISecurityAudit>('SecurityAudit', securityAuditSchema);

