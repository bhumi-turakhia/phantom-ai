import { Schema, model, Document, Types } from 'mongoose';

export type CaseStatus =
  | 'open'
  | 'in_progress'
  | 'under_review'
  | 'resolved'
  | 'archived'
  | 'closed';

export type CasePriority = 'low' | 'medium' | 'high' | 'critical';

export type CollaboratorPermission = 'read_only' | 'editor';

export interface ICaseLocation {
  address?: string;
  city?: string;
  state?: string;
  coordinates?: {
    lat: number;
    lng: number;
  };
}

export interface ICaseCollaborator {
  user: Types.ObjectId;
  permission: CollaboratorPermission;
  addedAt: Date;
  addedBy: Types.ObjectId;
}

export interface ICase extends Document {
  _id: Types.ObjectId;
  caseNumber: string;
  title: string;
  description?: string;
  status: CaseStatus;
  priority: CasePriority;
  createdBy: Types.ObjectId;
  assignedTo: Types.ObjectId[];
  collaborators: ICaseCollaborator[];
  location?: ICaseLocation;
  incidentDate?: Date;
  tags: string[];
  isDeleted: boolean;

  // Lifecycle fields
  resolvedBy?: Types.ObjectId;
  resolvedAt?: Date;
  resolutionSummary?: string;
  resolutionReason?: string;

  reopenedBy?: Types.ObjectId;
  reopenedAt?: Date;
  reopenReason?: string;

  archivedBy?: Types.ObjectId;
  archivedAt?: Date;
  archiveReason?: string;

  restoredBy?: Types.ObjectId;
  restoredAt?: Date;
  restoreReason?: string;

  deletedBy?: Types.ObjectId;
  deletedAt?: Date;
  deletionReason?: string;

  createdAt: Date;
  updatedAt: Date;
}

const caseLocationSchema = new Schema<ICaseLocation>(
  {
    address: { type: String, trim: true },
    city: { type: String, trim: true },
    state: { type: String, trim: true },
    coordinates: {
      lat: { type: Number, min: -90, max: 90 },
      lng: { type: Number, min: -180, max: 180 },
    },
  },
  { _id: false }
);

const caseCollaboratorSchema = new Schema<ICaseCollaborator>(
  {
    user: {
      type: Schema.Types.ObjectId,
      ref: 'User',
      required: true,
    },
    permission: {
      type: String,
      enum: ['read_only', 'editor'],
      default: 'editor',
    },
    addedAt: {
      type: Date,
      default: Date.now,
    },
    addedBy: {
      type: Schema.Types.ObjectId,
      ref: 'User',
      required: true,
    },
  },
  { _id: false }
);

const caseSchema = new Schema<ICase>(
  {
    caseNumber: {
      type: String,
      required: [true, 'Case number is required'],
      unique: true,
      trim: true,
      uppercase: true,
    },
    title: {
      type: String,
      required: [true, 'Case title is required'],
      trim: true,
      maxlength: [200, 'Title must not exceed 200 characters'],
    },
    description: {
      type: String,
      trim: true,
      maxlength: [5000, 'Description must not exceed 5000 characters'],
    },
    status: {
      type: String,
      enum: {
        values: ['open', 'in_progress', 'under_review', 'resolved', 'archived', 'closed'],
        message: '{VALUE} is not a valid status',
      },
      default: 'open',
    },
    priority: {
      type: String,
      enum: {
        values: ['low', 'medium', 'high', 'critical'],
        message: '{VALUE} is not a valid priority',
      },
      default: 'medium',
    },
    createdBy: {
      type: Schema.Types.ObjectId,
      ref: 'User',
      required: [true, 'Created by is required'],
    },
    assignedTo: [
      {
        type: Schema.Types.ObjectId,
        ref: 'User',
      },
    ],
    collaborators: [caseCollaboratorSchema],
    location: {
      type: caseLocationSchema,
    },
    incidentDate: {
      type: Date,
    },
    tags: [
      {
        type: String,
        trim: true,
        lowercase: true,
      },
    ],
    isDeleted: {
      type: Boolean,
      default: false,
      select: false,
    },

    // Lifecycle audit tracking
    resolvedBy: {
      type: Schema.Types.ObjectId,
      ref: 'User',
    },
    resolvedAt: {
      type: Date,
    },
    resolutionSummary: {
      type: String,
      trim: true,
      maxlength: [2000, 'Resolution summary must not exceed 2000 characters'],
    },
    resolutionReason: {
      type: String,
      trim: true,
      maxlength: [1000, 'Resolution reason must not exceed 1000 characters'],
    },

    reopenedBy: {
      type: Schema.Types.ObjectId,
      ref: 'User',
    },
    reopenedAt: {
      type: Date,
    },
    reopenReason: {
      type: String,
      trim: true,
      maxlength: [1000, 'Reopen reason must not exceed 1000 characters'],
    },

    archivedBy: {
      type: Schema.Types.ObjectId,
      ref: 'User',
    },
    archivedAt: {
      type: Date,
    },
    archiveReason: {
      type: String,
      trim: true,
      maxlength: [1000, 'Archive reason must not exceed 1000 characters'],
    },

    restoredBy: {
      type: Schema.Types.ObjectId,
      ref: 'User',
    },
    restoredAt: {
      type: Date,
    },
    restoreReason: {
      type: String,
      trim: true,
      maxlength: [1000, 'Restore reason must not exceed 1000 characters'],
    },

    deletedBy: {
      type: Schema.Types.ObjectId,
      ref: 'User',
    },
    deletedAt: {
      type: Date,
    },
    deletionReason: {
      type: String,
      trim: true,
      maxlength: [1000, 'Deletion reason must not exceed 1000 characters'],
    },
  },
  { timestamps: true }
);

caseSchema.index({ status: 1, priority: 1 });
caseSchema.index({ createdBy: 1 });
caseSchema.index({ assignedTo: 1 });
caseSchema.index({ 'collaborators.user': 1 });
caseSchema.index({ isDeleted: 1 });
caseSchema.index({ status: 1, isDeleted: 1 });
caseSchema.index({ title: 'text', description: 'text' });

export const Case = model<ICase>('Case', caseSchema);
