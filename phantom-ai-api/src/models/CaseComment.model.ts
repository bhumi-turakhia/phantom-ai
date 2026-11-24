import { Schema, model, Document, Types } from 'mongoose';

export type CaseCommentNoteType =
  | 'general'
  | 'lead_follow_up'
  | 'supervisory_review'
  | 'quality_assurance';

export type LinkedResourceType =
  | 'witness_description'
  | 'composite'
  | 'report';

export interface ILinkedResource {
  resourceType: LinkedResourceType;
  resourceId: Types.ObjectId;
  label?: string;
}

export interface ICaseCommentEditEntry {
  previousContent: string;
  previousNoteType?: CaseCommentNoteType;
  editedBy: Types.ObjectId;
  editedAt: Date;
  editReason?: string;
}

export interface ICaseComment extends Document {
  _id: Types.ObjectId;
  case: Types.ObjectId;
  author: Types.ObjectId;
  content: string;
  noteType: CaseCommentNoteType;
  linkedResource?: ILinkedResource;
  isEdited: boolean;
  editedAt?: Date;
  editHistory: ICaseCommentEditEntry[];
  isDeleted: boolean;
  deletedAt?: Date;
  deletedBy?: Types.ObjectId;
  deleteReason?: string;
  createdAt: Date;
  updatedAt: Date;
}

const linkedResourceSchema = new Schema<ILinkedResource>(
  {
    resourceType: {
      type: String,
      enum: ['witness_description', 'composite', 'report'],
      required: true,
    },
    resourceId: {
      type: Schema.Types.ObjectId,
      required: true,
    },
    label: {
      type: String,
      trim: true,
      maxlength: [200, 'Linked resource label cannot exceed 200 characters'],
    },
  },
  { _id: false }
);

const editHistorySchema = new Schema<ICaseCommentEditEntry>(
  {
    previousContent: {
      type: String,
      required: true,
    },
    previousNoteType: {
      type: String,
      enum: ['general', 'lead_follow_up', 'supervisory_review', 'quality_assurance'],
    },
    editedBy: {
      type: Schema.Types.ObjectId,
      ref: 'User',
      required: true,
    },
    editedAt: {
      type: Date,
      default: Date.now,
    },
    editReason: {
      type: String,
      trim: true,
      maxlength: [500, 'Edit reason cannot exceed 500 characters'],
    },
  },
  { _id: false }
);

const caseCommentSchema = new Schema<ICaseComment>(
  {
    case: {
      type: Schema.Types.ObjectId,
      ref: 'Case',
      required: [true, 'Case reference is required'],
      index: true,
    },
    author: {
      type: Schema.Types.ObjectId,
      ref: 'User',
      required: [true, 'Comment author is required'],
      index: true,
    },
    content: {
      type: String,
      required: [true, 'Comment content is required'],
      trim: true,
      minlength: [1, 'Comment content cannot be empty'],
      maxlength: [4000, 'Comment content cannot exceed 4000 characters'],
    },
    noteType: {
      type: String,
      enum: {
        values: ['general', 'lead_follow_up', 'supervisory_review', 'quality_assurance'],
        message:
          'Note type must be general, lead_follow_up, supervisory_review, or quality_assurance',
      },
      default: 'general',
    },
    linkedResource: {
      type: linkedResourceSchema,
      default: undefined,
    },
    isEdited: {
      type: Boolean,
      default: false,
    },
    editedAt: {
      type: Date,
    },
    editHistory: {
      type: [editHistorySchema],
      default: [],
    },
    isDeleted: {
      type: Boolean,
      default: false,
      index: true,
    },
    deletedAt: {
      type: Date,
    },
    deletedBy: {
      type: Schema.Types.ObjectId,
      ref: 'User',
    },
    deleteReason: {
      type: String,
      trim: true,
      maxlength: [500, 'Delete reason cannot exceed 500 characters'],
    },
  },
  {
    timestamps: true,
  }
);

caseCommentSchema.index({ case: 1, createdAt: 1 });
caseCommentSchema.index({ case: 1, isDeleted: 1, createdAt: 1 });

export const CaseComment = model<ICaseComment>('CaseComment', caseCommentSchema);

