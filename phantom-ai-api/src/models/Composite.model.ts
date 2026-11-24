import { Schema, model, Document, Types } from 'mongoose';

export type CompositeStatus = 'draft' | 'under_review' | 'approved' | 'superseded';

export interface ICompositeVariation {
  _id?: Types.ObjectId;
  variationIndex: number;
  seed: number;
  storageKey: string;
  imageUrl: string;
  dataUrl?: string;
  mimeType: string;
  fileSizeBytes: number;
  width: number;
  height: number;
  isSelected: boolean;
  approvedAt?: Date;
  approvedBy?: Types.ObjectId;
  selectionNotes?: string;
  variationRole?: 'standard' | 'semantic_edited' | 'semantic_baseline';
  gcitVector?: number[];
  gcitMask?: number[];
  sha256?: string;
}

export interface ICompositeSelectionHistoryItem {
  variationIndex: number;
  variationId?: Types.ObjectId;
  seed: number;
  previousVariationIndex: number | null;
  selectedBy: Types.ObjectId;
  selectedAt: Date;
  status: CompositeStatus;
  notes?: string;
}

export interface IGeneratorMetadata {
  modelName: string;
  architecture: string;
  phase: string;
  checkpointPath: string;
  checkpointSha256: string;
  bestEpoch: number;
  parameterCount: number;
  device: string;
}

export interface IComposite extends Document {
  _id: Types.ObjectId;
  case: Types.ObjectId;
  witnessDescription: Types.ObjectId;
  version: number;
  status: CompositeStatus;
  generatorMetadata: IGeneratorMetadata;
  approvedEvidenceSnapshot: Record<string, any>;
  evidenceTrace?: Array<Record<string, any>>;
  gcitVector?: number[];
  gcitMask?: number[];
  imageHashes?: string[];
  conditionedAttributes: string[];
  unconditionedAttributes: string[];
  warnings: string[];
  variations: ICompositeVariation[];
  selectionHistory?: ICompositeSelectionHistoryItem[];
  parentComposite?: Types.ObjectId;
  editingType?: 'initial' | 'conditioning_traversal' | 'latent_traversal' | 'hybrid_traversal';
  traversalParameters?: {
    continuousAdjustments?: Record<string, number>;
    latentTraversals?: Array<{ attribute: string; alpha: number }>;
  };
  createdBy: Types.ObjectId;
  reviewedBy?: Types.ObjectId;
  reviewedAt?: Date;
  notes?: string;
  createdAt: Date;
  updatedAt: Date;
}

const compositeVariationSchema = new Schema<ICompositeVariation>(
  {
    variationIndex: { type: Number, required: true },
    seed: { type: Number, required: true },
    storageKey: { type: String, required: true, trim: true },
    imageUrl: { type: String, required: true, trim: true },
    dataUrl: { type: String },
    mimeType: { type: String, default: 'image/png' },
    fileSizeBytes: { type: Number, required: true },
    width: { type: Number, default: 64 },
    height: { type: Number, default: 64 },
    isSelected: { type: Boolean, default: false },
    approvedAt: { type: Date },
    approvedBy: { type: Schema.Types.ObjectId, ref: 'User' },
    selectionNotes: { type: String, trim: true },
    variationRole: {
      type: String,
      enum: ['standard', 'semantic_edited', 'semantic_baseline'],
      default: 'standard',
    },
    gcitVector: { type: [Number], default: undefined },
    gcitMask: { type: [Number], default: undefined },
    sha256: { type: String },
  },
  { _id: true }
);

const compositeSelectionHistorySchema = new Schema<ICompositeSelectionHistoryItem>(
  {
    variationIndex: { type: Number, required: true },
    variationId: { type: Schema.Types.ObjectId },
    seed: { type: Number, required: true },
    previousVariationIndex: { type: Number, default: null },
    selectedBy: { type: Schema.Types.ObjectId, ref: 'User', required: true },
    selectedAt: { type: Date, default: Date.now },
    status: {
      type: String,
      enum: ['draft', 'under_review', 'approved', 'superseded'],
      default: 'approved',
    },
    notes: { type: String, trim: true },
  },
  { _id: true }
);

const generatorMetadataSchema = new Schema<IGeneratorMetadata>(
  {
    modelName: { type: String, required: true },
    architecture: { type: String, required: true },
    phase: { type: String, default: 'Phase 5.4' },
    checkpointPath: { type: String, required: true },
    checkpointSha256: { type: String, required: true },
    bestEpoch: { type: Number, default: 4 },
    parameterCount: { type: Number, default: 4102336 },
    device: { type: String, default: 'mps' },
  },
  { _id: false }
);

const compositeSchema = new Schema<IComposite>(
  {
    case: {
      type: Schema.Types.ObjectId,
      ref: 'Case',
      required: [true, 'Case reference is required'],
      index: true,
    },
    witnessDescription: {
      type: Schema.Types.ObjectId,
      ref: 'WitnessDescription',
      required: [true, 'Witness description reference is required'],
      index: true,
    },
    version: {
      type: Number,
      required: [true, 'Version number is required'],
      default: 1,
    },
    status: {
      type: String,
      enum: {
        values: ['draft', 'under_review', 'approved', 'superseded'],
        message: '{VALUE} is not a valid composite status',
      },
      default: 'draft',
      index: true,
    },
    generatorMetadata: {
      type: generatorMetadataSchema,
      required: true,
    },
    approvedEvidenceSnapshot: {
      type: Schema.Types.Mixed,
      required: true,
    },
    evidenceTrace: {
      type: [Schema.Types.Mixed],
      default: [],
    },
    gcitVector: {
      type: [Number],
      default: [],
    },
    gcitMask: {
      type: [Number],
      default: [],
    },
    imageHashes: {
      type: [String],
      default: [],
    },
    conditionedAttributes: {
      type: [String],
      default: [],
    },
    unconditionedAttributes: {
      type: [String],
      default: [],
    },
    warnings: {
      type: [String],
      default: [],
    },
    variations: {
      type: [compositeVariationSchema],
      default: [],
    },
    selectionHistory: {
      type: [compositeSelectionHistorySchema],
      default: [],
    },
    parentComposite: {
      type: Schema.Types.ObjectId,
      ref: 'Composite',
      default: null,
    },
    editingType: {
      type: String,
      enum: ['initial', 'conditioning_traversal', 'latent_traversal', 'hybrid_traversal'],
      default: 'initial',
    },
    traversalParameters: {
      type: Schema.Types.Mixed,
      default: null,
    },
    createdBy: {
      type: Schema.Types.ObjectId,
      ref: 'User',
      required: [true, 'Created by is required'],
    },
    reviewedBy: {
      type: Schema.Types.ObjectId,
      ref: 'User',
    },
    reviewedAt: {
      type: Date,
    },
    notes: {
      type: String,
      trim: true,
      maxlength: [2000, 'Notes must not exceed 2000 characters'],
    },
  },
  {
    timestamps: true,
  }
);

// Unique index compound on case and version to guarantee version immutability
compositeSchema.index({ case: 1, version: 1 }, { unique: true });

export const Composite = model<IComposite>('Composite', compositeSchema);

