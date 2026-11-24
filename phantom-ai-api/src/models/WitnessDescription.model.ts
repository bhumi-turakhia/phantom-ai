import { Schema, model, Document, Types } from 'mongoose';

export type Gender = 'male' | 'female' | 'other' | 'unknown';
export type AgeRange =
  | 'child'
  | 'teen'
  | '18-25'
  | '26-35'
  | '36-45'
  | '46-55'
  | '56-65'
  | '65+';
export type SkinTone =
  | 'very_light'
  | 'light'
  | 'medium'
  | 'tan'
  | 'dark'
  | 'very_dark';
export type BuildType = 'slim' | 'average' | 'athletic' | 'heavy' | 'unknown';
export type EvidenceState = 'explicit' | 'uncertain' | 'negated' | 'contradictory' | 'unknown';
export type ReviewStatus = 'pending' | 'in_review' | 'approved' | 'rejected';

export interface IExtractedAttribute {
  category: string;
  attributeName?: string;
  value: string | null;
  originalValue?: string | null;
  sourceText: string;
  start: number | null;
  end: number | null;
  surroundingContext?: string;
  evidenceState: EvidenceState;
  originalEvidenceState?: EvidenceState;
  extractedBy: string;
  confidence?: number | null;
  isCorrected?: boolean;
  correction?: {
    correctedValue: string | null;
    correctedEvidenceState: string;
    correctedBy: Types.ObjectId;
    reason: string;
    timestamp: Date;
    rejected?: boolean;
  };
  gcitStatus?: {
    isEligible: boolean;
    gcitDimension: string | null;
    gcitIndex: number | null;
    conditionValue: number;
    maskValue: number;
    polarity: 'normal' | 'inverted' | 'none';
    isIncluded: boolean;
    inclusionReason: string;
  };
}

export interface IInvestigatorCorrection {
  attributeCategory: string;
  previousValue: string | null;
  correctedValue: string | null;
  previousEvidenceState: string;
  correctedEvidenceState: string;
  correctedBy: Types.ObjectId;
  timestamp: Date;
  reason: string;
  rejected: boolean;
}

export interface IModelMetadata {
  name?: string;
  architecture?: string;
  phase?: string;
  checkpoint?: string;
  checkpointSha256?: string;
}

export interface IFacialFeatures {
  faceShape?: string;
  hairColor?: string;
  hairStyle?: string;
  hairLength?: string;
  eyeColor?: string;
  eyeShape?: string;
  noseShape?: string;
  lipShape?: string;
  eyebrowShape?: string;
  facialHair?: string;
  ears?: string;
  jawline?: string;
  cheekbones?: string;
  forehead?: string;
}

export interface IDistinguishingFeature {
  type: string;
  location: string;
  description: string;
}

export interface IWitnessDescription extends Document {
  _id: Types.ObjectId;
  case: Types.ObjectId;
  witnessName?: string;
  witnessContact?: string;
  isAnonymous: boolean;
  recordedBy: Types.ObjectId;
  rawStatement: string;
  estimatedAge?: AgeRange;
  gender?: Gender;
  ethnicity?: string;
  height?: string;
  build?: BuildType;
  skinTone?: SkinTone;
  facialFeatures?: IFacialFeatures;
  distinguishingFeatures: IDistinguishingFeature[];
  clothingDescription?: string;
  additionalNotes?: string;
  confidenceLevel?: number | null;
  extractedAttributes: IExtractedAttribute[];
  corrections: IInvestigatorCorrection[];
  reviewStatus: ReviewStatus;
  reviewedBy?: Types.ObjectId;
  reviewedAt?: Date;
  modelMetadata?: IModelMetadata;
  createdAt: Date;
  updatedAt: Date;
}

const facialFeaturesSchema = new Schema<IFacialFeatures>(
  {
    faceShape: { type: String, trim: true },
    hairColor: { type: String, trim: true },
    hairStyle: { type: String, trim: true },
    hairLength: { type: String, trim: true },
    eyeColor: { type: String, trim: true },
    eyeShape: { type: String, trim: true },
    noseShape: { type: String, trim: true },
    lipShape: { type: String, trim: true },
    eyebrowShape: { type: String, trim: true },
    facialHair: { type: String, trim: true },
    ears: { type: String, trim: true },
    jawline: { type: String, trim: true },
    cheekbones: { type: String, trim: true },
    forehead: { type: String, trim: true },
  },
  { _id: false }
);

const distinguishingFeatureSchema = new Schema<IDistinguishingFeature>(
  {
    type: { type: String, required: true, trim: true },
    location: { type: String, required: true, trim: true },
    description: { type: String, required: true, trim: true },
  },
  { _id: false }
);

const extractedAttributeSchema = new Schema<IExtractedAttribute>(
  {
    category: { type: String, required: true },
    attributeName: { type: String },
    value: { type: String, default: null },
    originalValue: { type: String, default: null },
    sourceText: { type: String, default: '' },
    start: { type: Number, default: null },
    end: { type: Number, default: null },
    surroundingContext: { type: String, default: '' },
    evidenceState: {
      type: String,
      enum: ['explicit', 'uncertain', 'negated', 'contradictory', 'unknown'],
      default: 'unknown',
    },
    originalEvidenceState: {
      type: String,
      enum: ['explicit', 'uncertain', 'negated', 'contradictory', 'unknown'],
      default: 'unknown',
    },
    extractedBy: { type: String, default: 'bilstm_crf' },
    confidence: { type: Number, default: null, required: false },
    isCorrected: { type: Boolean, default: false },
    correction: {
      correctedValue: { type: String, default: null },
      correctedEvidenceState: { type: String, default: 'explicit' },
      correctedBy: { type: Schema.Types.ObjectId, ref: 'User' },
      reason: { type: String, default: '' },
      timestamp: { type: Date },
      rejected: { type: Boolean, default: false },
    },
    gcitStatus: {
      isEligible: { type: Boolean, default: false },
      gcitDimension: { type: String, default: null },
      gcitIndex: { type: Number, default: null },
      conditionValue: { type: Number, default: 0.0 },
      maskValue: { type: Number, default: 0.0 },
      polarity: { type: String, enum: ['normal', 'inverted', 'none'], default: 'none' },
      isIncluded: { type: Boolean, default: false },
      inclusionReason: { type: String, default: '' },
    },
  },
  { _id: false }
);

const investigatorCorrectionSchema = new Schema<IInvestigatorCorrection>(
  {
    attributeCategory: { type: String, required: true },
    previousValue: { type: String, default: null },
    correctedValue: { type: String, default: null },
    previousEvidenceState: { type: String, default: 'unknown' },
    correctedEvidenceState: { type: String, default: 'explicit' },
    correctedBy: { type: Schema.Types.ObjectId, ref: 'User', required: true },
    timestamp: { type: Date, default: Date.now },
    reason: { type: String, required: true, trim: true },
    rejected: { type: Boolean, default: false },
  },
  { _id: true }
);

const modelMetadataSchema = new Schema<IModelMetadata>(
  {
    name: { type: String },
    architecture: { type: String },
    phase: { type: String },
    checkpoint: { type: String },
    checkpointSha256: { type: String },
  },
  { _id: false }
);

const witnessDescriptionSchema = new Schema<IWitnessDescription>(
  {
    case: {
      type: Schema.Types.ObjectId,
      ref: 'Case',
      required: [true, 'Case reference is required'],
    },
    witnessName: {
      type: String,
      trim: true,
      maxlength: [100, 'Witness name must not exceed 100 characters'],
    },
    witnessContact: {
      type: String,
      trim: true,
    },
    isAnonymous: {
      type: Boolean,
      default: false,
    },
    recordedBy: {
      type: Schema.Types.ObjectId,
      ref: 'User',
      required: [true, 'Recorded by is required'],
    },
    rawStatement: {
      type: String,
      required: [true, 'Witness raw statement is required'],
      trim: true,
      maxlength: [10000, 'Statement must not exceed 10000 characters'],
    },
    estimatedAge: {
      type: String,
      enum: {
        values: ['child', 'teen', '18-25', '26-35', '36-45', '46-55', '56-65', '65+'],
        message: '{VALUE} is not a valid age range',
      },
    },
    gender: {
      type: String,
      enum: {
        values: ['male', 'female', 'other', 'unknown'],
        message: '{VALUE} is not a valid gender',
      },
      default: 'unknown',
    },
    ethnicity: {
      type: String,
      trim: true,
    },
    height: {
      type: String,
      trim: true,
    },
    build: {
      type: String,
      enum: {
        values: ['slim', 'average', 'athletic', 'heavy', 'unknown'],
        message: '{VALUE} is not a valid build type',
      },
      default: 'unknown',
    },
    skinTone: {
      type: String,
      enum: {
        values: ['very_light', 'light', 'medium', 'tan', 'dark', 'very_dark'],
        message: '{VALUE} is not a valid skin tone',
      },
    },
    facialFeatures: {
      type: facialFeaturesSchema,
      default: () => ({}),
    },
    distinguishingFeatures: {
      type: [distinguishingFeatureSchema],
      default: [],
    },
    clothingDescription: {
      type: String,
      trim: true,
      maxlength: [2000, 'Clothing description must not exceed 2000 characters'],
    },
    additionalNotes: {
      type: String,
      trim: true,
      maxlength: [2000, 'Additional notes must not exceed 2000 characters'],
    },
    confidenceLevel: {
      type: Number,
      min: [0, 'Confidence level must be at least 0'],
      max: [100, 'Confidence level must not exceed 100'],
      default: null,
      required: false,
    },
    extractedAttributes: {
      type: [extractedAttributeSchema],
      default: [],
    },
    corrections: {
      type: [investigatorCorrectionSchema],
      default: [],
    },
    reviewStatus: {
      type: String,
      enum: ['pending', 'in_review', 'approved', 'rejected'],
      default: 'pending',
    },
    reviewedBy: {
      type: Schema.Types.ObjectId,
      ref: 'User',
    },
    reviewedAt: {
      type: Date,
    },
    modelMetadata: {
      type: modelMetadataSchema,
      default: null,
    },
  },
  { timestamps: true }
);

witnessDescriptionSchema.index({ case: 1 });
witnessDescriptionSchema.index({ recordedBy: 1 });
witnessDescriptionSchema.index({ createdAt: -1 });

export const WitnessDescription = model<IWitnessDescription>(
  'WitnessDescription',
  witnessDescriptionSchema
);