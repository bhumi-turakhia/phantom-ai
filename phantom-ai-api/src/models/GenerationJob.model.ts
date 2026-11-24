import { Schema, model, Document, Types } from 'mongoose';

export type JobStatus =
  | 'pending'
  | 'processing_prompt'
  | 'generating_image'
  | 'completed'
  | 'failed'
  | 'cancelled';

export type GenerationProvider = 'stable-diffusion-xl' | 'claude';

export interface IGenerationParams {
  steps?: number;
  cfgScale?: number;
  seed?: number;
  sampler?: string;
  width?: number;
  height?: number;
  numImages?: number;
}

export interface IGeneratedImage {
  imageUrl: string;
  storageKey: string;
  width: number;
  height: number;
  seed?: number;
  isSelected: boolean;
  generatedAt: Date;
}

export interface IGenerationJob extends Document {
  _id: Types.ObjectId;
  case: Types.ObjectId;
  witnessDescription: Types.ObjectId;
  requestedBy: Types.ObjectId;
  status: JobStatus;
  generatedPrompt?: string;
  promptModel?: string;
  imageModel: GenerationProvider;
  generationParams?: IGenerationParams;
  images: IGeneratedImage[];
  errorMessage?: string;
  retryCount: number;
  startedAt?: Date;
  completedAt?: Date;
  createdAt: Date;
  updatedAt: Date;
}

const generationParamsSchema = new Schema<IGenerationParams>(
  {
    steps: { type: Number, min: 1, max: 150 },
    cfgScale: { type: Number, min: 1, max: 30 },
    seed: { type: Number },
    sampler: { type: String, trim: true },
    width: { type: Number, min: 256, max: 2048 },
    height: { type: Number, min: 256, max: 2048 },
    numImages: { type: Number, min: 1, max: 4 },
  },
  { _id: false }
);

const generatedImageSchema = new Schema<IGeneratedImage>(
  {
    imageUrl: { type: String, required: true, trim: true },
    storageKey: { type: String, required: true, trim: true },
    width: { type: Number, required: true },
    height: { type: Number, required: true },
    seed: { type: Number },
    isSelected: { type: Boolean, default: false },
    generatedAt: { type: Date, default: Date.now },
  },
  { _id: true }
);

const generationJobSchema = new Schema<IGenerationJob>(
  {
    case: {
      type: Schema.Types.ObjectId,
      ref: 'Case',
      required: [true, 'Case reference is required'],
    },
    witnessDescription: {
      type: Schema.Types.ObjectId,
      ref: 'WitnessDescription',
      required: [true, 'Witness description reference is required'],
    },
    requestedBy: {
      type: Schema.Types.ObjectId,
      ref: 'User',
      required: [true, 'Requested by is required'],
    },
    status: {
      type: String,
      enum: {
        values: [
          'pending',
          'processing_prompt',
          'generating_image',
          'completed',
          'failed',
          'cancelled',
        ],
        message: '{VALUE} is not a valid job status',
      },
      default: 'pending',
    },
    generatedPrompt: {
      type: String,
      trim: true,
      maxlength: [5000, 'Generated prompt must not exceed 5000 characters'],
    },
    promptModel: {
      type: String,
      trim: true,
    },
    imageModel: {
  type: String,
  enum: {
    values: [
      'stable-diffusion-xl',
      'claude',
      'black-forest-labs/FLUX.1-schnell'
    ],
    message: '{VALUE} is not a valid image generation provider',
  },
  default: 'stable-diffusion-xl',
},
    generationParams: {
      type: generationParamsSchema,
    },
    images: {
      type: [generatedImageSchema],
      default: [],
    },
    errorMessage: {
      type: String,
      trim: true,
    },
    retryCount: {
      type: Number,
      default: 0,
      min: 0,
      max: 5,
    },
    startedAt: {
      type: Date,
    },
    completedAt: {
      type: Date,
    },
  },
  { timestamps: true }
);

generationJobSchema.index({ case: 1 });
generationJobSchema.index({ witnessDescription: 1 });
generationJobSchema.index({ status: 1 });
generationJobSchema.index({ requestedBy: 1 });
generationJobSchema.index({ createdAt: -1 });

export const GenerationJob = model<IGenerationJob>(
  'GenerationJob',
  generationJobSchema
);