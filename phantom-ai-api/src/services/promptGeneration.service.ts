import { Types } from 'mongoose';
import { WitnessDescription, IWitnessDescription } from '../models/WitnessDescription.model';
import { GenerationJob } from '../models/GenerationJob.model';
import { ApiError } from '../utils/apiResponse';

const AGE_RANGE_MAP: Record<string, string> = {
  child: 'approximately 8-12 year old',
  teen: 'approximately 14-17 year old',
  '18-25': 'approximately 18-25 year old',
  '26-35': 'approximately 26-35 year old',
  '36-45': 'approximately 36-45 year old',
  '46-55': 'approximately 46-55 year old',
  '56-65': 'approximately 56-65 year old',
  '65+': 'approximately 65 or older',
};

const SKIN_TONE_MAP: Record<string, string> = {
  very_light: 'very light',
  light: 'light',
  medium: 'medium',
  tan: 'tan',
  dark: 'dark',
  very_dark: 'very dark',
};

const BUILD_MAP: Record<string, string> = {
  slim: 'slim',
  average: 'average',
  athletic: 'athletic and muscular',
  heavy: 'heavy-set',
  unknown: 'average',
};

const GENDER_MAP: Record<string, string> = {
  male: 'male',
  female: 'female',
  other: 'person',
  unknown: 'individual',
};

const buildAgeGenderPhrase = (
  gender?: string,
  estimatedAge?: string
): string => {
  const genderLabel = GENDER_MAP[gender ?? 'unknown'] ?? 'individual';
  const ageLabel = estimatedAge
    ? AGE_RANGE_MAP[estimatedAge] ?? ''
    : '';

  if (ageLabel) {
    return `${ageLabel} ${genderLabel}`;
  }
  return genderLabel;
};

const buildPhysicalPhrase = (doc: IWitnessDescription): string => {
  const parts: string[] = [];

  if (doc.build && doc.build !== 'unknown') {
    parts.push(`${BUILD_MAP[doc.build]} build`);
  }

  if (doc.skinTone) {
    parts.push(`${SKIN_TONE_MAP[doc.skinTone]} skin tone`);
  }

  if (doc.ethnicity) {
    parts.push(`${doc.ethnicity} ethnicity`);
  }

  if (doc.height) {
    parts.push(`approximately ${doc.height} tall`);
  }

  return parts.join(', ');
};

const buildHairPhrase = (features?: IWitnessDescription['facialFeatures']): string => {
  if (!features) return '';

  const parts: string[] = [];

  if (features.hairLength) parts.push(features.hairLength);
  if (features.hairStyle) parts.push(features.hairStyle);
  if (features.hairColor) parts.push(`${features.hairColor} hair`);

  if (parts.length === 0) return '';

  return parts.join(', ');
};

const buildEyePhrase = (features?: IWitnessDescription['facialFeatures']): string => {
  if (!features) return '';

  const parts: string[] = [];

  if (features.eyeColor) parts.push(features.eyeColor);
  if (features.eyeShape) parts.push(`${features.eyeShape} shaped`);

  if (parts.length === 0) return '';

  return `${parts.join(', ')} eyes`;
};

const buildFacialStructurePhrase = (
  features?: IWitnessDescription['facialFeatures']
): string => {
  if (!features) return '';

  const parts: string[] = [];

  if (features.faceShape) parts.push(`${features.faceShape} face shape`);
  if (features.jawline) parts.push(`${features.jawline} jawline`);
  if (features.cheekbones) parts.push(`${features.cheekbones} cheekbones`);
  if (features.forehead) parts.push(`${features.forehead} forehead`);
  if (features.noseShape) parts.push(`${features.noseShape} nose`);
  if (features.lipShape) parts.push(`${features.lipShape} lips`);
  if (features.eyebrowShape) parts.push(`${features.eyebrowShape} eyebrows`);
  if (features.ears) parts.push(`${features.ears} ears`);
  if (features.facialHair) parts.push(features.facialHair);

  return parts.join(', ');
};

const buildDistinguishingPhrase = (
  features: IWitnessDescription['distinguishingFeatures']
): string => {
  if (!features || features.length === 0) return '';

  return features
    .map((f) => `${f.description} (${f.type}) on ${f.location}`)
    .join('; ');
};

const buildClothingPhrase = (clothing?: string): string => {
  if (!clothing) return '';
  return clothing.trim();
};

export const buildPromptFromWitnessDescription = (
  doc: IWitnessDescription
): string => {
  const segments: string[] = [];

  segments.push(
  'Professional forensic composite sketch created by a police sketch artist'
);
  const ageGender = buildAgeGenderPhrase(doc.gender, doc.estimatedAge);
  segments.push(ageGender);

  const physical = buildPhysicalPhrase(doc);
  if (physical) segments.push(`with ${physical}`);

  const hair = buildHairPhrase(doc.facialFeatures);
  if (hair) segments.push(hair);

  const eyes = buildEyePhrase(doc.facialFeatures);
  if (eyes) segments.push(eyes);

  const facialStructure = buildFacialStructurePhrase(doc.facialFeatures);
  if (facialStructure) segments.push(facialStructure);

  const distinguishing = buildDistinguishingPhrase(doc.distinguishingFeatures);
  if (distinguishing) {
    segments.push(`distinguishing features: ${distinguishing}`);
  }

  const clothing = buildClothingPhrase(doc.clothingDescription);
  if (clothing) segments.push(`wearing ${clothing}`);

  segments.push(
  'black and white pencil drawing, grayscale forensic sketch, criminal suspect reconstruction, front-facing view, neutral expression, plain white background, law enforcement composite style, detailed facial features, investigative reconstruction, no glamour, no beauty enhancement, no fashion photography, no cinematic lighting'
);
  return segments.join(', ');
};

export const generatePromptForWitnessDescription = async (
  witnessDescriptionId: string,
  userId: string
): Promise<{
  prompt: string;
  witnessDescriptionId: string;
  jobId: string;
}> => {
  if (!Types.ObjectId.isValid(witnessDescriptionId)) {
    throw new ApiError('Invalid Witness Description ID', 400);
  }

  const witnessDescription = await WitnessDescription.findById(
    witnessDescriptionId
  ).populate('case', 'caseNumber title');

  if (!witnessDescription) {
    throw new ApiError('Witness description not found', 404);
  }

  const generatedPrompt = buildPromptFromWitnessDescription(witnessDescription);

  const job = await GenerationJob.create({
    case: witnessDescription.case,
    witnessDescription: witnessDescription._id,
    requestedBy: new Types.ObjectId(userId),
    status: 'pending',
    generatedPrompt,
    promptModel: 'phantom-ai-prompt-builder-v1',
    imageModel: 'stable-diffusion-xl',
    generationParams: {
      steps: 40,
      cfgScale: 7,
      width: 512,
      height: 512,
      numImages: 1,
    },
  });

  return {
    prompt: generatedPrompt,
    witnessDescriptionId: witnessDescription._id.toString(),
    jobId: job._id.toString(),
  };
};

export const getPromptByJobId = async (
  jobId: string
): Promise<{
  jobId: string;
  status: string;
  prompt: string | undefined;
  witnessDescriptionId: string;
}> => {
  if (!Types.ObjectId.isValid(jobId)) {
    throw new ApiError('Invalid Job ID', 400);
  }

  const job = await GenerationJob.findById(jobId)
    .populate('witnessDescription', 'rawStatement gender estimatedAge')
    .populate('requestedBy', 'name email');

  if (!job) {
    throw new ApiError('Generation job not found', 404);
  }

  return {
    jobId: job._id.toString(),
    status: job.status,
    prompt: job.generatedPrompt,
    witnessDescriptionId: job.witnessDescription.toString(),
  };
};