import { InferenceClient } from '@huggingface/inference';
import { Types } from 'mongoose';
import { GenerationJob, IGenerationJob } from '../models/GenerationJob.model';
import { ApiError } from '../utils/apiResponse';
import { env } from '../config/env';
import { logger } from '../config/logger';

// ---------------------------------------------------------------------------
// Image generation via Gemini image models
// Uses the same generativelanguage.googleapis.com endpoint as Gemini text
// gemini-2.5-flash-image is available on standard Gemini API keys
// ---------------------------------------------------------------------------

// Models tried in order — first success wins
// NOTE: On free-tier accounts some model IDs may not be enabled.
// Keep this list conservative; errors will show upstream reason.

const hf = new InferenceClient(env.HUGGINGFACE_API_KEY);


const generateWithGeminiImage = async (
  prompt: string
): Promise<{ buffer: Buffer; modelUsed: string }> => {
  try {
    logger.info('Generating image using Hugging Face');

    const forensicPrompt = `
      forensic composite sketch,
      police artist drawing,
      criminal identification portrait,
      front facing,
      neutral background,
      highly detailed,
      witness description:
      ${prompt}
    `;

    const image = await hf.textToImage({
      model: env.HUGGINGFACE_MODEL,
      inputs: forensicPrompt,
    });

    const blob = image as unknown as Blob;
    const arrayBuffer = await blob.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    logger.info(
      `Hugging Face image generated successfully (${buffer.length} bytes)`
    );

    return {
      buffer,
      modelUsed: env.HUGGINGFACE_MODEL,
    };
  } catch (error: any) {
    logger.error(
      `Hugging Face image generation failed: ${error?.message}`
    );

    throw new ApiError(
      `Image generation failed: ${error?.message}`,
      500
    );
  }
};
// ---------------------------------------------------------------------------
// Exported service functions (unchanged logic, new generator)
// ---------------------------------------------------------------------------

export const generateImageForJob = async (
  jobId: string,
  userId: string
): Promise<IGenerationJob> => {
  if (!Types.ObjectId.isValid(jobId)) {
    throw new ApiError('Invalid Generation Job ID', 400);
  }

  const job = await GenerationJob.findById(jobId);

  if (!job) {
    throw new ApiError('Generation job not found', 404);
  }

  if (job.requestedBy.toString() !== userId) {
    throw new ApiError('You do not have permission to access this job', 403);
  }

  if (!job.generatedPrompt || job.generatedPrompt.trim().length === 0) {
    throw new ApiError('Generation job has no prompt. Run prompt generation first.', 400);
  }

  if (job.status === 'completed') {
    throw new ApiError('This generation job has already been completed.', 409);
  }

  if (job.status === 'generating_image' || job.status === 'processing_prompt') {
    throw new ApiError('This generation job is already in progress.', 409);
  }

  if (job.status === 'cancelled') {
    throw new ApiError('This generation job has been cancelled.', 409);
  }

  await GenerationJob.findByIdAndUpdate(jobId, {
    $set: { status: 'generating_image', startedAt: new Date() },
  });

  logger.info(`Image generation started for job ${jobId}`);

  const params = {
    width: job.generationParams?.width ?? 512,
    height: job.generationParams?.height ?? 512,
    seed: job.generationParams?.seed,
  };

  let imageBuffer: Buffer;
  let modelUsed: string;

  try {
    const result = await generateWithGeminiImage(job.generatedPrompt);
    imageBuffer = result.buffer;
    modelUsed = result.modelUsed;
  } catch (err) {
    const message =
      err instanceof ApiError
        ? err.message
        : err instanceof Error
        ? err.message
        : 'Unknown image generation error';

    await GenerationJob.findByIdAndUpdate(jobId, {
      $set: { status: 'failed', errorMessage: message },
      $inc: { retryCount: 1 },
    });

    logger.error(`Image generation failed for job ${jobId}: ${message}`);
    throw err instanceof ApiError ? err : new ApiError(message, 502);
  }

  const base64Image = imageBuffer.toString('base64');
  const imageDataUrl = `data:image/png;base64,${base64Image}`;

  const completedJob = await GenerationJob.findByIdAndUpdate(
    jobId,
    {
      $set: {
        status: 'completed',
        completedAt: new Date(),
        errorMessage: undefined,
        imageModel: modelUsed,
      },
      $push: {
        images: {
          imageUrl: imageDataUrl,
          storageKey: `job_${jobId}_image_${Date.now()}`,
          width: params.width,
          height: params.height,
          seed: params.seed,
          isSelected: true,
          generatedAt: new Date(),
        },
      },
    },
    { new: true, runValidators: true }
  )
    .populate('witnessDescription', 'rawStatement gender estimatedAge')
    .populate('requestedBy', 'name email')
    .populate('case', 'caseNumber title');

  if (!completedJob) {
    throw new ApiError('Failed to update generation job after image creation', 500);
  }

  logger.info(`Image generation completed for job ${jobId}. Stored as Base64.`);
  return completedJob;
};

export const getJobById = async (
  jobId: string,
  userId: string
): Promise<IGenerationJob> => {
  if (!Types.ObjectId.isValid(jobId)) {
    throw new ApiError('Invalid Generation Job ID', 400);
  }

  const job = await GenerationJob.findById(jobId)
    .populate('witnessDescription', 'rawStatement gender estimatedAge skinTone')
    .populate('requestedBy', 'name email role')
    .populate('case', 'caseNumber title status');

  if (!job) {
    throw new ApiError('Generation job not found', 404);
  }

  if ((job.requestedBy as { _id: Types.ObjectId })._id.toString() !== userId) {
    throw new ApiError('You do not have permission to access this job', 403);
  }

  return job;
};

export const getJobsByWitnessDescription = async (
  witnessDescriptionId: string,
  userId: string
): Promise<IGenerationJob[]> => {
  if (!Types.ObjectId.isValid(witnessDescriptionId)) {
    throw new ApiError('Invalid Witness Description ID', 400);
  }

  const jobs = await GenerationJob.find({
    witnessDescription: new Types.ObjectId(witnessDescriptionId),
    requestedBy: new Types.ObjectId(userId),
  })
    .populate('case', 'caseNumber title')
    .populate('requestedBy', 'name email')
    .sort({ createdAt: -1 })
    .lean();

  return jobs as unknown as IGenerationJob[];
};

export const cancelJob = async (
  jobId: string,
  userId: string
): Promise<IGenerationJob> => {
  if (!Types.ObjectId.isValid(jobId)) {
    throw new ApiError('Invalid Generation Job ID', 400);
  }

  const job = await GenerationJob.findById(jobId);

  if (!job) {
    throw new ApiError('Generation job not found', 404);
  }

  if (job.requestedBy.toString() !== userId) {
    throw new ApiError('You do not have permission to cancel this job', 403);
  }

  if (job.status === 'completed' || job.status === 'generating_image') {
    throw new ApiError(`Cannot cancel a job with status: ${job.status}`, 409);
  }

  const cancelled = await GenerationJob.findByIdAndUpdate(
    jobId,
    { $set: { status: 'cancelled' } },
    { new: true }
  );

  if (!cancelled) {
    throw new ApiError('Failed to cancel generation job', 500);
  }

  logger.info(`Generation job ${jobId} cancelled by user ${userId}`);
  return cancelled;
};