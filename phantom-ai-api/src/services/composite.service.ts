import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { Types } from 'mongoose';
import { Composite, IComposite, ICompositeVariation } from '../models/Composite.model';
import { WitnessDescription } from '../models/WitnessDescription.model';
import { SecurityAudit } from '../models/SecurityAudit.model';
import { ApiError } from '../utils/apiResponse';
import { assertCaseAccess } from './witnessDescription.service';
import { mlClient } from './mlClient.service';
import { logger } from '../config/logger';
import {
  resolveGCITTraceForAttribute,
  buildGCITTensorsFromTrace,
  formatAttributeCategory,
  extractSurroundingSentence,
  GCIT_20_DIMENSIONS,
  IGCITTraceItem,
} from './gcitTrace.service';

const COMPOSITES_STORAGE_DIR = path.resolve(process.cwd(), 'data', 'composites');

export interface GenerateCompositeInput {
  witnessDescriptionId: string;
  seed?: number;
  numVariations?: number;
  notes?: string;
  clientIp?: string;
  userAgent?: string;
}

export interface EditCompositeInput {
  seed: number;
  continuousAdjustments?: Record<string, number>;
  latentTraversals?: Array<{ attribute: string; alpha: number }>;
  notes?: string;
  clientIp?: string;
  userAgent?: string;
}

const ensureStorageDir = () => {
  if (!fs.existsSync(COMPOSITES_STORAGE_DIR)) {
    fs.mkdirSync(COMPOSITES_STORAGE_DIR, { recursive: true });
  }
};

const recordAudit = async (
  event: any,
  status: 'success' | 'failure' | 'warning',
  userId: string,
  details: Record<string, unknown>,
  ipAddress?: string,
  userAgent?: string
) => {
  try {
    await SecurityAudit.create({
      userId: new Types.ObjectId(userId),
      event,
      status,
      ipAddress: ipAddress || '127.0.0.1',
      userAgent: userAgent || 'PhantomAI-Backend',
      details,
    });
  } catch (err) {
    logger.error(`[AUDIT-LOG-ERROR] Failed to record audit event ${event}: ${err}`);
  }
};

export const generateCompositeForCase = async (
  caseId: string,
  userId: string,
  userRole: string,
  input: GenerateCompositeInput
): Promise<IComposite> => {
  ensureStorageDir();

  // 1. Enforce RBAC case-level access
  await assertCaseAccess(caseId, userId, userRole, 'edit');

  const { witnessDescriptionId, seed, numVariations = 1, notes, clientIp, userAgent } = input;

  if (!Types.ObjectId.isValid(witnessDescriptionId)) {
    throw new ApiError('Invalid witness description ID format.', 400);
  }

  // 2. Fetch witness description
  const witnessDoc = await WitnessDescription.findOne({
    _id: witnessDescriptionId,
    case: caseId,
  });

  if (!witnessDoc) {
    throw new ApiError(`Witness description ${witnessDescriptionId} not found for case ${caseId}.`, 404);
  }

  // 3. Strictly require investigator approval
  if (witnessDoc.reviewStatus !== 'approved') {
    await recordAudit(
      'COMPOSITE_GENERATION_FAILURE',
      'failure',
      userId,
      {
        caseId,
        witnessDescriptionId,
        reason: 'Attempted generation without investigator approval',
        currentStatus: witnessDoc.reviewStatus,
      },
      clientIp,
      userAgent
    );

    throw new ApiError(
      `Cannot generate facial composite: Witness description ${witnessDescriptionId} has review status '${witnessDoc.reviewStatus}'. ` +
        'Per forensic protocol, all extracted descriptors must be reviewed, corrected if necessary, and formally approved by an investigator before visual composite synthesis.',
      400
    );
  }

  // 4. Construct approved attribute snapshot and complete evidence trace from extractedAttributes
  const approvedSnapshot: Record<string, any> = {};
  const contradictoryCategories: string[] = [];
  const evidenceTrace: IGCITTraceItem[] = [];

  for (const attr of witnessDoc.extractedAttributes) {
    const rawAttr = (attr as any).toObject ? (attr as any).toObject() : attr;
    if (rawAttr.evidenceState === 'contradictory') {
      contradictoryCategories.push(rawAttr.category);
    }
    approvedSnapshot[rawAttr.category] = {
      category: rawAttr.category,
      value: rawAttr.value,
      state: rawAttr.evidenceState,
      source_text: rawAttr.sourceText,
      extracted_by: rawAttr.extractedBy,
    };

    const gcitStatus =
      rawAttr.gcitStatus ||
      resolveGCITTraceForAttribute(rawAttr.category, rawAttr.value, rawAttr.evidenceState);

    evidenceTrace.push({
      attributeName: rawAttr.attributeName || formatAttributeCategory(rawAttr.category),
      category: rawAttr.category,
      effectiveValue: rawAttr.value,
      originalValue: rawAttr.originalValue !== undefined ? rawAttr.originalValue : rawAttr.value,
      evidenceState: rawAttr.evidenceState,
      originalEvidenceState: rawAttr.originalEvidenceState || rawAttr.evidenceState,
      sourceSpan: rawAttr.sourceText || '',
      startChar: typeof rawAttr.start === 'number' ? rawAttr.start : null,
      endChar: typeof rawAttr.end === 'number' ? rawAttr.end : null,
      surroundingContext:
        rawAttr.surroundingContext ||
        extractSurroundingSentence(witnessDoc.rawStatement, rawAttr.start, rawAttr.end),
      extractedBy: rawAttr.extractedBy || 'char_bilstm_crf_phase3c2',
      confidence: typeof rawAttr.confidence === 'number' ? rawAttr.confidence : null,
      isCorrected: rawAttr.isCorrected ?? false,
      correction: rawAttr.correction
        ? {
            correctedValue: rawAttr.correction.correctedValue,
            correctedEvidenceState: rawAttr.correction.correctedEvidenceState,
            reason: rawAttr.correction.reason,
            timestamp: rawAttr.correction.timestamp,
            correctedBy: rawAttr.correction.correctedBy?.toString(),
            rejected: rawAttr.correction.rejected,
          }
        : undefined,
      reviewStatus: witnessDoc.reviewStatus,
      gcitStatus,
    });
  }

  const { cVector, mVector } = buildGCITTensorsFromTrace(evidenceTrace);

  if (contradictoryCategories.length > 0) {
    throw new ApiError(
      `Cannot generate composite: Unresolved contradictory evidence exists in category/categories: [${contradictoryCategories.join(', ')}]. ` +
        'Investigator review and resolution required before generation can execute.',
      409
    );
  }

  await recordAudit(
    'COMPOSITE_GENERATION_INITIATED',
    'success',
    userId,
    {
      caseId,
      witnessDescriptionId,
      seed,
      numVariations,
    },
    clientIp,
    userAgent
  );

  // 5. Query Python ML service for actual conditional composite generation
  let mlResponse;
  try {
    mlResponse = await mlClient.generateComposite({
      forensic_profile: approvedSnapshot,
      seed,
      num_variations: numVariations,
      case_id: caseId,
      witness_id: witnessDescriptionId,
    });
  } catch (err: any) {
    await recordAudit(
      'COMPOSITE_GENERATION_FAILURE',
      'failure',
      userId,
      {
        caseId,
        witnessDescriptionId,
        error: err.message,
      },
      clientIp,
      userAgent
    );
    throw err;
  }

  // 6. Determine version number (sequential per case, zero overwriting)
  const currentVersionCount = await Composite.countDocuments({ case: caseId });
  const version = currentVersionCount + 1;

  // 7. Securely store generated images on disk
  const storedVariations: ICompositeVariation[] = [];
  const createdFilePaths: string[] = [];
  const imageHashes: string[] = [];

  try {
    for (const varItem of mlResponse.variations) {
      const rawB64 = varItem.image_base64;
      const imgBuffer = Buffer.from(rawB64, 'base64');

      // Validate size (between 100 bytes and 5 MB)
      if (imgBuffer.length < 100 || imgBuffer.length > 5 * 1024 * 1024) {
        throw new ApiError('Generated image validation failed: invalid file size.', 500);
      }

      // Validate PNG magic bytes: 89 50 4E 47 0D 0A 1A 0A
      if (
        imgBuffer[0] !== 0x89 ||
        imgBuffer[1] !== 0x50 ||
        imgBuffer[2] !== 0x4e ||
        imgBuffer[3] !== 0x47 ||
        imgBuffer[4] !== 0x0d ||
        imgBuffer[5] !== 0x0a ||
        imgBuffer[6] !== 0x1a ||
        imgBuffer[7] !== 0x0a
      ) {
        throw new ApiError('Generated image validation failed: corrupt or invalid PNG format.', 500);
      }

      // Compute cryptographic SHA-256 digest of image bytes
      const imgSha256 = crypto.createHash('sha256').update(imgBuffer).digest('hex');
      imageHashes.push(imgSha256);

      // Secure unique filename with UUID and case/version tagging
      const safeFilename = `composite_${caseId}_v${version}_var${varItem.variation_index}_${crypto.randomUUID()}.png`;
      const fullPath = path.resolve(COMPOSITES_STORAGE_DIR, safeFilename);

      // Prevent path traversal
      if (!fullPath.startsWith(COMPOSITES_STORAGE_DIR)) {
        throw new ApiError('Security violation: invalid storage path traversal attempt.', 400);
      }

      fs.writeFileSync(fullPath, imgBuffer);
      createdFilePaths.push(fullPath);

      storedVariations.push({
        variationIndex: varItem.variation_index,
        seed: varItem.seed,
        storageKey: safeFilename,
        imageUrl: `/api/v1/cases/${caseId}/composites/placeholder/image/${varItem.variation_index}`,
        dataUrl: varItem.data_url,
        mimeType: 'image/png',
        fileSizeBytes: imgBuffer.length,
        width: varItem.width,
        height: varItem.height,
        isSelected: varItem.variation_index === 1,
        variationRole: 'standard',
        gcitVector: [...cVector],
        gcitMask: [...mVector],
        sha256: imgSha256,
      });
    }

    // 8. Create composite document with full evidence traceability snapshot
    const compositeDoc = await Composite.create({
      case: new Types.ObjectId(caseId),
      witnessDescription: new Types.ObjectId(witnessDescriptionId),
      version,
      status: 'draft',
      generatorMetadata: {
        modelName: mlResponse.generator.model_name,
        architecture: mlResponse.generator.architecture,
        phase: mlResponse.generator.phase,
        checkpointPath: mlResponse.generator.checkpoint_path,
        checkpointSha256: mlResponse.generator.checkpoint_sha256,
        bestEpoch: mlResponse.generator.best_epoch,
        parameterCount: mlResponse.generator.parameter_count,
        device: mlResponse.generator.device,
      },
      approvedEvidenceSnapshot: approvedSnapshot,
      evidenceTrace,
      gcitVector: cVector,
      gcitMask: mVector,
      imageHashes,
      conditionedAttributes: mlResponse.conditioned_attributes,
      unconditionedAttributes: mlResponse.unconditioned_attributes,
      warnings: mlResponse.warnings,
      variations: storedVariations,
      selectionHistory: [],
      createdBy: new Types.ObjectId(userId),
      notes: notes?.trim(),
    });

    // Update image URLs with actual compositeDoc ID
    compositeDoc.variations.forEach((v) => {
      v.imageUrl = `/api/v1/cases/${caseId}/composites/${compositeDoc._id}/image/${v.variationIndex}`;
    });
    await compositeDoc.save();

    await recordAudit(
      'COMPOSITE_GENERATED_SUCCESS',
      'success',
      userId,
      {
        caseId,
        compositeId: compositeDoc._id.toString(),
        version,
        numVariations: storedVariations.length,
        checkpointSha256: mlResponse.generator.checkpoint_sha256,
      },
      clientIp,
      userAgent
    );

    return compositeDoc;
  } catch (err) {
    // Clean up any partially written files on storage error or database error
    for (const f of createdFilePaths) {
      if (fs.existsSync(f)) {
        try {
          fs.unlinkSync(f);
        } catch (_) {
          // Ignore cleanup errors
        }
      }
    }
    throw err;
  }
};

export const getCompositesForCase = async (
  caseId: string,
  userId: string,
  userRole: string
): Promise<IComposite[]> => {
  await assertCaseAccess(caseId, userId, userRole, 'read');

  const composites = await Composite.find({ case: caseId })
    .populate('createdBy', 'name email role investigatorId')
    .populate('reviewedBy', 'name email role investigatorId')
    .populate('variations.approvedBy', 'name email role investigatorId')
    .populate('selectionHistory.selectedBy', 'name email role investigatorId')
    .populate('witnessDescription', 'witnessName rawStatement reviewStatus')
    .sort({ version: -1 })
    .lean();

  return composites as unknown as IComposite[];
};

export const getCompositeById = async (
  caseId: string,
  compositeId: string,
  userId: string,
  userRole: string
): Promise<IComposite> => {
  await assertCaseAccess(caseId, userId, userRole, 'read');

  if (!Types.ObjectId.isValid(compositeId)) {
    throw new ApiError('Invalid composite ID format.', 400);
  }

  const composite = await Composite.findOne({ _id: compositeId, case: caseId })
    .populate('createdBy', 'name email role investigatorId')
    .populate('reviewedBy', 'name email role investigatorId')
    .populate('variations.approvedBy', 'name email role investigatorId')
    .populate('selectionHistory.selectedBy', 'name email role investigatorId')
    .populate('witnessDescription', 'witnessName rawStatement reviewStatus');

  if (!composite) {
    throw new ApiError(`Composite ${compositeId} not found for case ${caseId}.`, 404);
  }

  return composite;
};

export const getCompositeImageFile = async (
  caseId: string,
  compositeId: string,
  variationIndex: number,
  userId: string,
  userRole: string,
  clientIp?: string,
  userAgent?: string
): Promise<{ filePath: string; mimeType: string }> => {
  await assertCaseAccess(caseId, userId, userRole, 'read');

  if (!Types.ObjectId.isValid(compositeId)) {
    throw new ApiError('Invalid composite ID format.', 400);
  }

  const composite = await Composite.findOne({ _id: compositeId, case: caseId });
  if (!composite) {
    throw new ApiError('Composite not found.', 404);
  }

  const variation = composite.variations.find((v) => v.variationIndex === Number(variationIndex));
  if (!variation) {
    throw new ApiError(`Variation ${variationIndex} not found on composite ${compositeId}.`, 404);
  }

  const safeFilename = path.basename(variation.storageKey);
  const fullPath = path.resolve(COMPOSITES_STORAGE_DIR, safeFilename);

  if (!fullPath.startsWith(COMPOSITES_STORAGE_DIR) || !fs.existsSync(fullPath)) {
    throw new ApiError('Composite image file not found on server storage.', 404);
  }

  await recordAudit(
    'COMPOSITE_VIEWED',
    'success',
    userId,
    {
      caseId,
      compositeId,
      variationIndex,
      storageKey: safeFilename,
    },
    clientIp,
    userAgent
  );

  return {
    filePath: fullPath,
    mimeType: variation.mimeType || 'image/png',
  };
};

export const selectCompositeVariation = async (
  caseId: string,
  compositeId: string,
  variationIndex: number,
  userId: string,
  userRole: string,
  status: 'draft' | 'under_review' | 'approved' | 'superseded' = 'approved',
  notes?: string,
  clientIp?: string,
  userAgent?: string
): Promise<IComposite> => {
  await assertCaseAccess(caseId, userId, userRole, 'edit');

  if (!Types.ObjectId.isValid(compositeId)) {
    throw new ApiError('Invalid composite ID format.', 400);
  }

  const composite = await Composite.findOne({ _id: compositeId, case: caseId });
  if (!composite) {
    throw new ApiError('Composite not found.', 404);
  }

  const previousSelected = composite.variations.find((v) => v.isSelected);
  const previousVariationIndex = previousSelected ? previousSelected.variationIndex : null;

  const now = new Date();
  let selectedVariation: ICompositeVariation | undefined;
  for (const v of composite.variations) {
    if (v.variationIndex === Number(variationIndex)) {
      v.isSelected = true;
      v.approvedAt = now;
      v.approvedBy = new Types.ObjectId(userId);
      if (notes) {
        v.selectionNotes = notes.trim();
      }
      selectedVariation = v;
    } else {
      v.isSelected = false;
    }
  }

  if (!selectedVariation) {
    throw new ApiError(`Variation ${variationIndex} not found on composite ${compositeId}.`, 404);
  }

  if (!Array.isArray(composite.selectionHistory)) {
    composite.selectionHistory = [];
  }

  composite.selectionHistory.push({
    variationIndex: Number(variationIndex),
    variationId: selectedVariation._id,
    seed: selectedVariation.seed,
    previousVariationIndex,
    selectedBy: new Types.ObjectId(userId),
    selectedAt: now,
    status,
    notes: notes?.trim() || 'Marked as preferred variation for further investigative review',
  });

  composite.status = status;
  composite.reviewedBy = new Types.ObjectId(userId);
  composite.reviewedAt = now;
  if (notes) {
    composite.notes = notes.trim();
  }

  await composite.save();

  if (typeof (composite as any).populate === 'function') {
    try {
      await composite.populate([
        { path: 'createdBy', select: 'name email role investigatorId' },
        { path: 'reviewedBy', select: 'name email role investigatorId' },
        { path: 'variations.approvedBy', select: 'name email role investigatorId' },
        { path: 'selectionHistory.selectedBy', select: 'name email role investigatorId' },
      ]);
    } catch (_) {
      // Populate is optional for mocked unit tests
    }
  }

  await recordAudit(
    'COMPOSITE_APPROVED',
    'success',
    userId,
    {
      caseId,
      compositeId,
      selectedVariationIndex: Number(variationIndex),
      previousVariationIndex,
      seed: selectedVariation.seed,
      newStatus: status,
      notes: notes?.trim() || null,
    },
    clientIp,
    userAgent
  );

  await recordAudit(
    'COMPOSITE_VARIATION_SELECTED',
    'success',
    userId,
    {
      caseId,
      compositeId,
      version: composite.version,
      selectedVariationIndex: Number(variationIndex),
      previousVariationIndex,
      seed: selectedVariation.seed,
      status,
      notes: notes?.trim() || null,
    },
    clientIp,
    userAgent
  );

  return composite;
};

/**
 * Applies continuous GCIT adjustments (c_i in [-1.0, +1.0], m_i = 1.0) to a base 20-D GCIT vector & mask.
 */
export const applyContinuousAdjustmentsToGCIT = (
  baseCVector: number[],
  baseMVector: number[],
  continuousAdjustments?: Record<string, number>
): { cVector: number[]; mVector: number[] } => {
  const cVector = new Array(20).fill(0.0).map((_, idx) =>
    Array.isArray(baseCVector) && typeof baseCVector[idx] === 'number' ? baseCVector[idx] : 0.0
  );
  const mVector = new Array(20).fill(0.0).map((_, idx) =>
    Array.isArray(baseMVector) && typeof baseMVector[idx] === 'number' ? baseMVector[idx] : 0.0
  );

  if (continuousAdjustments && typeof continuousAdjustments === 'object') {
    for (const [attrName, rawVal] of Object.entries(continuousAdjustments)) {
      const dim = GCIT_20_DIMENSIONS.find((d) => d.name === attrName);
      if (dim && typeof rawVal === 'number' && !Number.isNaN(rawVal)) {
        const clamped = Math.max(-1.0, Math.min(1.0, Number(rawVal)));
        cVector[dim.index] = Math.round(clamped * 1000) / 1000;
        mVector[dim.index] = 1.0;
      }
    }
  }

  return { cVector, mVector };
};

/**
 * Retrieves the evaluation status and supported dimensions for latent directions.
 */
export const getLatentDirections = async () => {
  return await mlClient.getLatentDirections();
};

/**
 * Executes controlled semantic editing on an approved composite version.
 * Preserves original composite, verifies case authorization, increments version,
 * and records provenance in MongoDB and SecurityAudit.
 */
export const editCompositeForCase = async (
  caseId: string,
  compositeId: string,
  userId: string,
  userRole: string,
  input: EditCompositeInput
): Promise<{ composite: IComposite; originalVariation?: any }> => {
  ensureStorageDir();
  await assertCaseAccess(caseId, userId, userRole, 'edit');

  if (!Types.ObjectId.isValid(compositeId)) {
    throw new ApiError('Invalid composite ID format.', 400);
  }

  const sourceComposite = await Composite.findOne({ _id: compositeId, case: caseId });
  if (!sourceComposite) {
    throw new ApiError('Source composite not found for this case.', 404);
  }

  const witness = await WitnessDescription.findOne({ _id: sourceComposite.witnessDescription, case: caseId });
  if (!witness) {
    throw new ApiError('Associated witness description record not found.', 404);
  }

  if (witness.reviewStatus !== 'approved') {
    throw new ApiError(
      `Cannot edit composite: Witness description ${witness._id} review status is '${witness.reviewStatus}'. Only approved forensic evidence profiles can be edited.`,
      400
    );
  }

  // Determine next sequential version
  const latestComposite = await Composite.findOne({ case: caseId }).sort({ version: -1 });
  const nextVersion = (latestComposite?.version || 1) + 1;

  // Determine editing type
  const hasAdj = input.continuousAdjustments && Object.keys(input.continuousAdjustments).length > 0;
  const hasTrav = input.latentTraversals && input.latentTraversals.length > 0;
  let editingType: 'initial' | 'conditioning_traversal' | 'latent_traversal' | 'hybrid_traversal' = 'conditioning_traversal';
  if (hasAdj && hasTrav) {
    editingType = 'hybrid_traversal';
  } else if (hasTrav) {
    editingType = 'latent_traversal';
  }

  // Call Python ML service
  const mlResponse = await mlClient.editComposite({
    forensic_profile: sourceComposite.approvedEvidenceSnapshot,
    seed: input.seed,
    continuous_gcit_adjustments: input.continuousAdjustments || {},
    latent_traversals: input.latentTraversals || [],
    include_original: true,
    case_id: caseId,
    witness_id: sourceComposite.witnessDescription.toString(),
  });

  const createdFilePaths: string[] = [];
  const editImageHashes: string[] = [];

  try {
    const storedVariations: ICompositeVariation[] = [];

    // Resolve baseline and edited 20-D GCIT conditioning vectors
    let baseCVector = Array.isArray(sourceComposite.gcitVector) && sourceComposite.gcitVector.length === 20
      ? [...sourceComposite.gcitVector]
      : [];
    let baseMVector = Array.isArray(sourceComposite.gcitMask) && sourceComposite.gcitMask.length === 20
      ? [...sourceComposite.gcitMask]
      : [];

    if (baseCVector.length !== 20 || baseMVector.length !== 20) {
      const built = buildGCITTensorsFromTrace((sourceComposite.evidenceTrace || []) as unknown as IGCITTraceItem[]);
      baseCVector = built.cVector;
      baseMVector = built.mVector;
    }

    const { cVector: editedCVector, mVector: editedMVector } = applyContinuousAdjustmentsToGCIT(
      baseCVector,
      baseMVector,
      input.continuousAdjustments
    );

    // Helper to validate and save variation
    const saveVariation = (
      varItem: any,
      index: number,
      isSelected: boolean,
      role: 'semantic_edited' | 'semantic_baseline',
      varCVector: number[],
      varMVector: number[]
    ): ICompositeVariation => {
      const imgBuffer = Buffer.from(varItem.image_base64, 'base64');
      if (imgBuffer.length < 100 || imgBuffer.length > 5 * 1024 * 1024) {
        throw new ApiError('Generated image validation failed: invalid file size.', 500);
      }
      if (
        imgBuffer[0] !== 0x89 ||
        imgBuffer[1] !== 0x50 ||
        imgBuffer[2] !== 0x4e ||
        imgBuffer[3] !== 0x47 ||
        imgBuffer[4] !== 0x0d ||
        imgBuffer[5] !== 0x0a ||
        imgBuffer[6] !== 0x1a ||
        imgBuffer[7] !== 0x0a
      ) {
        throw new ApiError('Generated image validation failed: corrupt or invalid PNG format.', 500);
      }

      const imgSha256 = crypto.createHash('sha256').update(imgBuffer).digest('hex');
      editImageHashes.push(imgSha256);

      const safeFilename = `composite_${caseId}_v${nextVersion}_var${index}_${crypto.randomUUID()}.png`;
      const fullPath = path.resolve(COMPOSITES_STORAGE_DIR, safeFilename);

      if (!fullPath.startsWith(COMPOSITES_STORAGE_DIR)) {
        throw new ApiError('Security violation: invalid storage path traversal attempt.', 400);
      }

      fs.writeFileSync(fullPath, imgBuffer);
      createdFilePaths.push(fullPath);

      return {
        variationIndex: index,
        seed: varItem.seed,
        storageKey: safeFilename,
        imageUrl: `/api/v1/cases/${caseId}/composites/placeholder/image/${index}`,
        dataUrl: varItem.data_url,
        mimeType: 'image/png',
        fileSizeBytes: imgBuffer.length,
        width: varItem.width,
        height: varItem.height,
        isSelected,
        variationRole: role,
        gcitVector: [...varCVector],
        gcitMask: [...varMVector],
        sha256: imgSha256,
      };
    };

    // Save edited composite (Variation 1 - Selected by default)
    storedVariations.push(
      saveVariation(mlResponse.edited_composite, 1, true, 'semantic_edited', editedCVector, editedMVector)
    );

    // Save baseline comparison composite (Variation 2 - for comparison)
    if (mlResponse.original_composite) {
      storedVariations.push(
        saveVariation(mlResponse.original_composite, 2, false, 'semantic_baseline', baseCVector, baseMVector)
      );
    }

    const newComposite = await Composite.create({
      case: new Types.ObjectId(caseId),
      witnessDescription: sourceComposite.witnessDescription,
      version: nextVersion,
      status: 'draft',
      generatorMetadata: {
        modelName: mlResponse.generator.model_name,
        architecture: mlResponse.generator.architecture,
        phase: 'Phase 5.7 Semantic Editing',
        checkpointPath: mlResponse.generator.checkpoint_path,
        checkpointSha256: mlResponse.generator.checkpoint_sha256,
        bestEpoch: mlResponse.generator.best_epoch,
        parameterCount: mlResponse.generator.parameter_count,
        device: mlResponse.generator.device,
      },
      approvedEvidenceSnapshot: sourceComposite.approvedEvidenceSnapshot,
      evidenceTrace: sourceComposite.evidenceTrace || [],
      gcitVector: editedCVector,
      gcitMask: editedMVector,
      imageHashes: editImageHashes,
      conditionedAttributes: mlResponse.conditioned_attributes,
      unconditionedAttributes: [],
      warnings: mlResponse.warnings,
      variations: storedVariations,
      selectionHistory: [],
      parentComposite: sourceComposite._id,
      editingType,
      traversalParameters: {
        continuousAdjustments: input.continuousAdjustments,
        latentTraversals: input.latentTraversals,
      },
      createdBy: new Types.ObjectId(userId),
      notes: input.notes?.trim(),
    });

    // Update image URLs
    newComposite.variations.forEach((v) => {
      v.imageUrl = `/api/v1/cases/${caseId}/composites/${newComposite._id}/image/${v.variationIndex}`;
    });
    await newComposite.save();

    await recordAudit(
      'COMPOSITE_SEMANTIC_EDIT',
      'success',
      userId,
      {
        caseId,
        sourceCompositeId: compositeId,
        newCompositeId: newComposite._id.toString(),
        version: nextVersion,
        editingType,
        appliedAdjustments: mlResponse.applied_continuous_adjustments,
        appliedTraversals: mlResponse.applied_latent_traversals,
      },
      input.clientIp,
      input.userAgent
    );

    return {
      composite: newComposite,
      originalVariation: mlResponse.original_composite,
    };
  } catch (err) {
    for (const filePath of createdFilePaths) {
      try {
        if (fs.existsSync(filePath)) {
          fs.unlinkSync(filePath);
        }
      } catch (cleanupErr) {
        logger.error(`Failed to clean up partial file ${filePath}:`, cleanupErr);
      }
    }
    throw err;
  }
};

/**
 * Retrieves the complete evidence-to-generation traceability manifest for a composite version.
 * Exposes all 15 traceability points for each attribute, the exact GCIT vectors, image SHA-256 hashes,
 * and immutable provenance snapshots.
 */
export const getCompositeTrace = async (
  caseId: string,
  compositeId: string,
  userId: string,
  userRole: string
) => {
  await assertCaseAccess(caseId, userId, userRole, 'read');

  if (!Types.ObjectId.isValid(compositeId)) {
    throw new ApiError('Invalid composite ID format.', 400);
  }

  const composite = await Composite.findOne({ _id: compositeId, case: caseId })
    .populate('witnessDescription', 'rawStatement witnessName reviewStatus recordedBy createdAt')
    .populate('createdBy', 'name email role')
    .populate('reviewedBy', 'name email role');

  if (!composite) {
    throw new ApiError('Composite not found for this case.', 404);
  }

  return {
    compositeId: composite._id,
    caseId: composite.case,
    version: composite.version,
    status: composite.status,
    witnessDescription: composite.witnessDescription,
    generatorMetadata: composite.generatorMetadata,
    evidenceTrace: composite.evidenceTrace || [],
    gcitVector: composite.gcitVector || [],
    gcitMask: composite.gcitMask || [],
    imageHashes: composite.imageHashes || [],
    conditionedAttributes: composite.conditionedAttributes || [],
    unconditionedAttributes: composite.unconditionedAttributes || [],
    warnings: composite.warnings || [],
    variationsCount: composite.variations.length,
    parentComposite: composite.parentComposite || null,
    editingType: composite.editingType || 'initial',
    traversalParameters: composite.traversalParameters || null,
    createdBy: composite.createdBy,
    reviewedBy: composite.reviewedBy || null,
    createdAt: composite.createdAt,
    updatedAt: composite.updatedAt,
  };
};

export interface CompareVariationTarget {
  compositeId: string;
  variationIndex: number;
}

export interface CompareVariationsInput {
  compositeId?: string;
  variationIndices?: number[];
  targets?: CompareVariationTarget[];
  clientIp?: string;
  userAgent?: string;
}

const SUPPORTED_GCIT_CATEGORIES = new Set([
  'hair',
  'hair_color',
  'hair_style',
  'hairline',
  'facial_hair',
  'beard',
  'mustache',
  'nose',
  'nose_shape',
  'lips',
  'lip_shape',
  'face_shape',
  'glasses',
]);

/**
 * Resolves the effective 20-D GCIT conditioning vector, binary mask, and evidence trace for a specific variation.
 * Handles both standard variations and Phase 5.7 semantic editing variations (where Variation 1 is edited
 * and Variation 2 is the unadjusted baseline).
 */
const resolveVariationConditioning = (
  composite: IComposite,
  variation: ICompositeVariation,
  parentComposite?: IComposite | null
): {
  cVector: number[];
  mVector: number[];
  evidenceTrace: IGCITTraceItem[];
  continuousAdjustments: Record<string, number>;
  latentTraversals: Array<{ attribute: string; alpha: number }>;
  role: 'standard' | 'semantic_edited' | 'semantic_baseline';
} => {
  // 1. Reconstruct evidenceTrace if not explicitly stored
  let traceList: IGCITTraceItem[] = Array.isArray(composite.evidenceTrace) && composite.evidenceTrace.length > 0
    ? (composite.evidenceTrace as unknown as IGCITTraceItem[])
    : [];

  if (traceList.length === 0 && composite.approvedEvidenceSnapshot && typeof composite.approvedEvidenceSnapshot === 'object') {
    for (const [cat, snap] of Object.entries(composite.approvedEvidenceSnapshot)) {
      if (snap && typeof snap === 'object') {
        const val = snap.value ?? null;
        const state = snap.state || snap.evidence_state || (val ? 'explicit' : 'unknown');
        const gcitStatus = resolveGCITTraceForAttribute(cat, val, state);
        traceList.push({
          attributeName: formatAttributeCategory(cat),
          category: cat,
          effectiveValue: val,
          originalValue: val,
          evidenceState: state,
          originalEvidenceState: state,
          sourceSpan: snap.source_text || val || '',
          startChar: null,
          endChar: null,
          surroundingContext: snap.source_text || '',
          extractedBy: snap.extracted_by || 'char_bilstm_crf_phase3c2',
          confidence: null,
          isCorrected: false,
          reviewStatus: 'approved',
          gcitStatus,
        });
      }
    }
  }

  // 2. Determine variation role
  const isEditedComposite = Boolean(composite.editingType && composite.editingType !== 'initial');
  let role: 'standard' | 'semantic_edited' | 'semantic_baseline' = variation.variationRole || 'standard';
  if (!variation.variationRole && isEditedComposite) {
    role = variation.variationIndex === 1 ? 'semantic_edited' : 'semantic_baseline';
  }

  // 3. Determine base GCIT vectors (before continuous adjustments on this composite)
  const builtFromTrace = buildGCITTensorsFromTrace(traceList);
  let baseCVector = builtFromTrace.cVector;
  let baseMVector = builtFromTrace.mVector;

  if (parentComposite && Array.isArray(parentComposite.gcitVector) && parentComposite.gcitVector.length === 20) {
    baseCVector = [...parentComposite.gcitVector];
  }
  if (parentComposite && Array.isArray(parentComposite.gcitMask) && parentComposite.gcitMask.length === 20) {
    baseMVector = [...parentComposite.gcitMask];
  }

  // 4. Determine effective continuous adjustments and latent traversals for this variation
  const rawContAdj =
    role === 'semantic_edited' && composite.traversalParameters?.continuousAdjustments
      ? composite.traversalParameters.continuousAdjustments
      : {};
  const rawLatTrav =
    role === 'semantic_edited' && Array.isArray(composite.traversalParameters?.latentTraversals)
      ? composite.traversalParameters!.latentTraversals!
      : [];

  // 5. Resolve final 20-D cVector and mVector for this variation
  let cVector: number[];
  let mVector: number[];

  if (Array.isArray(variation.gcitVector) && variation.gcitVector.length === 20 &&
      Array.isArray(variation.gcitMask) && variation.gcitMask.length === 20) {
    cVector = [...variation.gcitVector];
    mVector = [...variation.gcitMask];
  } else if (role === 'semantic_baseline') {
    cVector = [...baseCVector];
    mVector = [...baseMVector];
  } else if (role === 'semantic_edited') {
    const baseForEdit =
      Array.isArray(composite.gcitVector) && composite.gcitVector.length === 20
        ? [...composite.gcitVector]
        : baseCVector;
    const maskForEdit =
      Array.isArray(composite.gcitMask) && composite.gcitMask.length === 20
        ? [...composite.gcitMask]
        : baseMVector;
    const adjusted = applyContinuousAdjustmentsToGCIT(baseForEdit, maskForEdit, rawContAdj);
    cVector = adjusted.cVector;
    mVector = adjusted.mVector;
  } else if (
    Array.isArray(composite.gcitVector) && composite.gcitVector.length === 20 &&
    Array.isArray(composite.gcitMask) && composite.gcitMask.length === 20
  ) {
    cVector = [...composite.gcitVector];
    mVector = [...composite.gcitMask];
  } else {
    cVector = [...baseCVector];
    mVector = [...baseMVector];
  }

  return {
    cVector,
    mVector,
    evidenceTrace: traceList,
    continuousAdjustments: rawContAdj,
    latentTraversals: rawLatTrav,
    role,
  };
};

/**
 * Phase 7.1: Compares multiple composite variations belonging to the same case.
 * Computes side-by-side metadata, 20-dimensional GCIT conditioning and mask differences,
 * recorded attribute classification (conditioned, masked, uncertain, unsupported),
 * and investigator selection history without inferring attributes from generated pixels.
 */
export const compareCompositeVariations = async (
  caseId: string,
  userId: string,
  userRole: string,
  input: CompareVariationsInput
) => {
  // 1. Enforce RBAC case-level authorization
  await assertCaseAccess(caseId, userId, userRole, 'read');

  // 2. Resolve target variations list
  let rawTargets: CompareVariationTarget[] = [];

  if (Array.isArray(input.targets) && input.targets.length > 0) {
    rawTargets = input.targets.map((t) => ({
      compositeId: String(t.compositeId).trim(),
      variationIndex: Number(t.variationIndex),
    }));
  } else if (input.compositeId) {
    const compId = String(input.compositeId).trim();
    if (!Types.ObjectId.isValid(compId)) {
      throw new ApiError('Invalid composite ID format.', 400);
    }

    const singleComp = await Composite.findOne({ _id: compId, case: caseId })
      .populate('createdBy', 'name email role investigatorId')
      .populate('reviewedBy', 'name email role investigatorId')
      .populate('variations.approvedBy', 'name email role investigatorId')
      .populate('selectionHistory.selectedBy', 'name email role investigatorId')
      .populate('witnessDescription', 'witnessName rawStatement reviewStatus');

    if (!singleComp) {
      throw new ApiError(`Composite ${compId} not found for case ${caseId}.`, 404);
    }

    if (Array.isArray(input.variationIndices) && input.variationIndices.length > 0) {
      rawTargets = input.variationIndices.map((idx) => ({
        compositeId: compId,
        variationIndex: Number(idx),
      }));
    } else {
      rawTargets = (singleComp.variations || []).map((v) => ({
        compositeId: compId,
        variationIndex: v.variationIndex,
      }));
    }
  } else {
    throw new ApiError('Comparison requires a compositeId or a non-empty targets list.', 400);
  }

  // Deduplicate targets while preserving order
  const seenKeys = new Set<string>();
  const targets: CompareVariationTarget[] = [];
  for (const t of rawTargets) {
    if (!Number.isInteger(t.variationIndex) || t.variationIndex < 1) {
      throw new ApiError(`Invalid variation index: ${t.variationIndex}. Must be a positive integer.`, 400);
    }
    const key = `${t.compositeId}:${t.variationIndex}`;
    if (!seenKeys.has(key)) {
      seenKeys.add(key);
      targets.push(t);
    }
  }

  if (targets.length === 0) {
    throw new ApiError('No variations available for comparison.', 400);
  }

  // 3. Validate all composite IDs and verify case ownership (prevent cross-case access)
  const uniqueCompositeIds = Array.from(new Set(targets.map((t) => t.compositeId)));
  const compositeMap = new Map<string, IComposite>();
  const parentCompositeMap = new Map<string, IComposite | null>();

  for (const compId of uniqueCompositeIds) {
    if (!Types.ObjectId.isValid(compId)) {
      throw new ApiError(`Invalid composite ID format: ${compId}.`, 400);
    }

    const compDoc = await Composite.findOne({ _id: compId, case: caseId })
      .populate('createdBy', 'name email role investigatorId')
      .populate('reviewedBy', 'name email role investigatorId')
      .populate('variations.approvedBy', 'name email role investigatorId')
      .populate('selectionHistory.selectedBy', 'name email role investigatorId')
      .populate('witnessDescription', 'witnessName rawStatement reviewStatus');

    if (!compDoc) {
      throw new ApiError(`Composite ${compId} not found for case ${caseId}.`, 404);
    }

    compositeMap.set(compId, compDoc);

    if (compDoc.parentComposite && Types.ObjectId.isValid(compDoc.parentComposite.toString())) {
      const parentDoc = await Composite.findOne({ _id: compDoc.parentComposite, case: caseId });
      parentCompositeMap.set(compId, parentDoc || null);
    } else {
      parentCompositeMap.set(compId, null);
    }
  }

  // 4. Resolve each variation's metadata and conditioning
  const resolvedItems = targets.map((target) => {
    const composite = compositeMap.get(target.compositeId)!;
    const variationIdxInArray = composite.variations.findIndex(
      (v) => v.variationIndex === Number(target.variationIndex)
    );
    if (variationIdxInArray === -1) {
      throw new ApiError(
        `Variation ${target.variationIndex} not found on composite ${target.compositeId}.`,
        404
      );
    }

    const variation = composite.variations[variationIdxInArray];
    const parentComposite = parentCompositeMap.get(target.compositeId) || null;
    const conditioning = resolveVariationConditioning(composite, variation, parentComposite);

    const sha256 =
      variation.sha256 ||
      (Array.isArray(composite.imageHashes) ? composite.imageHashes[variationIdxInArray] : null) ||
      null;

    const conditionedCount = conditioning.mVector.filter((m) => m >= 0.5).length;
    const maskedCount = 20 - conditionedCount;

    return {
      variationKey: `${target.compositeId}:${variation.variationIndex}`,
      variationId: variation._id ? variation._id.toString() : null,
      compositeId: composite._id.toString(),
      caseId: composite.case.toString(),
      witnessDescriptionId:
        (composite.witnessDescription as any)?._id?.toString() ||
        composite.witnessDescription?.toString() ||
        null,
      version: composite.version,
      compositeStatus: composite.status,
      variationIndex: variation.variationIndex,
      variationRole: conditioning.role,
      seed: variation.seed,
      storageKey: variation.storageKey,
      imageUrl: variation.imageUrl,
      dataUrl: variation.dataUrl || null,
      mimeType: variation.mimeType || 'image/png',
      width: variation.width || 64,
      height: variation.height || 64,
      fileSizeBytes: variation.fileSizeBytes,
      sha256,
      isSelected: Boolean(variation.isSelected),
      selectedAt: variation.approvedAt || null,
      selectedBy: variation.approvedBy || null,
      selectionNotes: variation.selectionNotes || composite.notes || null,
      createdAt: composite.createdAt,
      createdBy: composite.createdBy || null,
      reviewedBy: composite.reviewedBy || null,
      reviewedAt: composite.reviewedAt || null,
      editingType: composite.editingType || 'initial',
      parentCompositeId: composite.parentComposite ? composite.parentComposite.toString() : null,
      traversalParameters: {
        continuousAdjustments: conditioning.continuousAdjustments,
        latentTraversals: conditioning.latentTraversals,
      },
      generatorMetadata: composite.generatorMetadata,
      gcitVector: conditioning.cVector,
      gcitMask: conditioning.mVector,
      conditionedCount,
      maskedCount,
      _trace: conditioning.evidenceTrace,
    };
  });

  // 5. Build 20-Dimensional GCIT Comparison Matrix
  const dimensionComparison = GCIT_20_DIMENSIONS.map((dim) => {
    const perVariation = resolvedItems.map((item) => {
      const cVal = Number(item.gcitVector[dim.index] ?? 0.0);
      const mVal = Number(item.gcitMask[dim.index] ?? 0.0);
      const isConditioned = mVal >= 0.5;

      const contAdj = item.traversalParameters.continuousAdjustments?.[dim.name];
      const isContinuouslyAdjusted = typeof contAdj === 'number';

      const latTrav = item.traversalParameters.latentTraversals?.find(
        (lt) => lt.attribute === dim.name
      );
      const latentTraversalAlpha = latTrav ? Number(latTrav.alpha) : null;

      // Match trace item mapped to this GCIT index, or fallback to category trace item
      const mappedTrace = item._trace.find((tr) => tr.gcitStatus?.gcitIndex === dim.index);
      const categoryTrace = item._trace.find((tr) => {
        const normCat = (tr.category || '').toLowerCase();
        if (dim.category === 'other_observable') {
          return (normCat === 'other_observable' || normCat === 'glasses') &&
            (tr.effectiveValue || '').toLowerCase().includes('glass');
        }
        return normCat === dim.category || normCat.startsWith(dim.category);
      });

      let status: 'conditioned' | 'masked_uncertain' | 'masked_unknown' | 'contradictory' =
        'masked_unknown';
      if (isConditioned) {
        status = 'conditioned';
      } else if (categoryTrace?.evidenceState === 'contradictory') {
        status = 'contradictory';
      } else if (categoryTrace?.evidenceState === 'uncertain') {
        status = 'masked_uncertain';
      }

      let recordedAttributeValue: string | null = null;
      let evidenceState: 'explicit' | 'uncertain' | 'negated' | 'contradictory' | 'unknown' =
        'unknown';

      if (isContinuouslyAdjusted) {
        const baseLabel = mappedTrace?.effectiveValue || categoryTrace?.effectiveValue;
        const formattedC = cVal >= 0 ? `+${cVal.toFixed(2)}` : cVal.toFixed(2);
        recordedAttributeValue = baseLabel
          ? `${baseLabel} (Adjusted: ${formattedC})`
          : `Continuous GCIT Adjustment (${formattedC})`;
        evidenceState = 'explicit';
      } else if (isConditioned && mappedTrace) {
        recordedAttributeValue = mappedTrace.effectiveValue;
        evidenceState = mappedTrace.evidenceState;
      } else if (isConditioned) {
        recordedAttributeValue =
          dim.polarity === 'inverted'
            ? cVal > 0
              ? 'Clean-shaven / No beard'
              : 'Beard present'
            : cVal > 0
            ? `Affirmed (${dim.name})`
            : `Negated (${dim.name})`;
        evidenceState = cVal > 0 ? 'explicit' : 'negated';
      } else if (status === 'masked_uncertain' && categoryTrace) {
        recordedAttributeValue = categoryTrace.effectiveValue;
        evidenceState = 'uncertain';
      }

      let explanation = '';
      if (isConditioned) {
        if (dim.polarity === 'inverted') {
          explanation =
            cVal >= 0
              ? `Conditioned (c = +${cVal.toFixed(2)}, m = 1.0) on inverted No_Beard dimension to enforce absence of beard / clean-shaven appearance.`
              : `Conditioned (c = ${cVal.toFixed(2)}, m = 1.0) on inverted No_Beard dimension to suppress clean-shaven prior (beard present).`;
        } else {
          explanation =
            cVal >= 0
              ? `Conditioned (c = +${cVal.toFixed(2)}, m = 1.0) on direct dimension "${dim.name}" to affirm feature.`
              : `Conditioned (c = ${cVal.toFixed(2)}, m = 1.0) on direct dimension "${dim.name}" to actively suppress negated feature.`;
        }
        if (isContinuouslyAdjusted) {
          explanation += ` Modified via Phase 5.7 continuous GCIT traversal.`;
        }
        if (latentTraversalAlpha !== null) {
          explanation += ` Latent hyperplane traversal applied (alpha = ${latentTraversalAlpha >= 0 ? '+' : ''}${latentTraversalAlpha.toFixed(2)}).`;
        }
      } else if (status === 'masked_uncertain') {
        explanation = `Witness expressed epistemic uncertainty ("${recordedAttributeValue || 'hedged'}"). Masked to c = 0.0, m = 0.0 to prevent cognitive bias. Sampled from empirical dataset prior; NOT guaranteed to be absent.`;
      } else {
        explanation = `Unconditioned (c = 0.0, m = 0.0). Sampled stochastically from empirical dataset prior; NOT guaranteed to be absent or suppressed.`;
        if (latentTraversalAlpha !== null) {
          explanation += ` Note: Latent hyperplane traversal applied in Z space (alpha = ${latentTraversalAlpha >= 0 ? '+' : ''}${latentTraversalAlpha.toFixed(2)}).`;
        }
      }

      return {
        variationKey: item.variationKey,
        compositeId: item.compositeId,
        version: item.version,
        variationIndex: item.variationIndex,
        conditionValue: cVal,
        maskValue: mVal,
        status,
        recordedAttributeValue,
        evidenceState,
        isContinuouslyAdjusted,
        latentTraversalAlpha,
        explanation,
      };
    });

    const cValues = perVariation.map((p) => p.conditionValue);
    const mValues = perVariation.map((p) => p.maskValue);
    const latValues = perVariation.map((p) => p.latentTraversalAlpha ?? 0);
    const statuses = new Set(perVariation.map((p) => p.status));
    const recVals = new Set(perVariation.map((p) => p.recordedAttributeValue || ''));

    const hasConditionDifference =
      cValues.length > 1 ? Math.max(...cValues) - Math.min(...cValues) > 1e-4 : false;
    const hasMaskDifference =
      mValues.length > 1 ? Math.max(...mValues) - Math.min(...mValues) > 1e-4 : false;
    const hasLatentDifference =
      latValues.length > 1 ? Math.max(...latValues) - Math.min(...latValues) > 1e-4 : false;
    const hasStateDifference = statuses.size > 1 || recVals.size > 1;
    const hasAnyDifference =
      hasConditionDifference || hasMaskDifference || hasLatentDifference || hasStateDifference;

    let differenceSummary = 'Identical across compared variations';
    if (hasAnyDifference) {
      const parts: string[] = [];
      if (hasConditionDifference) parts.push('condition value (c_i)');
      if (hasMaskDifference) parts.push('conditioning mask (m_i)');
      if (hasLatentDifference) parts.push('latent traversal (alpha)');
      if (hasStateDifference && !hasConditionDifference && !hasMaskDifference) {
        parts.push('recorded descriptor/state');
      }
      differenceSummary = `Differs in ${parts.join(', ')}: ` +
        perVariation
          .map(
            (p) =>
              `v${p.version}#${p.variationIndex} [c=${p.conditionValue >= 0 ? '+' : ''}${p.conditionValue.toFixed(1)}, m=${p.maskValue.toFixed(1)}${
                p.latentTraversalAlpha !== null ? `, alpha=${p.latentTraversalAlpha}` : ''
              }]`
          )
          .join(' vs ');
    }

    return {
      index: dim.index,
      dimensionName: dim.name,
      category: dim.category,
      categoryDisplayName: formatAttributeCategory(dim.category),
      description: dim.description,
      polarity: dim.polarity,
      hasConditionDifference,
      hasMaskDifference,
      hasLatentDifference,
      hasStateDifference,
      hasAnyDifference,
      differenceSummary,
      perVariation,
    };
  });

  // 6. Build Recorded Evidence & Unsupported / Uncertain Attribute Comparison
  const allCategoriesSet = new Set<string>();
  for (const item of resolvedItems) {
    for (const tr of item._trace) {
      if (tr.category) {
        allCategoriesSet.add(tr.category.toLowerCase());
      }
    }
  }
  // Ensure unsupported forensic modalities are always represented for investigator clarity
  allCategoriesSet.add('eyes');
  allCategoriesSet.add('distinguishing_marks');

  const recordedAttributesComparison = Array.from(allCategoriesSet).map((cat) => {
    const perVariation = resolvedItems.map((item) => {
      const tr = item._trace.find((t) => (t.category || '').toLowerCase() === cat);
      const val = tr?.effectiveValue ?? null;
      const origVal = tr?.originalValue ?? val;
      const state = tr?.evidenceState || 'unknown';
      const gcitStatus = tr?.gcitStatus || resolveGCITTraceForAttribute(cat, val, state);

      const isGlassesObservable =
        (cat === 'other_observable' || cat === 'accessories') &&
        (val || '').toLowerCase().includes('glass');
      const isSupportedCategory = SUPPORTED_GCIT_CATEGORIES.has(cat) || isGlassesObservable;

      // Check if overridden by continuous GCIT adjustments on this variation
      let effectiveC = gcitStatus.conditionValue ?? 0.0;
      let effectiveM = gcitStatus.maskValue ?? 0.0;
      let effectiveGcitDim = gcitStatus.gcitDimension ?? null;
      let effectiveGcitIdx = gcitStatus.gcitIndex ?? null;

      if (typeof effectiveGcitIdx === 'number' && effectiveGcitIdx >= 0 && effectiveGcitIdx < 20) {
        effectiveC = item.gcitVector[effectiveGcitIdx];
        effectiveM = item.gcitMask[effectiveGcitIdx];
      } else {
        // Check if any dimension in this category was activated via continuous adjustment
        const catDim = GCIT_20_DIMENSIONS.find(
          (d) => d.category === cat && item.gcitMask[d.index] >= 0.5
        );
        if (catDim) {
          effectiveGcitDim = catDim.name;
          effectiveGcitIdx = catDim.index;
          effectiveC = item.gcitVector[catDim.index];
          effectiveM = item.gcitMask[catDim.index];
        }
      }

      let classification:
        | 'conditioned'
        | 'masked_uncertain'
        | 'masked_unmentioned'
        | 'unsupported_attribute'
        | 'contradictory_blocked';

      if (state === 'contradictory') {
        classification = 'contradictory_blocked';
      } else if (!isSupportedCategory) {
        classification = 'unsupported_attribute';
      } else if (effectiveM >= 0.5) {
        classification = 'conditioned';
      } else if (state === 'uncertain') {
        classification = 'masked_uncertain';
      } else {
        classification = 'masked_unmentioned';
      }

      return {
        variationKey: item.variationKey,
        compositeId: item.compositeId,
        version: item.version,
        variationIndex: item.variationIndex,
        recordedValue: val,
        originalValue: origVal,
        evidenceState: state,
        classification,
        isSupportedInGCIT: isSupportedCategory,
        gcitDimension: effectiveGcitDim,
        gcitIndex: effectiveGcitIdx,
        conditionValue: effectiveC,
        maskValue: effectiveM,
        isCorrected: Boolean(tr?.isCorrected),
        rationale:
          tr?.correction?.reason ||
          gcitStatus.inclusionReason ||
          'Unconditioned; sampled from empirical training prior (not guaranteed absent).',
      };
    });

    const isSupportedInGCIT = perVariation.some((p) => p.isSupportedInGCIT);
    const valSet = new Set(perVariation.map((p) => `${p.recordedValue ?? ''}|${p.evidenceState}|${p.classification}|${p.conditionValue}|${p.maskValue}`));
    const hasDifference = valSet.size > 1;

    return {
      category: cat,
      categoryDisplayName: formatAttributeCategory(cat),
      isSupportedInGCIT,
      hasDifference,
      perVariation,
    };
  });

  // 7. Collect selection audit history across compared composites
  const selectionHistory: Array<Record<string, any>> = [];
  for (const [compId, compDoc] of compositeMap.entries()) {
    if (Array.isArray(compDoc.selectionHistory)) {
      for (const h of compDoc.selectionHistory) {
        const rawH = (h as any).toObject ? (h as any).toObject() : h;
        selectionHistory.push({
          compositeId: compId,
          version: compDoc.version,
          ...rawH,
        });
      }
    }
  }

  // Strip internal _trace before sending response
  const publicVariations = resolvedItems.map(({ _trace, ...rest }) => rest);

  const differingDimensions = dimensionComparison.filter((d) => d.hasAnyDifference);
  const differingRecordedAttributes = recordedAttributesComparison.filter((a) => a.hasDifference);
  const sameCompositeSession = uniqueCompositeIds.length === 1;
  const seedsCompared = publicVariations.map((v) => v.seed);
  const uniqueSeeds = Array.from(new Set(seedsCompared));

  let seedDifferenceNote = '';
  if (differingDimensions.length === 0 && uniqueSeeds.length > 1) {
    seedDifferenceNote =
      `All ${publicVariations.length} compared variations share identical 20-dimensional GCIT conditioning vectors and binary masks. ` +
      `Visual differences between these variations arise strictly from distinct stochastic latent draws z ~ N(0, I_100) ` +
      `(seeds: ${seedsCompared.join(', ')}) across unconditioned degrees of freedom.`;
  } else if (differingDimensions.length > 0 && uniqueSeeds.length === 1) {
    seedDifferenceNote =
      `Compared variations share the exact same random seed (${uniqueSeeds[0]}) while differing in ${differingDimensions.length} GCIT conditioning dimension(s): ` +
      `[${differingDimensions.map((d) => d.dimensionName).join(', ')}]. Visual differences isolate the effect of the conditioning/traversal changes.`;
  } else if (differingDimensions.length > 0) {
    seedDifferenceNote =
      `Compared variations differ across ${differingDimensions.length} GCIT conditioning dimension(s) ` +
      `([${differingDimensions.map((d) => d.dimensionName).join(', ')}]) and use seeds [${seedsCompared.join(', ')}].`;
  } else {
    seedDifferenceNote =
      `Compared variations share identical 20-dimensional GCIT conditioning and seed (${seedsCompared.join(', ')}).`;
  }

  // 8. Record comparison audit trail
  await recordAudit(
    'COMPOSITE_VARIATIONS_COMPARED',
    'success',
    userId,
    {
      caseId,
      comparedVariations: targets,
      totalVariationsCompared: publicVariations.length,
      sameCompositeSession,
      differingGCITDimensionsCount: differingDimensions.length,
      differingGCITDimensions: differingDimensions.map((d) => d.dimensionName),
    },
    input.clientIp,
    input.userAgent
  );

  return {
    caseId,
    comparedAt: new Date().toISOString(),
    variations: publicVariations,
    dimensionComparison,
    recordedAttributesComparison,
    selectionHistory,
    summary: {
      totalVariationsCompared: publicVariations.length,
      sameCompositeSession,
      compositeIds: uniqueCompositeIds,
      differingGCITDimensionsCount: differingDimensions.length,
      differingGCITDimensions: differingDimensions.map((d) => d.dimensionName),
      differingRecordedAttributesCount: differingRecordedAttributes.length,
      seedsCompared,
      seedDifferenceNote,
    },
    disclosures: {
      unconditionedPriorDisclosure:
        'Unconditioned or masked attributes (m_i = 0.0, c_i = 0.0) are sampled stochastically from the empirical CelebA training distribution prior. An unconditioned attribute is NOT guaranteed to be absent, nor is it actively suppressed. Only explicitly negated attributes (m_i = 1.0 with negative or inverted conditioning) are actively suppressed.',
      noPixelInferenceDisclosure:
        'All attribute values, conditioning states, and difference highlights are derived strictly from recorded witness testimony, audited investigator overrides, and input 20-D GCIT conditioning tensors. No post-hoc facial attribute inference from generated pixels is performed or fabricated.',
      nonBiometricSelectionDisclosure:
        'Marking a variation as preferred designates it as the primary investigative lead candidate for further review and forensic dossier export. A selected variation is NOT a verified identity, correct match, or biometric identification.',
    },
  };
};

