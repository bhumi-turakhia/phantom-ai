import { Request, Response } from 'express';
import { asyncHandler } from '../utils/asyncHandler';
import { ApiResponse } from '../utils/apiResponse';
import * as compositeService from '../services/composite.service';

/**
 * @route   POST /api/v1/cases/:caseId/composites/generate
 * @desc    Generates condition-consistent forensic composites using the Phase 5.4 Conditional WGAN-GP
 * @access  Private (Investigator / Supervisor / Admin with case access)
 */
export const generateComposite = asyncHandler(async (req: Request, res: Response) => {
  const { caseId } = req.params;
  const user = (req as any).user;

  const result = await compositeService.generateCompositeForCase(caseId, user.id, user.role, {
    witnessDescriptionId: req.body.witnessDescriptionId,
    seed: req.body.seed !== undefined ? Number(req.body.seed) : undefined,
    numVariations: req.body.numVariations !== undefined ? Number(req.body.numVariations) : 1,
    notes: req.body.notes,
    clientIp: req.ip || req.socket.remoteAddress,
    userAgent: req.get('user-agent'),
  });

  return ApiResponse.success(res, 'Forensic facial composite generated successfully.', result, 201);
});

/**
 * @route   GET /api/v1/cases/:caseId/composites
 * @desc    Lists all composite versions for a case
 * @access  Private
 */
export const getComposites = asyncHandler(async (req: Request, res: Response) => {
  const { caseId } = req.params;
  const user = (req as any).user;

  const composites = await compositeService.getCompositesForCase(caseId, user.id, user.role);

  return ApiResponse.success(res, 'Case composites retrieved successfully.', composites);
});

/**
 * @route   GET /api/v1/cases/:caseId/composites/:compositeId
 * @desc    Retrieves a single composite version by ID
 * @access  Private
 */
export const getCompositeById = asyncHandler(async (req: Request, res: Response) => {
  const { caseId, compositeId } = req.params;
  const user = (req as any).user;

  const composite = await compositeService.getCompositeById(caseId, compositeId, user.id, user.role);

  return ApiResponse.success(res, 'Composite retrieved successfully.', composite);
});

/**
 * @route   GET /api/v1/cases/:caseId/composites/:compositeId/image/:variationIndex
 * @desc    Streams authenticated composite variation image
 * @access  Private (Authenticated & Case Authorized)
 */
export const getCompositeImage = asyncHandler(async (req: Request, res: Response) => {
  const { caseId, compositeId, variationIndex } = req.params;
  const user = (req as any).user;

  const { filePath, mimeType } = await compositeService.getCompositeImageFile(
    caseId,
    compositeId,
    Number(variationIndex),
    user.id,
    user.role,
    req.ip || req.socket.remoteAddress,
    req.get('user-agent')
  );

  res.setHeader('Content-Type', mimeType);
  res.setHeader('Cache-Control', 'private, max-age=3600');
  return res.sendFile(filePath);
});

/**
 * @route   PATCH /api/v1/cases/:caseId/composites/:compositeId/select
 * @desc    Selects and approves a specific variation of a composite version
 * @access  Private (Investigator / Supervisor)
 */
export const selectVariation = asyncHandler(async (req: Request, res: Response) => {
  const { caseId, compositeId } = req.params;
  const user = (req as any).user;
  const { variationIndex, status = 'approved', notes } = req.body;

  const updated = await compositeService.selectCompositeVariation(
    caseId,
    compositeId,
    Number(variationIndex),
    user.id,
    user.role,
    status,
    notes,
    req.ip || req.socket.remoteAddress,
    req.get('user-agent')
  );

  return ApiResponse.success(res, 'Composite variation selected and approved successfully.', updated);
});

/**
 * @route   GET /api/v1/cases/latent-directions
 * @desc    Retrieves evaluation and support status of latent directions in R^{100}
 * @access  Private (Authenticated)
 */
export const getLatentDirections = asyncHandler(async (_req: Request, res: Response) => {
  const directions = await compositeService.getLatentDirections();
  return ApiResponse.success(res, 'Latent directions metadata retrieved successfully.', directions);
});

/**
 * @route   POST /api/v1/cases/:caseId/composites/:compositeId/edit
 * @desc    Executes controlled semantic editing via continuous GCIT adjustments and latent traversal
 * @access  Private (Investigator / Supervisor with case access)
 */
export const editComposite = asyncHandler(async (req: Request, res: Response) => {
  const { caseId, compositeId } = req.params;
  const user = (req as any).user;

  const effectiveSeed = req.body.seed !== undefined ? Number(req.body.seed) : Number(req.body.selectedSeed);
  let traversals = req.body.latentTraversals;
  if (traversals && !Array.isArray(traversals) && typeof traversals === 'object') {
    traversals = Object.entries(traversals).map(([attr, alpha]) => ({
      attribute: attr,
      alpha: Number(alpha),
    }));
  }

  const result = await compositeService.editCompositeForCase(caseId, compositeId, user.id, user.role, {
    seed: effectiveSeed,
    continuousAdjustments: req.body.continuousAdjustments,
    latentTraversals: traversals,
    notes: req.body.notes,
    clientIp: req.ip || req.socket.remoteAddress,
    userAgent: req.get('user-agent'),
  });

  return ApiResponse.success(res, 'Semantic editing completed and new composite version created.', result, 201);
});

/**
 * @route   GET /api/v1/cases/:caseId/composites/:compositeId/trace
 * @desc    Retrieves the complete evidence-to-generation traceability manifest for a composite version
 * @access  Private (Investigator / Supervisor / Admin with case access)
 */
export const getCompositeTrace = asyncHandler(async (req: Request, res: Response) => {
  const { caseId, compositeId } = req.params;
  const user = (req as any).user;

  const trace = await compositeService.getCompositeTrace(caseId, compositeId, user.id, user.role);

  return ApiResponse.success(res, 'Composite evidence traceability manifest retrieved successfully.', trace);
});

/**
 * @route   GET /api/v1/cases/:caseId/composites/:compositeId/compare
 * @desc    Compares variations within a composite generation session and analyzes 20-D GCIT differences
 * @access  Private (Investigator / Supervisor / Admin with case access)
 */
export const compareSingleCompositeVariations = asyncHandler(async (req: Request, res: Response) => {
  const { caseId, compositeId } = req.params;
  const user = (req as any).user;

  const variationsQuery = req.query.variations as string | undefined;
  let variationIndices: number[] | undefined;
  if (variationsQuery && variationsQuery.trim().length > 0) {
    variationIndices = variationsQuery
      .split(',')
      .map((s) => Number(s.trim()))
      .filter((n) => !Number.isNaN(n) && n >= 1);
  }

  const comparison = await compositeService.compareCompositeVariations(caseId, user.id, user.role, {
    compositeId,
    variationIndices,
    clientIp: req.ip || req.socket.remoteAddress,
    userAgent: req.get('user-agent'),
  });

  return ApiResponse.success(
    res,
    'Multi-variation comparison and GCIT attribute difference analysis retrieved successfully.',
    comparison
  );
});

/**
 * @route   POST /api/v1/cases/:caseId/composites/compare
 * @desc    Compares multiple selected variations within or across composite versions of the same case
 * @access  Private (Investigator / Supervisor / Admin with case access)
 */
export const compareCaseVariations = asyncHandler(async (req: Request, res: Response) => {
  const { caseId } = req.params;
  const user = (req as any).user;

  const comparison = await compositeService.compareCompositeVariations(caseId, user.id, user.role, {
    compositeId: req.body.compositeId,
    variationIndices: req.body.variationIndices,
    targets: req.body.targets,
    clientIp: req.ip || req.socket.remoteAddress,
    userAgent: req.get('user-agent'),
  });

  return ApiResponse.success(
    res,
    'Multi-variation comparison and GCIT attribute difference analysis retrieved successfully.',
    comparison
  );
});
