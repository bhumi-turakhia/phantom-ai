import { Router } from 'express';
import {
  generateComposite,
  getComposites,
  getCompositeById,
  getCompositeImage,
  selectVariation,
  editComposite,
  getLatentDirections,
  getCompositeTrace,
  compareSingleCompositeVariations,
  compareCaseVariations,
} from '../controllers/composite.controller';
import { authenticate } from '../middlewares/auth.middleware';
import { validate } from '../middlewares/validate.middleware';
import {
  generateCompositeSchema,
  caseIdParamsSchema,
  compositeIdParamsSchema,
  compositeImageParamsSchema,
  selectVariationSchema,
  editCompositeSchema,
  compareSingleCompositeSchema,
  compareCaseVariationsSchema,
} from '../validations/composite.validation';

const router = Router({ mergeParams: true });

router.use(authenticate);

/**
 * @route   GET /api/v1/cases/:caseId/composites/latent-directions
 * @desc    Retrieves evaluation and support status of latent directions in R^{100}
 * @access  Private (Authenticated)
 */
router.get('/latent-directions', getLatentDirections);

/**
 * @route   POST /api/v1/cases/:caseId/composites/generate
 * @desc    Generates conditional composite variations using the Phase 5.4 WGAN-GP generator
 * @access  Private (Authenticated & Authorized Case Investigator)
 */
router.post('/generate', validate(generateCompositeSchema), generateComposite);

/**
 * @route   POST /api/v1/cases/:caseId/composites/compare
 * @desc    Compares selected variations within or across composite versions of the same case
 * @access  Private (Authenticated & Authorized Case Investigator)
 */
router.post('/compare', validate(compareCaseVariationsSchema), compareCaseVariations);

/**
 * @route   POST /api/v1/cases/:caseId/composites/:compositeId/edit
 * @desc    Executes controlled semantic editing via continuous GCIT adjustments and latent traversal
 * @access  Private (Authenticated & Authorized Case Investigator)
 */
router.post('/:compositeId/edit', validate(editCompositeSchema), editComposite);

/**
 * @route   GET /api/v1/cases/:caseId/composites
 * @desc    Retrieves all composite versions for a case
 * @access  Private
 */
router.get('/', validate(caseIdParamsSchema), getComposites);

/**
 * @route   GET /api/v1/cases/:caseId/composites/:compositeId/compare
 * @desc    Compares variations within a composite generation session and analyzes 20-D GCIT differences
 * @access  Private
 */
router.get(
  '/:compositeId/compare',
  validate(compareSingleCompositeSchema),
  compareSingleCompositeVariations
);

/**
 * @route   GET /api/v1/cases/:caseId/composites/:compositeId
 * @desc    Retrieves a single composite version by ID
 * @access  Private
 */
router.get('/:compositeId', validate(compositeIdParamsSchema), getCompositeById);

/**
 * @route   GET /api/v1/cases/:caseId/composites/:compositeId/trace
 * @desc    Retrieves the complete evidence-to-generation traceability manifest for a composite
 * @access  Private
 */
router.get('/:compositeId/trace', validate(compositeIdParamsSchema), getCompositeTrace);

/**
 * @route   GET /api/v1/cases/:caseId/composites/:compositeId/image/:variationIndex
 * @desc    Streams authenticated composite variation image
 * @access  Private (Authenticated & Case Authorized)
 */
router.get(
  '/:compositeId/image/:variationIndex',
  validate(compositeImageParamsSchema),
  getCompositeImage
);

/**
 * @route   PATCH /api/v1/cases/:caseId/composites/:compositeId/select
 * @desc    Selects and approves a specific variation of a composite version
 * @access  Private
 */
router.patch('/:compositeId/select', validate(selectVariationSchema), selectVariation);

export default router;

