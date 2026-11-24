import { Router } from 'express';
import {
  createWitnessDescription,
  getWitnessDescriptionsByCase,
  getWitnessDescriptionById,
  updateWitnessDescription,
  deleteWitnessDescription,
  addCorrection,
  updateApprovalStatus,
  getWitnessDescriptionTrace,
} from '../controllers/witnessDescription.controller';
import { authenticate } from '../middlewares/auth.middleware';
import { validate } from '../middlewares/validate.middleware';
import {
  createWitnessDescriptionSchema,
  updateWitnessDescriptionSchema,
  witnessDescriptionIdParamSchema,
  listByCaseParamSchema,
} from '../validations/witnessDescription.validation';

const router = Router();

router.use(authenticate);

/**
 * @route   POST /api/v1/witness-descriptions
 * @desc    Create a new witness description linked to a case
 * @access  Private
 */
router.post(
  '/',
  validate(createWitnessDescriptionSchema),
  createWitnessDescription
);

/**
 * @route   GET /api/v1/witness-descriptions/case/:caseId
 * @desc    Get all witness descriptions for a specific case
 * @access  Private
 */
router.get(
  '/case/:caseId',
  validate(listByCaseParamSchema),
  getWitnessDescriptionsByCase
);

/**
 * @route   GET /api/v1/witness-descriptions/:id
 * @desc    Get a single witness description by ID
 * @access  Private
 */
router.get(
  '/:id',
  validate(witnessDescriptionIdParamSchema),
  getWitnessDescriptionById
);

/**
 * @route   PATCH /api/v1/witness-descriptions/:id
 * @desc    Update a witness description by ID
 * @access  Private
 */
router.patch(
  '/:id',
  validate(updateWitnessDescriptionSchema),
  updateWitnessDescription
);

/**
 * @route   DELETE /api/v1/witness-descriptions/:id
 * @desc    Delete a witness description by ID
 * @access  Private
 */
router.delete(
  '/:id',
  validate(witnessDescriptionIdParamSchema),
  deleteWitnessDescription
);

/**
 * @route   POST /api/v1/witness-descriptions/:id/corrections
 * @desc    Record investigator correction on extracted attributes
 * @access  Private
 */
router.post(
  '/:id/corrections',
  validate(witnessDescriptionIdParamSchema),
  addCorrection
);

/**
 * @route   POST /api/v1/witness-descriptions/:id/approve
 * @desc    Approve or reject witness analysis
 * @access  Private
 */
router.post(
  '/:id/approve',
  validate(witnessDescriptionIdParamSchema),
  updateApprovalStatus
);

/**
 * @route   GET /api/v1/witness-descriptions/:id/trace
 * @desc    Get complete evidence traceability report for witness description
 * @access  Private
 */
router.get(
  '/:id/trace',
  validate(witnessDescriptionIdParamSchema),
  getWitnessDescriptionTrace
);

export default router;