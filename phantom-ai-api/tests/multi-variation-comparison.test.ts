import { Types } from 'mongoose';
import * as compositeService from '../src/services/composite.service';
import { Composite } from '../src/models/Composite.model';
import { Case } from '../src/models/Case.model';
import { SecurityAudit } from '../src/models/SecurityAudit.model';

/**
 * Helper to create a Mongoose-like chainable query mock that supports arbitrary .populate() chains
 * and resolves to `doc` when awaited.
 */
function createChainablePopulateQuery<T>(doc: T) {
  const query: any = {
    populate: jest.fn(() => query),
    sort: jest.fn(() => query),
    then: (resolve: any, reject: any) => Promise.resolve(doc).then(resolve, reject),
  };
  return query;
}

describe('Phase 7.1 Multi-Variation Comparison and Attribute Difference Analysis Suite', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  // =========================================================================
  // AREA 1: Single-Composite Variation Retrieval & Seed Difference Analysis
  // =========================================================================
  describe('Area 1: Intra-Composite Multi-Variation Retrieval & Stochastic Seed Comparison', () => {
    it('compares multiple variations within a single composite session (identical 20-D GCIT conditioning, distinct seeds)', async () => {
      const caseId = new Types.ObjectId();
      const compositeId = new Types.ObjectId();
      const witnessId = new Types.ObjectId();
      const userId = new Types.ObjectId();

      const cVector = new Array(20).fill(0.0);
      const mVector = new Array(20).fill(0.0);
      // Conditioned on hair_black (index 0, +1.0), facial_no_beard (index 13, +1.0 inverted), glasses (index 19, -1.0 negated)
      cVector[0] = 1.0;
      mVector[0] = 1.0;
      cVector[13] = 1.0;
      mVector[13] = 1.0;
      cVector[19] = -1.0;
      mVector[19] = 1.0;

      const mockCompositeDoc: any = {
        _id: compositeId,
        case: caseId,
        witnessDescription: witnessId,
        version: 1,
        status: 'draft',
        editingType: 'initial',
        gcitVector: cVector,
        gcitMask: mVector,
        imageHashes: [
          '1111111111111111111111111111111111111111111111111111111111111111',
          '2222222222222222222222222222222222222222222222222222222222222222',
          '3333333333333333333333333333333333333333333333333333333333333333',
        ],
        variations: [
          {
            _id: new Types.ObjectId(),
            variationIndex: 1,
            variationRole: 'standard',
            seed: 42,
            storageKey: 'case1/v1_var1.png',
            imageUrl: '/uploads/composites/case1/v1_var1.png',
            mimeType: 'image/png',
            width: 64,
            height: 64,
            fileSizeBytes: 4096,
            sha256: '1111111111111111111111111111111111111111111111111111111111111111',
            gcitVector: cVector,
            gcitMask: mVector,
            isSelected: true,
          },
          {
            _id: new Types.ObjectId(),
            variationIndex: 2,
            variationRole: 'standard',
            seed: 43,
            storageKey: 'case1/v1_var2.png',
            imageUrl: '/uploads/composites/case1/v1_var2.png',
            mimeType: 'image/png',
            width: 64,
            height: 64,
            fileSizeBytes: 4120,
            sha256: '2222222222222222222222222222222222222222222222222222222222222222',
            gcitVector: cVector,
            gcitMask: mVector,
            isSelected: false,
          },
          {
            _id: new Types.ObjectId(),
            variationIndex: 3,
            variationRole: 'standard',
            seed: 44,
            storageKey: 'case1/v1_var3.png',
            imageUrl: '/uploads/composites/case1/v1_var3.png',
            mimeType: 'image/png',
            width: 64,
            height: 64,
            fileSizeBytes: 4088,
            sha256: '3333333333333333333333333333333333333333333333333333333333333333',
            gcitVector: cVector,
            gcitMask: mVector,
            isSelected: false,
          },
        ],
        evidenceTrace: [
          {
            attributeName: 'Hair Color',
            category: 'hair_color',
            effectiveValue: 'black',
            originalValue: 'black',
            evidenceState: 'explicit',
            originalEvidenceState: 'explicit',
            sourceSpan: 'black hair',
            startChar: 7,
            endChar: 17,
            surroundingContext: 'He had black hair, no beard, and no glasses.',
            extractedBy: 'char_bilstm_crf_phase3c2',
            confidence: null,
            isCorrected: false,
            gcitStatus: {
              isEligible: true,
              gcitDimension: 'hair_black',
              gcitIndex: 0,
              conditionValue: 1.0,
              maskValue: 1.0,
              polarity: 'normal',
              isIncluded: true,
              inclusionReason: 'Explicitly affirmed black hair.',
            },
          },
          {
            attributeName: 'Facial Hair',
            category: 'facial_hair',
            effectiveValue: 'clean-shaven',
            originalValue: 'no beard',
            evidenceState: 'negated',
            originalEvidenceState: 'negated',
            sourceSpan: 'no beard',
            startChar: 19,
            endChar: 27,
            surroundingContext: 'He had black hair, no beard, and no glasses.',
            extractedBy: 'char_bilstm_crf_phase3c2',
            confidence: null,
            isCorrected: false,
            gcitStatus: {
              isEligible: true,
              gcitDimension: 'facial_no_beard',
              gcitIndex: 13,
              conditionValue: 1.0,
              maskValue: 1.0,
              polarity: 'inverted',
              isIncluded: true,
              inclusionReason: 'Inverted polarity on facial_no_beard (+1.0 = No Beard).',
            },
          },
          {
            attributeName: 'Eyeglasses',
            category: 'glasses',
            effectiveValue: 'glasses',
            originalValue: 'glasses',
            evidenceState: 'negated',
            originalEvidenceState: 'negated',
            sourceSpan: 'no glasses',
            startChar: 33,
            endChar: 43,
            surroundingContext: 'He had black hair, no beard, and no glasses.',
            extractedBy: 'char_bilstm_crf_phase3c2',
            confidence: null,
            isCorrected: false,
            gcitStatus: {
              isEligible: true,
              gcitDimension: 'glasses',
              gcitIndex: 19,
              conditionValue: -1.0,
              maskValue: 1.0,
              polarity: 'normal',
              isIncluded: true,
              inclusionReason: 'Explicitly negated glasses (-1.0, mask 1.0).',
            },
          },
        ],
        selectionHistory: [],
        generatorMetadata: {
          modelName: 'ConditionalWGANGPGenerator',
          checkpointSha256: 'a1714389d0d1eddad10fae74a736ca41de5e230eada095e6868e5daeeb2a1964',
        },
        createdAt: new Date(),
      };

      jest.spyOn(Case, 'findOne').mockResolvedValue({
        _id: caseId,
        createdBy: userId,
        assignedTo: [userId],
        isDeleted: false,
      } as any);

      jest
        .spyOn(Composite, 'findOne')
        .mockImplementation(() => createChainablePopulateQuery(mockCompositeDoc));

      const auditSpy = jest.spyOn(SecurityAudit, 'create').mockResolvedValue({} as any);

      const result = await compositeService.compareCompositeVariations(
        caseId.toString(),
        userId.toString(),
        'investigator',
        { compositeId: compositeId.toString() }
      );

      expect(result.caseId).toBe(caseId.toString());
      expect(result.variations).toHaveLength(3);
      expect(result.summary.totalVariationsCompared).toBe(3);
      expect(result.summary.sameCompositeSession).toBe(true);
      expect(result.summary.differingGCITDimensionsCount).toBe(0);
      expect(result.summary.seedsCompared).toEqual([42, 43, 44]);
      expect(result.summary.seedDifferenceNote).toContain(
        'identical 20-dimensional GCIT conditioning vectors and binary masks'
      );
      expect(result.summary.seedDifferenceNote).toContain('seeds: 42, 43, 44');

      // Verify 20-D dimension matrix
      expect(result.dimensionComparison).toHaveLength(20);
      expect(result.dimensionComparison[0].dimensionName).toBe('hair_black');
      expect(result.dimensionComparison[0].hasAnyDifference).toBe(false);
      expect(result.dimensionComparison[0].perVariation[0].conditionValue).toBe(1.0);
      expect(result.dimensionComparison[0].perVariation[0].maskValue).toBe(1.0);

      // Verify inverted facial_no_beard dimension (index 13)
      expect(result.dimensionComparison[13].dimensionName).toBe('facial_no_beard');
      expect(result.dimensionComparison[13].polarity).toBe('inverted');
      expect(result.dimensionComparison[13].perVariation[0].conditionValue).toBe(1.0);
      expect(result.dimensionComparison[13].perVariation[0].maskValue).toBe(1.0);

      // Verify negated glasses dimension (index 19)
      expect(result.dimensionComparison[19].dimensionName).toBe('glasses');
      expect(result.dimensionComparison[19].perVariation[0].conditionValue).toBe(-1.0);
      expect(result.dimensionComparison[19].perVariation[0].maskValue).toBe(1.0);

      // Verify audit event recorded
      expect(auditSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          event: 'COMPOSITE_VARIATIONS_COMPARED',
          status: 'success',
        })
      );
    });

    it('supports subset variation filtering via variationIndices', async () => {
      const caseId = new Types.ObjectId();
      const compositeId = new Types.ObjectId();
      const userId = new Types.ObjectId();

      const mockCompositeDoc: any = {
        _id: compositeId,
        case: caseId,
        version: 1,
        status: 'draft',
        gcitVector: new Array(20).fill(0.0),
        gcitMask: new Array(20).fill(0.0),
        variations: [
          { variationIndex: 1, seed: 101, imageUrl: '/v1.png', fileSizeBytes: 1000, isSelected: true },
          { variationIndex: 2, seed: 102, imageUrl: '/v2.png', fileSizeBytes: 1000, isSelected: false },
          { variationIndex: 3, seed: 103, imageUrl: '/v3.png', fileSizeBytes: 1000, isSelected: false },
        ],
        evidenceTrace: [],
        createdAt: new Date(),
      };

      jest.spyOn(Case, 'findOne').mockResolvedValue({
        _id: caseId,
        createdBy: userId,
        assignedTo: [userId],
        isDeleted: false,
      } as any);
      jest
        .spyOn(Composite, 'findOne')
        .mockImplementation(() => createChainablePopulateQuery(mockCompositeDoc));
      jest.spyOn(SecurityAudit, 'create').mockResolvedValue({} as any);

      const result = await compositeService.compareCompositeVariations(
        caseId.toString(),
        userId.toString(),
        'investigator',
        {
          compositeId: compositeId.toString(),
          variationIndices: [1, 3],
        }
      );

      expect(result.variations).toHaveLength(2);
      expect(result.variations.map((v) => v.variationIndex)).toEqual([1, 3]);
      expect(result.summary.seedsCompared).toEqual([101, 103]);
    });
  });

  // =========================================================================
  // AREA 2: GCIT Dimension, Mask, and Latent Traversal Difference Analysis
  // =========================================================================
  describe('Area 2: GCIT Conditioning, Mask, and Semantic Editing Difference Detection', () => {
    it('detects exact GCIT condition, mask, and latent traversal differences between Semantic Edited (Var #1) and Baseline (Var #2)', async () => {
      const caseId = new Types.ObjectId();
      const compositeV2Id = new Types.ObjectId();
      const parentCompositeV1Id = new Types.ObjectId();
      const userId = new Types.ObjectId();

      const baseC = new Array(20).fill(0.0);
      const baseM = new Array(20).fill(0.0);
      baseC[0] = 1.0; // hair_black
      baseM[0] = 1.0;

      const editedC = [...baseC];
      const editedM = [...baseM];
      editedC[14] = 0.75; // nose_big adjusted to +0.75
      editedM[14] = 1.0; // nose_big unmasked -> conditioned
      editedC[16] = -0.5; // lips_big adjusted to -0.50
      editedM[16] = 1.0;

      const mockEditedCompositeDoc: any = {
        _id: compositeV2Id,
        case: caseId,
        parentComposite: parentCompositeV1Id,
        version: 2,
        status: 'draft',
        editingType: 'hybrid_traversal',
        gcitVector: editedC,
        gcitMask: editedM,
        traversalParameters: {
          continuousAdjustments: {
            nose_big: 0.75,
            lips_big: -0.5,
          },
          latentTraversals: [{ attribute: 'glasses', alpha: 1.5 }],
        },
        variations: [
          {
            variationIndex: 1,
            variationRole: 'semantic_edited',
            seed: 42,
            imageUrl: '/uploads/composites/v2_edited.png',
            fileSizeBytes: 4200,
            sha256: 'aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111',
            gcitVector: editedC,
            gcitMask: editedM,
            isSelected: true,
          },
          {
            variationIndex: 2,
            variationRole: 'semantic_baseline',
            seed: 42,
            imageUrl: '/uploads/composites/v2_baseline.png',
            fileSizeBytes: 4100,
            sha256: 'bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222',
            gcitVector: baseC,
            gcitMask: baseM,
            isSelected: false,
          },
        ],
        evidenceTrace: [],
        createdAt: new Date(),
      };

      const mockParentDoc: any = {
        _id: parentCompositeV1Id,
        case: caseId,
        version: 1,
        gcitVector: baseC,
        gcitMask: baseM,
      };

      jest.spyOn(Case, 'findOne').mockResolvedValue({
        _id: caseId,
        createdBy: userId,
        assignedTo: [userId],
        isDeleted: false,
      } as any);

      jest.spyOn(Composite, 'findOne').mockImplementation((filter: any) => {
        if (filter._id.toString() === compositeV2Id.toString()) {
          return createChainablePopulateQuery(mockEditedCompositeDoc);
        }
        return createChainablePopulateQuery(mockParentDoc);
      });
      jest.spyOn(SecurityAudit, 'create').mockResolvedValue({} as any);

      const result = await compositeService.compareCompositeVariations(
        caseId.toString(),
        userId.toString(),
        'investigator',
        { compositeId: compositeV2Id.toString() }
      );

      expect(result.variations[0].variationRole).toBe('semantic_edited');
      expect(result.variations[1].variationRole).toBe('semantic_baseline');

      // 3 dimensions differ: nose_big (14), lips_big (16), glasses (19 via latent alpha)
      expect(result.summary.differingGCITDimensionsCount).toBe(3);
      expect(result.summary.differingGCITDimensions).toEqual([
        'nose_big',
        'lips_big',
        'glasses',
      ]);
      expect(result.summary.seedDifferenceNote).toContain(
        'share the exact same random seed (42) while differing in 3 GCIT conditioning dimension(s)'
      );

      // Inspect nose_big (index 14)
      const noseRow = result.dimensionComparison[14];
      expect(noseRow.dimensionName).toBe('nose_big');
      expect(noseRow.hasConditionDifference).toBe(true);
      expect(noseRow.hasMaskDifference).toBe(true);
      expect(noseRow.hasAnyDifference).toBe(true);
      expect(noseRow.perVariation[0].conditionValue).toBe(0.75);
      expect(noseRow.perVariation[0].maskValue).toBe(1.0);
      expect(noseRow.perVariation[1].conditionValue).toBe(0.0);
      expect(noseRow.perVariation[1].maskValue).toBe(0.0);

      // Inspect glasses (index 19) latent traversal difference
      const glassesRow = result.dimensionComparison[19];
      expect(glassesRow.dimensionName).toBe('glasses');
      expect(glassesRow.hasLatentDifference).toBe(true);
      expect(glassesRow.perVariation[0].latentTraversalAlpha).toBe(1.5);
      expect(glassesRow.perVariation[1].latentTraversalAlpha).toBeNull();
    });

    it('compares variations across distinct composite versions (v1 vs v2) within the same case', async () => {
      const caseId = new Types.ObjectId();
      const compV1Id = new Types.ObjectId();
      const compV2Id = new Types.ObjectId();
      const userId = new Types.ObjectId();

      const cV1 = new Array(20).fill(0.0);
      const mV1 = new Array(20).fill(0.0);
      cV1[0] = 1.0; // hair_black
      mV1[0] = 1.0;

      const cV2 = new Array(20).fill(0.0);
      const mV2 = new Array(20).fill(0.0);
      cV2[1] = 1.0; // hair_brown (corrected in v2)
      mV2[1] = 1.0;

      const mockV1: any = {
        _id: compV1Id,
        case: caseId,
        version: 1,
        status: 'superseded',
        gcitVector: cV1,
        gcitMask: mV1,
        variations: [
          { variationIndex: 1, seed: 42, imageUrl: '/v1_1.png', fileSizeBytes: 1000, gcitVector: cV1, gcitMask: mV1 },
        ],
        evidenceTrace: [],
        createdAt: new Date(),
      };

      const mockV2: any = {
        _id: compV2Id,
        case: caseId,
        version: 2,
        status: 'approved',
        gcitVector: cV2,
        gcitMask: mV2,
        variations: [
          { variationIndex: 1, seed: 99, imageUrl: '/v2_1.png', fileSizeBytes: 1000, gcitVector: cV2, gcitMask: mV2 },
        ],
        evidenceTrace: [],
        createdAt: new Date(),
      };

      jest.spyOn(Case, 'findOne').mockResolvedValue({
        _id: caseId,
        createdBy: userId,
        assignedTo: [userId],
        isDeleted: false,
      } as any);

      jest.spyOn(Composite, 'findOne').mockImplementation((filter: any) => {
        if (filter._id.toString() === compV1Id.toString()) {
          return createChainablePopulateQuery(mockV1);
        }
        return createChainablePopulateQuery(mockV2);
      });
      jest.spyOn(SecurityAudit, 'create').mockResolvedValue({} as any);

      const result = await compositeService.compareCompositeVariations(
        caseId.toString(),
        userId.toString(),
        'investigator',
        {
          targets: [
            { compositeId: compV1Id.toString(), variationIndex: 1 },
            { compositeId: compV2Id.toString(), variationIndex: 1 },
          ],
        }
      );

      expect(result.summary.sameCompositeSession).toBe(false);
      expect(result.summary.compositeIds).toEqual([compV1Id.toString(), compV2Id.toString()]);
      expect(result.summary.differingGCITDimensionsCount).toBe(2);
      expect(result.summary.differingGCITDimensions).toEqual(['hair_black', 'hair_brown']);
    });

    it('clamps continuous GCIT adjustments to [-1.0, +1.0] and sets mask m_i = 1.0 via applyContinuousAdjustmentsToGCIT', () => {
      const baseC = new Array(20).fill(0.0);
      const baseM = new Array(20).fill(0.0);

      const { cVector, mVector } = compositeService.applyContinuousAdjustmentsToGCIT(
        baseC,
        baseM,
        {
          nose_big: 2.5, // clamped to +1.0
          lips_big: -3.0, // clamped to -1.0
          face_oval: 0.45,
        }
      );

      expect(cVector[14]).toBe(1.0);
      expect(mVector[14]).toBe(1.0);
      expect(cVector[16]).toBe(-1.0);
      expect(mVector[16]).toBe(1.0);
      expect(cVector[17]).toBe(0.45);
      expect(mVector[17]).toBe(1.0);
      // Unadjusted dimensions remain 0.0
      expect(cVector[0]).toBe(0.0);
      expect(mVector[0]).toBe(0.0);
    });
  });

  // =========================================================================
  // AREA 3: Missing, Uncertain, Unsupported Attributes & Mandatory Disclosures
  // =========================================================================
  describe('Area 3: Uncertain, Missing, Unsupported Attributes and Forensic Disclosures', () => {
    it('correctly classifies conditioned, masked_uncertain, masked_unmentioned, and unsupported_attribute categories', async () => {
      const caseId = new Types.ObjectId();
      const compositeId = new Types.ObjectId();
      const userId = new Types.ObjectId();

      const cVector = new Array(20).fill(0.0);
      const mVector = new Array(20).fill(0.0);
      cVector[2] = 1.0; // hair_blond conditioned
      mVector[2] = 1.0;

      const mockCompositeDoc: any = {
        _id: compositeId,
        case: caseId,
        version: 1,
        status: 'draft',
        gcitVector: cVector,
        gcitMask: mVector,
        variations: [
          { variationIndex: 1, seed: 10, imageUrl: '/v1.png', fileSizeBytes: 1000 },
          { variationIndex: 2, seed: 11, imageUrl: '/v2.png', fileSizeBytes: 1000 },
        ],
        evidenceTrace: [
          {
            attributeName: 'Hair Color',
            category: 'hair_color',
            effectiveValue: 'blond',
            originalValue: 'blond',
            evidenceState: 'explicit',
            originalEvidenceState: 'explicit',
            sourceSpan: 'blond hair',
            startChar: 0,
            endChar: 10,
            surroundingContext: 'Blond hair, maybe a mustache, blue eyes, scar on cheek.',
            extractedBy: 'char_bilstm_crf_phase3c2',
            confidence: null,
            isCorrected: false,
            gcitStatus: {
              isEligible: true,
              gcitDimension: 'hair_blond',
              gcitIndex: 2,
              conditionValue: 1.0,
              maskValue: 1.0,
              polarity: 'normal',
              isIncluded: true,
              inclusionReason: 'Conditioned on hair_blond.',
            },
          },
          {
            attributeName: 'Facial Hair',
            category: 'facial_hair',
            effectiveValue: 'mustache',
            originalValue: 'mustache',
            evidenceState: 'uncertain',
            originalEvidenceState: 'uncertain',
            sourceSpan: 'maybe a mustache',
            startChar: 12,
            endChar: 28,
            surroundingContext: 'Blond hair, maybe a mustache, blue eyes, scar on cheek.',
            extractedBy: 'char_bilstm_crf_phase3c2',
            confidence: null,
            isCorrected: false,
            gcitStatus: {
              isEligible: true,
              gcitDimension: 'facial_mustache',
              gcitIndex: 9,
              conditionValue: 0.0,
              maskValue: 0.0,
              polarity: 'normal',
              isIncluded: false,
              inclusionReason: 'Masked due to epistemic uncertainty.',
            },
          },
          {
            attributeName: 'Eyes',
            category: 'eyes',
            effectiveValue: 'blue eyes',
            originalValue: 'blue eyes',
            evidenceState: 'explicit',
            originalEvidenceState: 'explicit',
            sourceSpan: 'blue eyes',
            startChar: 30,
            endChar: 39,
            surroundingContext: 'Blond hair, maybe a mustache, blue eyes, scar on cheek.',
            extractedBy: 'char_bilstm_crf_phase3c2',
            confidence: null,
            isCorrected: false,
            gcitStatus: {
              isEligible: false,
              gcitDimension: null,
              gcitIndex: null,
              conditionValue: 0.0,
              maskValue: 0.0,
              polarity: 'normal',
              isIncluded: false,
              inclusionReason: 'Unsupported in 20-D GCIT.',
            },
          },
          {
            attributeName: 'Distinguishing Marks',
            category: 'distinguishing_marks',
            effectiveValue: 'scar on cheek',
            originalValue: 'scar on cheek',
            evidenceState: 'explicit',
            originalEvidenceState: 'explicit',
            sourceSpan: 'scar on cheek',
            startChar: 41,
            endChar: 54,
            surroundingContext: 'Blond hair, maybe a mustache, blue eyes, scar on cheek.',
            extractedBy: 'char_bilstm_crf_phase3c2',
            confidence: null,
            isCorrected: false,
            gcitStatus: {
              isEligible: false,
              gcitDimension: null,
              gcitIndex: null,
              conditionValue: 0.0,
              maskValue: 0.0,
              polarity: 'normal',
              isIncluded: false,
              inclusionReason: 'Unsupported in 20-D GCIT.',
            },
          },
        ],
        createdAt: new Date(),
      };

      jest.spyOn(Case, 'findOne').mockResolvedValue({
        _id: caseId,
        createdBy: userId,
        assignedTo: [userId],
        isDeleted: false,
      } as any);
      jest
        .spyOn(Composite, 'findOne')
        .mockImplementation(() => createChainablePopulateQuery(mockCompositeDoc));
      jest.spyOn(SecurityAudit, 'create').mockResolvedValue({} as any);

      const result = await compositeService.compareCompositeVariations(
        caseId.toString(),
        userId.toString(),
        'investigator',
        { compositeId: compositeId.toString() }
      );

      const hairAttr = result.recordedAttributesComparison.find(
        (r) => r.category === 'hair_color'
      );
      expect(hairAttr?.isSupportedInGCIT).toBe(true);
      expect(hairAttr?.perVariation[0].classification).toBe('conditioned');

      const mustacheAttr = result.recordedAttributesComparison.find(
        (r) => r.category === 'facial_hair'
      );
      expect(mustacheAttr?.isSupportedInGCIT).toBe(true);
      expect(mustacheAttr?.perVariation[0].classification).toBe('masked_uncertain');
      expect(mustacheAttr?.perVariation[0].maskValue).toBe(0.0);

      // Verify facial_mustache (index 9) in dimensionComparison is marked masked_uncertain
      expect(result.dimensionComparison[9].perVariation[0].status).toBe('masked_uncertain');

      // Verify unsupported attributes (eyes, distinguishing_marks)
      const eyesAttr = result.recordedAttributesComparison.find((r) => r.category === 'eyes');
      expect(eyesAttr?.isSupportedInGCIT).toBe(false);
      expect(eyesAttr?.perVariation[0].classification).toBe('unsupported_attribute');

      const marksAttr = result.recordedAttributesComparison.find(
        (r) => r.category === 'distinguishing_marks'
      );
      expect(marksAttr?.isSupportedInGCIT).toBe(false);
      expect(marksAttr?.perVariation[0].classification).toBe('unsupported_attribute');

      // Verify mandatory disclosures
      expect(result.disclosures.unconditionedPriorDisclosure).toContain(
        'NOT guaranteed to be absent'
      );
      expect(result.disclosures.noPixelInferenceDisclosure).toContain(
        'No post-hoc facial attribute inference from generated pixels is performed'
      );
      expect(result.disclosures.nonBiometricSelectionDisclosure).toContain(
        'NOT a verified identity'
      );
    });
  });

  // =========================================================================
  // AREA 4 & 5: Preferred Variation Selection, Audit Trail & Non-Destructive Preservation
  // =========================================================================
  describe('Area 4 & 5: Preferred Variation Selection and Selection Audit History', () => {
    it('records preferred variation selection, appends to selectionHistory, logs SecurityAudit, and preserves all other variations and hashes', async () => {
      const caseId = new Types.ObjectId();
      const compositeId = new Types.ObjectId();
      const userId = new Types.ObjectId();

      const var1Id = new Types.ObjectId();
      const var2Id = new Types.ObjectId();

      const mockCompositeDoc: any = {
        _id: compositeId,
        case: caseId,
        version: 1,
        status: 'draft',
        imageHashes: [
          'aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000',
          'bbbb1111bbbb1111bbbb1111bbbb1111bbbb1111bbbb1111bbbb1111bbbb1111',
        ],
        variations: [
          {
            _id: var1Id,
            variationIndex: 1,
            seed: 42,
            imageUrl: '/uploads/composites/var1.png',
            sha256: 'aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000',
            isSelected: true,
          },
          {
            _id: var2Id,
            variationIndex: 2,
            seed: 43,
            imageUrl: '/uploads/composites/var2.png',
            sha256: 'bbbb1111bbbb1111bbbb1111bbbb1111bbbb1111bbbb1111bbbb1111bbbb1111',
            isSelected: false,
          },
        ],
        selectionHistory: [],
        save: jest.fn().mockResolvedValue(true),
        populate: jest.fn().mockResolvedValue(true),
      };

      jest.spyOn(Case, 'findOne').mockResolvedValue({
        _id: caseId,
        createdBy: userId,
        assignedTo: [userId],
        isDeleted: false,
      } as any);
      jest.spyOn(Composite, 'findOne').mockResolvedValue(mockCompositeDoc);
      const auditSpy = jest.spyOn(SecurityAudit, 'create').mockResolvedValue({} as any);

      const updated = await compositeService.selectCompositeVariation(
        caseId.toString(),
        compositeId.toString(),
        2,
        userId.toString(),
        'investigator',
        'approved',
        'Variation #2 best aligns with witness jawline recollection.'
      );

      // Selected variation updated to #2
      expect(updated.status).toBe('approved');
      expect(updated.reviewedBy?.toString()).toBe(userId.toString());
      expect(updated.reviewedAt).toBeInstanceOf(Date);

      // Non-selected variation #1 is preserved with its original image & sha256
      expect(updated.variations).toHaveLength(2);
      expect(updated.variations[0].variationIndex).toBe(1);
      expect(updated.variations[0].isSelected).toBe(false);
      expect(updated.variations[0].sha256).toBe(
        'aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000'
      );

      // Variation #2 is marked selected with investigator attribution
      expect(updated.variations[1].variationIndex).toBe(2);
      expect(updated.variations[1].isSelected).toBe(true);
      expect(updated.variations[1].approvedBy?.toString()).toBe(userId.toString());
      expect(updated.variations[1].approvedAt).toBeInstanceOf(Date);
      expect(updated.variations[1].selectionNotes).toBe(
        'Variation #2 best aligns with witness jawline recollection.'
      );

      // Selection history records the transition from Var #1 -> Var #2
      expect(updated.selectionHistory).toHaveLength(1);
      expect(updated.selectionHistory![0].variationIndex).toBe(2);
      expect(updated.selectionHistory![0].previousVariationIndex).toBe(1);
      expect(updated.selectionHistory![0].seed).toBe(43);
      expect(updated.selectionHistory![0].selectedBy.toString()).toBe(userId.toString());
      expect(updated.selectionHistory![0].notes).toBe(
        'Variation #2 best aligns with witness jawline recollection.'
      );

      // SecurityAudit recorded COMPOSITE_VARIATION_SELECTED
      expect(auditSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          event: 'COMPOSITE_VARIATION_SELECTED',
          status: 'success',
        })
      );

      // Re-selecting Variation #1 appends a second immutable audit record without deleting the first
      const updatedAgain = await compositeService.selectCompositeVariation(
        caseId.toString(),
        compositeId.toString(),
        1,
        userId.toString(),
        'investigator',
        'approved',
        'Re-selected Variation #1 after secondary supervisor review.'
      );

      expect(updatedAgain.selectionHistory).toHaveLength(2);
      expect(updatedAgain.selectionHistory![0].variationIndex).toBe(2);
      expect(updatedAgain.selectionHistory![1].variationIndex).toBe(1);
      expect(updatedAgain.selectionHistory![1].previousVariationIndex).toBe(2);
      expect(updatedAgain.variations[0].isSelected).toBe(true);
      expect(updatedAgain.variations[1].isSelected).toBe(false);
    });

    it('classifies contradictory attributes as contradictory_blocked and keeps them masked', async () => {
      const caseId = new Types.ObjectId();
      const compositeId = new Types.ObjectId();
      const userId = new Types.ObjectId();

      const mockCompositeDoc: any = {
        _id: compositeId,
        case: caseId,
        version: 1,
        status: 'draft',
        gcitVector: new Array(20).fill(0.0),
        gcitMask: new Array(20).fill(0.0),
        variations: [
          { variationIndex: 1, seed: 50, imageUrl: '/v1.png', fileSizeBytes: 1000 },
        ],
        evidenceTrace: [
          {
            attributeName: 'Hair Color',
            category: 'hair_color',
            effectiveValue: 'black vs blond',
            originalValue: 'black vs blond',
            evidenceState: 'contradictory',
            originalEvidenceState: 'contradictory',
            sourceSpan: 'black hair ... blond hair',
            startChar: 0,
            endChar: 20,
            surroundingContext: 'He had black hair or blond hair.',
            extractedBy: 'char_bilstm_crf_phase3c2',
            confidence: null,
            isCorrected: false,
            gcitStatus: {
              isEligible: true,
              gcitDimension: null,
              gcitIndex: null,
              conditionValue: 0.0,
              maskValue: 0.0,
              polarity: 'normal',
              isIncluded: false,
              inclusionReason: 'Contradictory testimony blocked.',
            },
          },
        ],
        createdAt: new Date(),
      };

      jest.spyOn(Case, 'findOne').mockResolvedValue({
        _id: caseId,
        createdBy: userId,
        assignedTo: [userId],
        isDeleted: false,
      } as any);
      jest
        .spyOn(Composite, 'findOne')
        .mockImplementation(() => createChainablePopulateQuery(mockCompositeDoc));
      jest.spyOn(SecurityAudit, 'create').mockResolvedValue({} as any);

      const result = await compositeService.compareCompositeVariations(
        caseId.toString(),
        userId.toString(),
        'investigator',
        { compositeId: compositeId.toString() }
      );

      const hairRow = result.recordedAttributesComparison.find(
        (r) => r.category === 'hair_color'
      );
      expect(hairRow?.perVariation[0].classification).toBe('contradictory_blocked');
      expect(hairRow?.perVariation[0].maskValue).toBe(0.0);
    });
  });

  // =========================================================================
  // AREA 6: RBAC Authorization & Cross-Case Access Prevention
  // =========================================================================
  describe('Area 6: Authorization & Cross-Case Access Prevention', () => {
    it('blocks unassigned investigators from comparing variations (403 Forbidden)', async () => {
      const caseId = new Types.ObjectId();
      const assignedUserId = new Types.ObjectId();
      const unauthorizedUserId = new Types.ObjectId();

      jest.spyOn(Case, 'findOne').mockResolvedValue({
        _id: caseId,
        createdBy: assignedUserId,
        assignedTo: [assignedUserId],
        isDeleted: false,
      } as any);

      await expect(
        compositeService.compareCompositeVariations(
          caseId.toString(),
          unauthorizedUserId.toString(),
          'investigator',
          { compositeId: new Types.ObjectId().toString() }
        )
      ).rejects.toThrow('Forbidden: You are not authorized to view or modify this case file.');
    });

    it('prevents cross-case composite comparison when a compositeId belongs to a different case (404 Not Found)', async () => {
      const caseAId = new Types.ObjectId();
      const compositeFromCaseBId = new Types.ObjectId();
      const userId = new Types.ObjectId();

      jest.spyOn(Case, 'findOne').mockResolvedValue({
        _id: caseAId,
        createdBy: userId,
        assignedTo: [userId],
        isDeleted: false,
      } as any);

      // Composite.findOne({ _id: compositeFromCaseBId, case: caseAId }) returns null
      jest
        .spyOn(Composite, 'findOne')
        .mockImplementation(() => createChainablePopulateQuery(null));

      await expect(
        compositeService.compareCompositeVariations(
          caseAId.toString(),
          userId.toString(),
          'investigator',
          {
            targets: [{ compositeId: compositeFromCaseBId.toString(), variationIndex: 1 }],
          }
        )
      ).rejects.toThrow(
        `Composite ${compositeFromCaseBId.toString()} not found for case ${caseAId.toString()}.`
      );
    });

    it('rejects invalid composite IDs and out-of-range variation indices with appropriate errors', async () => {
      const caseId = new Types.ObjectId();
      const userId = new Types.ObjectId();

      jest.spyOn(Case, 'findOne').mockResolvedValue({
        _id: caseId,
        createdBy: userId,
        assignedTo: [userId],
        isDeleted: false,
      } as any);

      await expect(
        compositeService.compareCompositeVariations(
          caseId.toString(),
          userId.toString(),
          'investigator',
          { compositeId: 'not-a-valid-objectid' }
        )
      ).rejects.toThrow('Invalid composite ID format.');

      await expect(
        compositeService.compareCompositeVariations(
          caseId.toString(),
          userId.toString(),
          'investigator',
          {
            targets: [{ compositeId: new Types.ObjectId().toString(), variationIndex: 0 }],
          }
        )
      ).rejects.toThrow('Invalid variation index: 0. Must be a positive integer.');
    });
  });
});
