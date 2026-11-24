import { Types } from 'mongoose';
import {
  extractSurroundingSentence,
  formatAttributeCategory,
  resolveGCITTraceForAttribute,
  buildGCITTensorsFromTrace,
  GCIT_20_DIMENSIONS,
} from '../src/services/gcitTrace.service';
import * as witnessDescriptionService from '../src/services/witnessDescription.service';
import * as compositeService from '../src/services/composite.service';
import { WitnessDescription } from '../src/models/WitnessDescription.model';
import { Composite } from '../src/models/Composite.model';
import { Case } from '../src/models/Case.model';
import { SecurityAudit } from '../src/models/SecurityAudit.model';
import { mlClient } from '../src/services/mlClient.service';

describe('Phase 7.0 Evidence Traceability and Explainable AI (XAI) Suite', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.spyOn(Case, 'findOne').mockResolvedValue({
      _id: new Types.ObjectId(),
      status: 'open',
      isDeleted: false,
      createdBy: new Types.ObjectId(),
      assignedTo: [],
    } as any);
  });

  // =========================================================================
  // AREA 1: Exact Source-Span Traceability & Surrounding Context Extraction
  // =========================================================================
  describe('Area 1 & 2: Source-Span Traceability and Sentence Boundary Extraction', () => {
    const testimony =
      'The suspect was tall. He had dark wavy hair and a prominent sharp nose! He was definitely clean-shaven. He wore a blue hoodie.';

    it('extracts the exact surrounding grammatical sentence window for a valid character span', () => {
      // "sharp nose" is inside the second sentence
      const start = testimony.indexOf('sharp nose');
      const end = start + 'sharp nose'.length;

      const sentence = extractSurroundingSentence(testimony, start, end);
      expect(sentence).toBe('He had dark wavy hair and a prominent sharp nose!');
      expect(sentence).toContain('sharp nose');
    });

    it('extracts the sentence boundary properly when ended by a period', () => {
      const start = testimony.indexOf('clean-shaven');
      const end = start + 'clean-shaven'.length;

      const sentence = extractSurroundingSentence(testimony, start, end);
      expect(sentence).toBe('He was definitely clean-shaven.');
    });

    it('gracefully handles out-of-range offsets without throwing errors', () => {
      // start > end
      expect(() => extractSurroundingSentence(testimony, 50, 20)).not.toThrow();
      const res1 = extractSurroundingSentence(testimony, 50, 20);
      expect(typeof res1).toBe('string');

      // negative start
      expect(() => extractSurroundingSentence(testimony, -10, 20)).not.toThrow();

      // end beyond statement length
      expect(() => extractSurroundingSentence(testimony, 10, 999999)).not.toThrow();

      // null offsets fallback to full statement
      const resNull = extractSurroundingSentence(testimony, null, null);
      expect(resNull).toBe(testimony.trim());
    });

    it('formats category slugs into human-readable forensic labels', () => {
      expect(formatAttributeCategory('facial_hair')).toBe('Facial Hair');
      expect(formatAttributeCategory('face_shape')).toBe('Face Shape');
      expect(formatAttributeCategory('distinguishing_marks')).toBe('Distinguishing Marks');
      expect(formatAttributeCategory('other_observable')).toBe('Accessories & Observables');
    });
  });

  // =========================================================================
  // AREA 3 & 4: GCIT Mapping, Dimension Indexing, and Polarity Handling
  // =========================================================================
  describe('Area 3 & 4: GCIT Dimension Mapping and Polarity Resolution', () => {
    it('maps all 20 GCIT dimensions with correct indices and categories', () => {
      expect(GCIT_20_DIMENSIONS).toHaveLength(20);
      GCIT_20_DIMENSIONS.forEach((dim, idx) => {
        expect(dim.index).toBe(idx);
        expect(typeof dim.name).toBe('string');
        expect(typeof dim.category).toBe('string');
        expect(['normal', 'inverted']).toContain(dim.polarity);
      });
    });

    it('maps hair color and textures with normal polarity (+1.0 condition, 1.0 mask)', () => {
      const blackHair = resolveGCITTraceForAttribute('hair_color', 'black', 'explicit');
      expect(blackHair.isEligible).toBe(true);
      expect(blackHair.gcitDimension).toBe('hair_black');
      expect(blackHair.gcitIndex).toBe(0);
      expect(blackHair.conditionValue).toBe(1.0);
      expect(blackHair.maskValue).toBe(1.0);
      expect(blackHair.polarity).toBe('normal');
      expect(blackHair.isIncluded).toBe(true);

      const blondHair = resolveGCITTraceForAttribute('hair_color', 'blond', 'explicit');
      expect(blondHair.gcitIndex).toBe(2);
      expect(blondHair.conditionValue).toBe(1.0);
      expect(blondHair.maskValue).toBe(1.0);

      const wavyHair = resolveGCITTraceForAttribute('hair_style', 'wavy hair', 'explicit');
      expect(wavyHair.gcitIndex).toBe(7);
      expect(wavyHair.conditionValue).toBe(1.0);
    });

    it('maps facial geometry (nose, lips, face shape) with proper polarity', () => {
      const bigNose = resolveGCITTraceForAttribute('nose', 'large nose', 'explicit');
      expect(bigNose.gcitIndex).toBe(14);
      expect(bigNose.conditionValue).toBe(1.0);
      expect(bigNose.maskValue).toBe(1.0);

      const pointyNose = resolveGCITTraceForAttribute('nose', 'pointed nose', 'explicit');
      expect(pointyNose.gcitIndex).toBe(15);
      expect(pointyNose.conditionValue).toBe(1.0);

      const bigLips = resolveGCITTraceForAttribute('lips', 'thick full lips', 'explicit');
      expect(bigLips.gcitIndex).toBe(16);
      expect(bigLips.conditionValue).toBe(1.0);

      const ovalFace = resolveGCITTraceForAttribute('face_shape', 'oval', 'explicit');
      expect(ovalFace.gcitIndex).toBe(17);
      expect(ovalFace.conditionValue).toBe(1.0);

      const cheekbones = resolveGCITTraceForAttribute('face_shape', 'high cheekbones', 'explicit');
      expect(cheekbones.gcitIndex).toBe(18);
      expect(cheekbones.conditionValue).toBe(1.0);
    });

    it('correctly implements INVERTED polarity for CelebA No_Beard dimension', () => {
      // 1. Witness explicitly states clean-shaven / no beard (negated facial hair)
      // In CelebA: No_Beard = +1.0 means absence of beard!
      const noBeardNegated = resolveGCITTraceForAttribute('facial_hair', 'no beard', 'negated');
      expect(noBeardNegated.isEligible).toBe(true);
      expect(noBeardNegated.gcitDimension).toBe('facial_no_beard');
      expect(noBeardNegated.gcitIndex).toBe(13);
      expect(noBeardNegated.polarity).toBe('inverted');
      expect(noBeardNegated.conditionValue).toBe(1.0); // +1 enforces clean-shaven!
      expect(noBeardNegated.maskValue).toBe(1.0);
      expect(noBeardNegated.isIncluded).toBe(true);
      expect(noBeardNegated.inclusionReason).toContain('inverted CelebA dimension "facial_no_beard"');

      // 2. Clean-shaven explicitly stated
      const cleanShaven = resolveGCITTraceForAttribute('facial_hair', 'clean-shaven', 'explicit');
      expect(cleanShaven.gcitIndex).toBe(13);
      expect(cleanShaven.polarity).toBe('inverted');
      expect(cleanShaven.conditionValue).toBe(1.0);
      expect(cleanShaven.maskValue).toBe(1.0);

      // 3. Witness affirms beard (presence of beard) -> inverted condition receives -1.0
      const beardAffirmed = resolveGCITTraceForAttribute('facial_hair', 'thick beard', 'explicit');
      expect(beardAffirmed.gcitIndex).toBe(13);
      expect(beardAffirmed.polarity).toBe('inverted');
      expect(beardAffirmed.conditionValue).toBe(-1.0); // -1 enforces beard presence
      expect(beardAffirmed.maskValue).toBe(1.0);
    });

    it('correctly implements NORMAL polarity for negated features (e.g. negated mustache, negated glasses)', () => {
      const noMustache = resolveGCITTraceForAttribute('facial_hair', 'no mustache', 'negated');
      expect(noMustache.gcitDimension).toBe('facial_mustache');
      expect(noMustache.gcitIndex).toBe(9);
      expect(noMustache.polarity).toBe('normal');
      expect(noMustache.conditionValue).toBe(-1.0); // -1 actively suppresses mustache
      expect(noMustache.maskValue).toBe(1.0);

      const noGlasses = resolveGCITTraceForAttribute('other_observable', 'no glasses', 'negated');
      expect(noGlasses.gcitDimension).toBe('glasses');
      expect(noGlasses.gcitIndex).toBe(19);
      expect(noGlasses.polarity).toBe('normal');
      expect(noGlasses.conditionValue).toBe(-1.0);
      expect(noGlasses.maskValue).toBe(1.0);
    });
  });

  // =========================================================================
  // AREA 5, 6 & 7: Uncertain, Contradictory, and Unsupported Modalities
  // =========================================================================
  describe('Area 5, 6 & 7: Masking, Contradictions, and Unsupported Modalities', () => {
    it('masks UNCERTAIN attributes to 0.0 with mask = 0.0 to prevent cognitive bias', () => {
      const uncertainBeard = resolveGCITTraceForAttribute('facial_hair', 'maybe a goatee', 'uncertain');
      expect(uncertainBeard.isEligible).toBe(false);
      expect(uncertainBeard.conditionValue).toBe(0.0);
      expect(uncertainBeard.maskValue).toBe(0.0);
      expect(uncertainBeard.isIncluded).toBe(false);
      expect(uncertainBeard.inclusionReason).toContain('Masked to 0.0 to prevent cognitive bias');
    });

    it('masks UNKNOWN or unmentioned attributes to 0.0 with mask = 0.0', () => {
      const unknownHair = resolveGCITTraceForAttribute('hair', null, 'unknown');
      expect(unknownHair.isEligible).toBe(false);
      expect(unknownHair.conditionValue).toBe(0.0);
      expect(unknownHair.maskValue).toBe(0.0);
      expect(unknownHair.isIncluded).toBe(false);
      expect(unknownHair.inclusionReason).toContain('drawn from empirical prior');
    });

    it('blocks generation for CONTRADICTORY testimony', () => {
      const contradiction = resolveGCITTraceForAttribute('hair_color', 'black and blond', 'contradictory');
      expect(contradiction.isEligible).toBe(false);
      expect(contradiction.conditionValue).toBe(0.0);
      expect(contradiction.maskValue).toBe(0.0);
      expect(contradiction.isIncluded).toBe(false);
      expect(contradiction.inclusionReason).toContain('Blocks generation until explicitly arbitrated');
    });

    it('identifies and isolates UNSUPPORTED modalities (eye color, scars, clothing)', () => {
      // Eye color
      const blueEyes = resolveGCITTraceForAttribute('eyes', 'piercing blue eyes', 'explicit');
      expect(blueEyes.isEligible).toBe(false);
      expect(blueEyes.gcitDimension).toBeNull();
      expect(blueEyes.maskValue).toBe(0.0);
      expect(blueEyes.inclusionReason).toContain('Eye color is unannotated in CelebA');

      // Distinguishing marks / scars
      const scar = resolveGCITTraceForAttribute('distinguishing_marks', 'scar across cheek', 'explicit');
      expect(scar.isEligible).toBe(false);
      expect(scar.gcitDimension).toBeNull();
      expect(scar.maskValue).toBe(0.0);
      expect(scar.inclusionReason).toContain('Scars, tattoos, and birthmarks are unannotated');

      // Non-facial observable (clothing)
      const jacket = resolveGCITTraceForAttribute('other_observable', 'green jacket', 'explicit');
      expect(jacket.isEligible).toBe(false);
      expect(jacket.maskValue).toBe(0.0);
      expect(jacket.inclusionReason).toContain('Non-facial observable');
    });

    it('generates accurate 20-dimensional vector and binary mask from trace items', () => {
      const items: any[] = [
        {
          gcitStatus: {
            isIncluded: true,
            gcitIndex: 0, // hair_black
            conditionValue: 1.0,
            maskValue: 1.0,
          },
        },
        {
          gcitStatus: {
            isIncluded: true,
            gcitIndex: 13, // facial_no_beard
            conditionValue: 1.0,
            maskValue: 1.0,
          },
        },
        {
          gcitStatus: {
            isIncluded: false, // masked / unsupported
            gcitIndex: null,
            conditionValue: 0.0,
            maskValue: 0.0,
          },
        },
      ];

      const { cVector, mVector } = buildGCITTensorsFromTrace(items);
      expect(cVector).toHaveLength(20);
      expect(mVector).toHaveLength(20);

      // hair_black at index 0
      expect(cVector[0]).toBe(1.0);
      expect(mVector[0]).toBe(1.0);

      // facial_no_beard at index 13
      expect(cVector[13]).toBe(1.0);
      expect(mVector[13]).toBe(1.0);

      // unconditioned dimensions
      expect(cVector[1]).toBe(0.0);
      expect(mVector[1]).toBe(0.0);
      expect(cVector[19]).toBe(0.0);
      expect(mVector[19]).toBe(0.0);
    });
  });

  // =========================================================================
  // AREA 8 & 9: Non-Destructive Review & Investigator Corrections
  // =========================================================================
  describe('Area 8 & 9: Non-Destructive Audit Trail and Investigator Overrides', () => {
    it('preserves originalValue and originalEvidenceState when investigator applies corrections', async () => {
      const mockDoc: any = {
        _id: new Types.ObjectId(),
        case: new Types.ObjectId(),
        rawStatement: 'He had dark hair and maybe a beard.',
        reviewStatus: 'pending',
        extractedAttributes: [
          {
            attributeName: 'hair_color',
            category: 'hair',
            value: 'dark hair',
            originalValue: 'dark hair',
            evidenceState: 'explicit',
            originalEvidenceState: 'explicit',
            sourceText: 'dark hair',
            start: 7,
            end: 16,
            surroundingContext: 'He had dark hair and maybe a beard.',
            extractedBy: 'bilstm-crf',
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
              inclusionReason: 'Affirmed dark hair.',
            },
          },
        ],
        save: jest.fn().mockImplementation(function (this: any) {
          return Promise.resolve(this);
        }),
      };

      const mockQuery: any = Promise.resolve(mockDoc);
      mockQuery.populate = jest.fn().mockReturnValue(mockQuery);
      jest.spyOn(WitnessDescription, 'findById').mockReturnValue(mockQuery);

      const updatedMockDoc = {
        ...mockDoc,
        extractedAttributes: [
          {
            ...mockDoc.extractedAttributes[0],
            value: 'brown hair',
            isCorrected: true,
            originalValue: 'dark hair',
            originalEvidenceState: 'explicit',
            correction: {
              correctedValue: 'brown hair',
              reason: 'Witness clarified on re-interview that hair was chocolate brown under sunlight.',
            },
            gcitStatus: {
              gcitDimension: 'hair_brown',
              gcitIndex: 1,
            },
          },
        ],
      };

      const updateQuery: any = {
        populate: jest.fn().mockReturnThis(),
        then: (resolve: any) => Promise.resolve(updatedMockDoc).then(resolve),
      };
      jest.spyOn(WitnessDescription, 'findByIdAndUpdate').mockReturnValue(updateQuery);

      // Investigator corrects attribute from "dark hair" to "brown hair"
      const updated = await witnessDescriptionService.addCorrection(
        mockDoc._id.toString(),
        {
          attributeCategory: 'hair',
          correctedValue: 'brown hair',
          reason: 'Witness clarified on re-interview that hair was chocolate brown under sunlight.',
          correctedEvidenceState: 'explicit',
        },
        new Types.ObjectId().toString(),
        'admin'
      );

      const correctedAttr = updated.extractedAttributes[0];

      // Non-destructive verification: originalValue and originalEvidenceState are preserved!
      expect(correctedAttr.originalValue).toBe('dark hair');
      expect(correctedAttr.originalEvidenceState).toBe('explicit');

      // Active value is updated
      expect(correctedAttr.value).toBe('brown hair');
      expect(correctedAttr.isCorrected).toBe(true);

      // Audit correction sub-document is recorded
      expect(correctedAttr.correction).toBeDefined();
      expect(correctedAttr.correction?.correctedValue).toBe('brown hair');
      expect(correctedAttr.correction?.reason).toContain('Witness clarified on re-interview');

      // GCIT status is re-evaluated for the corrected value ("brown hair" -> index 1)
      expect(correctedAttr.gcitStatus?.gcitDimension).toBe('hair_brown');
      expect(correctedAttr.gcitStatus?.gcitIndex).toBe(1);
    });

    it('generates a full witness description evidence trace with GCIT preview', async () => {
      const caseId = new Types.ObjectId();
      const userId = new Types.ObjectId();
      const mockDoc: any = {
        _id: new Types.ObjectId(),
        case: caseId,
        rawStatement: 'He had black hair and was clean-shaven.',
        reviewStatus: 'approved',
        reviewedBy: userId,
        reviewedAt: new Date(),
        extractedAttributes: [
          {
            attributeName: 'hair_black',
            category: 'hair',
            value: 'black hair',
            originalValue: 'black hair',
            evidenceState: 'explicit',
            originalEvidenceState: 'explicit',
            sourceText: 'black hair',
            start: 7,
            end: 17,
            surroundingContext: 'He had black hair and was clean-shaven.',
            extractedBy: 'bilstm-crf',
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
              inclusionReason: 'Witness affirmed black hair.',
            },
          },
          {
            attributeName: 'facial_no_beard',
            category: 'facial_hair',
            value: 'clean-shaven',
            originalValue: 'clean-shaven',
            evidenceState: 'explicit',
            originalEvidenceState: 'explicit',
            sourceText: 'clean-shaven',
            start: 26,
            end: 38,
            surroundingContext: 'He had black hair and was clean-shaven.',
            extractedBy: 'bilstm-crf',
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
              inclusionReason: 'Witness negated facial hair (clean-shaven).',
            },
          },
        ],
      };

      jest.spyOn(WitnessDescription, 'findById').mockReturnValue({
        populate: jest.fn().mockReturnValue({
          populate: jest.fn().mockReturnValue({
            populate: jest.fn().mockReturnValue({
              populate: jest.fn().mockResolvedValue(mockDoc),
            }),
          }),
        }),
      } as any);

      jest.spyOn(Case, 'findById').mockResolvedValue({
        _id: caseId,
        assignedInvestigators: [userId],
      } as any);

      const trace = await witnessDescriptionService.getWitnessDescriptionTrace(
        mockDoc._id.toString(),
        userId.toString(),
        'admin'
      );

      expect(trace.witnessDescriptionId).toBe(mockDoc._id);
      expect(trace.reviewStatus).toBe('approved');
      expect(trace.evidenceTrace).toHaveLength(2);

      // Verify points on trace items
      const firstTrace = trace.evidenceTrace[0];
      expect(firstTrace.attributeName).toBe('hair_black');
      expect(firstTrace.category).toBe('hair');
      expect(firstTrace.effectiveValue).toBe('black hair');
      expect(firstTrace.originalValue).toBe('black hair');
      expect(firstTrace.startChar).toBe(7);
      expect(firstTrace.endChar).toBe(17);
      expect(firstTrace.surroundingContext).toBe(mockDoc.rawStatement);
      expect(firstTrace.evidenceState).toBe('explicit');
      expect(firstTrace.originalEvidenceState).toBe('explicit');
      expect(firstTrace.extractedBy).toBe('bilstm-crf');
      expect(firstTrace.confidence).toBeNull(); // Suppressed per Phase 3.7
      expect(firstTrace.isCorrected).toBe(false);
      expect(firstTrace.gcitStatus.isEligible).toBe(true);
      expect(firstTrace.gcitStatus.gcitIndex).toBe(0);

      // GCIT preview vector contains active dimensions
      expect(trace.gcitPreview.conditionVector[0]).toBe(1.0);
      expect(trace.gcitPreview.maskVector[0]).toBe(1.0);
      expect(trace.gcitPreview.conditionVector[13]).toBe(1.0);
      expect(trace.gcitPreview.maskVector[13]).toBe(1.0);
      expect(trace.gcitPreview.conditionedCount).toBe(2);
    });
  });

  // =========================================================================
  // AREA 10, 11 & 12: Composite Provenance Snapshot & Version Immutability
  // =========================================================================
  describe('Area 10, 11 & 12: Composite Snapshot, Hashes and Version Immutability', () => {
    it('captures full evidenceTrace, 20-D gcitVector, gcitMask, and SHA-256 image hashes on composite generation', async () => {
      const caseId = new Types.ObjectId();
      const witnessId = new Types.ObjectId();
      const userId = new Types.ObjectId();

      const mockCase: any = {
        _id: caseId,
        caseNumber: 'CASE-2026-XAI-001',
        title: 'XAI Validation Case',
        assignedInvestigators: [userId],
      };

      const mockWitness: any = {
        _id: witnessId,
        case: caseId,
        rawStatement: 'He was clean-shaven with short black hair.',
        reviewStatus: 'approved',
        extractedAttributes: [
          {
            attributeName: 'hair_black',
            category: 'hair',
            value: 'black hair',
            originalValue: 'black hair',
            evidenceState: 'explicit',
            originalEvidenceState: 'explicit',
            sourceText: 'black hair',
            start: 31,
            end: 41,
            surroundingContext: 'He was clean-shaven with short black hair.',
            extractedBy: 'bilstm-crf',
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
              inclusionReason: 'Witness affirmed black hair.',
            },
          },
        ],
      };

      jest.spyOn(Case, 'findById').mockResolvedValue(mockCase as any);
      jest.spyOn(WitnessDescription, 'findOne').mockResolvedValue(mockWitness as any);
      jest.spyOn(WitnessDescription, 'findById').mockResolvedValue(mockWitness as any);
      jest.spyOn(Composite, 'countDocuments').mockResolvedValue(0);

      let savedComposite: any = null;
      jest.spyOn(Composite, 'create').mockImplementation((data: any) => {
        savedComposite = {
          ...data,
          _id: new Types.ObjectId(),
          save: jest.fn().mockResolvedValue(true),
        };
        return Promise.resolve(savedComposite);
      });

      jest.spyOn(SecurityAudit, 'create').mockResolvedValue({} as any);

      const validPngB64 =
        'iVBORw0KGgoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=';

      jest.spyOn(mlClient, 'generateComposite').mockResolvedValue({
        status: 'success',
        generator: {
          model_name: 'ConditionalWGANGPGenerator',
          architecture: 'Conditional WGAN-GP with Projection Conditioning',
          phase: 'Phase 5.4 / 7.0',
          checkpoint_path: 'phantom-ai-ml/checkpoints/cgan_generator_phase54.pt',
          checkpoint_sha256: 'a1714389d0d1eddad10fae74a736ca41de5e230eada095e6868e5daeeb2a1964',
          best_epoch: 4,
          parameter_count: 4102336,
          device: 'cpu',
        },
        variations: [
          {
            variation_index: 0,
            seed: 42,
            image_base64: validPngB64,
            data_url: `data:image/png;base64,${validPngB64}`,
            width: 64,
            height: 64,
          },
          {
            variation_index: 1,
            seed: 43,
            image_base64: validPngB64,
            data_url: `data:image/png;base64,${validPngB64}`,
            width: 64,
            height: 64,
          },
        ],
        conditioned_attributes: ['hair_black', 'facial_no_beard'],
        unconditioned_attributes: ['hair_blond', 'hair_brown'],
        warnings: [],
        latency_ms: 120,
        timestamp: new Date().toISOString(),
      });

      const result = await compositeService.generateCompositeForCase(
        caseId.toString(),
        userId.toString(),
        'admin',
        {
          witnessDescriptionId: witnessId.toString(),
          numVariations: 2,
          seed: 42,
        }
      );

      expect(result).toBeDefined();
      expect(savedComposite).toBeDefined();

      // 1. Evidence trace snapshot captured
      expect(savedComposite.evidenceTrace).toBeDefined();
      expect(savedComposite.evidenceTrace.length).toBeGreaterThan(0);

      // 2. 20-D GCIT vectors saved
      expect(savedComposite.gcitVector).toHaveLength(20);
      expect(savedComposite.gcitMask).toHaveLength(20);

      // 3. Image hashes computed for every variation PNG buffer
      expect(savedComposite.imageHashes).toBeDefined();
      expect(savedComposite.imageHashes).toHaveLength(2);
      savedComposite.imageHashes.forEach((h: string) => {
        expect(h).toMatch(/^[a-f0-9]{64}$/); // Valid SHA-256 hex string!
      });

      // 4. Generator Checkpoint SHA-256 verified
      expect(savedComposite.generatorMetadata.checkpointSha256).toBe(
        'a1714389d0d1eddad10fae74a736ca41de5e230eada095e6868e5daeeb2a1964'
      );
    });

    it('serves composite trace snapshot via getCompositeTrace maintaining immutability', async () => {
      const compositeId = new Types.ObjectId();
      const caseId = new Types.ObjectId();
      const userId = new Types.ObjectId();

      const mockCompositeDoc: any = {
        _id: compositeId,
        case: caseId,
        witnessDescription: new Types.ObjectId(),
        version: 1,
        generatorMetadata: {
          modelName: 'ConditionalWGANGPGenerator',
          phase: 'Phase 5.4',
          checkpointSha256: 'a1714389d0d1eddad10fae74a736ca41de5e230eada095e6868e5daeeb2a1964',
        },
        gcitVector: new Array(20).fill(0.0),
        gcitMask: new Array(20).fill(0.0),
        imageHashes: [
          {
            variationIndex: 0,
            sha256: 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
          },
        ],
        variations: [
          {
            variationIndex: 0,
            seed: 42,
            imageUrl: '/uploads/composites/var0.png',
          },
        ],
        evidenceTrace: [
          {
            attributeName: 'hair_black',
            category: 'hair',
            categoryDisplayName: 'Hair Style & Color',
            extractedValue: 'black',
            currentValue: 'black',
            originalWitnessStatement: 'He had black hair.',
            exactCharacterOffsets: { start: 7, end: 12, matchedSubstring: 'black' },
            surroundingContextSentence: 'He had black hair.',
            extractedEvidenceState: 'explicit',
            currentEvidenceState: 'explicit',
            nlpProvenance: {
              modelName: 'BiLSTM-CRF with Character-Aware Embeddings',
              architecture: 'Character-Aware BiLSTM-CRF',
              checkpointSha256: '2d567c79d759e76940968fadb02d412b530d3405e377a2d6454c64e9f2e7b4a3',
              extractionConfidence: null,
              confidenceNote: 'Uncalibrated probabilities suppressed.',
            },
            investigatorCorrection: {
              isCorrected: false,
              correctedBy: null,
              correctedAt: null,
              originalValue: 'black',
              originalEvidenceState: 'explicit',
              rationale: null,
            },
            reviewStatus: 'approved',
            gcitTranslation: {
              isSupported: true,
              isEligible: true,
              dimensionIndex: 0,
              dimensionName: 'hair_black',
              polarity: 'direct',
              conditionValue: 1.0,
              maskValue: 1.0,
              inclusionStatus: 'conditioned',
              inclusionReason: 'Affirmed black hair.',
            },
            forensicDisclosure: {
              isBiometricIdentification: false,
              generativePriorNotice: 'Investigative lead aid only.',
            },
          },
        ],
        createdAt: new Date(),
      };

      jest.spyOn(Case, 'findById').mockResolvedValue({
        _id: caseId,
        assignedInvestigators: [userId],
      } as any);

      jest.spyOn(Composite, 'findOne').mockReturnValue({
        populate: jest.fn().mockReturnValue({
          populate: jest.fn().mockReturnValue({
            populate: jest.fn().mockResolvedValue(mockCompositeDoc),
          }),
        }),
      } as any);

      const trace = await compositeService.getCompositeTrace(
        caseId.toString(),
        compositeId.toString(),
        userId.toString(),
        'admin'
      );

      expect(trace.compositeId).toBe(compositeId);
      expect(trace.version).toBe(1);
      expect(trace.generatorMetadata.checkpointSha256).toBe(
        'a1714389d0d1eddad10fae74a736ca41de5e230eada095e6868e5daeeb2a1964'
      );
      expect(trace.imageHashes).toHaveLength(1);
      expect(trace.evidenceTrace).toHaveLength(1);
    });

    it('rejects access when composite does not belong to the requested case', async () => {
      const userId = new Types.ObjectId();
      const caseId = new Types.ObjectId();

      jest.spyOn(Case, 'findById').mockResolvedValue({
        _id: caseId,
        assignedInvestigators: [userId],
      } as any);

      jest.spyOn(Composite, 'findOne').mockReturnValue({
        populate: jest.fn().mockReturnValue({
          populate: jest.fn().mockReturnValue({
            populate: jest.fn().mockResolvedValue(null),
          }),
        }),
      } as any);

      await expect(
        compositeService.getCompositeTrace(
          caseId.toString(),
          new Types.ObjectId().toString(),
          userId.toString(),
          'admin'
        )
      ).rejects.toThrow('Composite not found for this case');
    });
  });
});
