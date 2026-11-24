import { Types } from 'mongoose';
import * as caseService from '../src/services/case.service';
import * as witnessDescriptionService from '../src/services/witnessDescription.service';
import * as compositeService from '../src/services/composite.service';
import * as caseCollaborationService from '../src/services/caseCollaboration.service';
import { Case } from '../src/models/Case.model';
import { WitnessDescription } from '../src/models/WitnessDescription.model';
import { Composite } from '../src/models/Composite.model';
import { Report } from '../src/models/Report.model';
import { SecurityAudit } from '../src/models/SecurityAudit.model';
import { CaseComment } from '../src/models/CaseComment.model';

function createChainableQuery<T>(doc: T): any {
  if (doc && typeof doc === 'object') {
    Object.defineProperties(doc, {
      populate: {
        value: jest.fn(() => doc),
        configurable: true,
        writable: true,
      },
      select: {
        value: jest.fn(() => doc),
        configurable: true,
        writable: true,
      },
      sort: {
        value: jest.fn(() => doc),
        configurable: true,
        writable: true,
      },
      skip: {
        value: jest.fn(() => doc),
        configurable: true,
        writable: true,
      },
      limit: {
        value: jest.fn(() => doc),
        configurable: true,
        writable: true,
      },
      lean: {
        value: jest.fn(() => Promise.resolve(doc)),
        configurable: true,
        writable: true,
      },
    });
  }
  return doc;
}

describe('Phase 7.8 Final Integration, System Validation & Regression Suite', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  // =========================================================================
  // 1. Case Visibility RBAC Isolation (GET /api/v1/cases Regression Fix)
  // =========================================================================
  describe('1. Case List RBAC Visibility Isolation (CaseService.getAllCases)', () => {
    it('restricts getAllCases query to creator or assignedTo for regular investigators', async () => {
      const investigatorId = new Types.ObjectId();
      const mockCases = [
        {
          _id: new Types.ObjectId(),
          caseNumber: 'CASE-780001',
          title: 'Assigned Case',
          createdBy: investigatorId,
          assignedTo: [],
          isDeleted: false,
        },
      ];

      let capturedFilter: any = null;
      jest.spyOn(Case, 'find').mockImplementation(((filter: any) => {
        capturedFilter = filter;
        return createChainableQuery(mockCases);
      }) as any);
      jest.spyOn(Case, 'countDocuments').mockResolvedValue(1 as any);

      const result = await caseService.getAllCases(
        { page: 1, limit: 20 },
        investigatorId.toString(),
        'investigator'
      );

      expect(result.total).toBe(1);
      expect(capturedFilter).toBeDefined();
      expect(capturedFilter.$or).toBeDefined();
      expect(capturedFilter.$or.length).toBeGreaterThanOrEqual(2);
      expect(capturedFilter.$or[0].createdBy.toString()).toBe(investigatorId.toString());
      expect(capturedFilter.$or[1].assignedTo.toString()).toBe(investigatorId.toString());
    });

    it('allows supervisors and admins to query all cases without $or restriction', async () => {
      const supervisorId = new Types.ObjectId();
      let capturedFilter: any = null;
      jest.spyOn(Case, 'find').mockImplementation(((filter: any) => {
        capturedFilter = filter;
        return createChainableQuery([]);
      }) as any);
      jest.spyOn(Case, 'countDocuments').mockResolvedValue(0 as any);

      await caseService.getAllCases({ page: 1, limit: 20 }, supervisorId.toString(), 'supervisor');
      expect(capturedFilter.$or).toBeUndefined();
    });
  });

  // =========================================================================
  // 2. Cross-Phase Lineage Integration (Phases 7.0 + 7.1 + 7.7)
  // =========================================================================
  describe('2. Cross-Phase Traceability, Multi-Variation Comparison & Timeline Integration', () => {
    it('maintains consistent lineage across Phase 7.0 XAI trace, Phase 7.1 variation selection, and Phase 7.7 timeline & progress', async () => {
      const caseId = new Types.ObjectId();
      const creatorId = new Types.ObjectId();
      const collaboratorId = new Types.ObjectId();
      const witnessId = new Types.ObjectId();
      const compositeId = new Types.ObjectId();
      const reportId = new Types.ObjectId();
      const commentId = new Types.ObjectId();

      const cVec = new Array(20).fill(0);
      const mVec = new Array(20).fill(0);
      cVec[0] = 1.0; // hair_black
      mVec[0] = 1.0;
      cVec[13] = 1.0; // facial_no_beard (inverted from beard=negated)
      mVec[13] = 1.0;

      const mockCaseDoc: any = {
        _id: caseId,
        caseNumber: 'CASE-780100',
        title: 'Integrated Audit Case',
        status: 'in_progress',
        priority: 'high',
        createdBy: {
          _id: creatorId,
          name: 'Det. Lead',
          email: 'lead@agency.gov',
          role: 'investigator',
          investigatorId: 'INV-7801',
        },
        assignedTo: [
          {
            _id: collaboratorId,
            name: 'Det. Partner',
            email: 'partner@agency.gov',
            role: 'investigator',
            investigatorId: 'INV-7802',
          },
        ],
        isDeleted: false,
        createdAt: new Date('2026-11-02T12:00:00.000Z'),
        updatedAt: new Date('2026-11-02T12:05:00.000Z'),
      };

      const mockWitnessDoc: any = {
        _id: witnessId,
        case: caseId,
        witnessName: 'Witness Alpha',
        isAnonymous: false,
        recordedBy: mockCaseDoc.createdBy,
        rawStatement: 'Suspect had short black hair and no beard.',
        reviewStatus: 'approved',
        reviewedBy: mockCaseDoc.createdBy,
        reviewedAt: new Date('2026-11-02T12:15:00.000Z'),
        extractedAttributes: [
          {
            attributeName: 'Hair Color',
            category: 'hair_color',
            value: 'black',
            originalValue: 'dark',
            evidenceState: 'explicit',
            originalEvidenceState: 'explicit',
            sourceText: 'black hair',
            start: 18,
            end: 28,
            surroundingContext: 'Suspect had short black hair and no beard.',
            extractedBy: 'bilstm-crf',
            confidence: null,
            isCorrected: true,
            correction: {
              correctedValue: 'black',
              correctedEvidenceState: 'explicit',
              correctedBy: creatorId,
              reason: 'Verified hair color with witness',
              timestamp: new Date('2026-11-02T12:12:00.000Z'),
            },
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
            attributeName: 'Facial Hair',
            category: 'facial_hair',
            value: 'no beard',
            originalValue: 'no beard',
            evidenceState: 'negated',
            originalEvidenceState: 'negated',
            sourceText: 'no beard',
            start: 33,
            end: 41,
            surroundingContext: 'Suspect had short black hair and no beard.',
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
              inclusionReason: 'Inverted polarity on facial_no_beard (+1.0 = No Beard).',
            },
          },
        ],
        corrections: [
          {
            attributeCategory: 'hair_color',
            previousValue: 'dark',
            correctedValue: 'black',
            previousEvidenceState: 'explicit',
            correctedEvidenceState: 'explicit',
            reason: 'Verified hair color with witness',
            correctedBy: mockCaseDoc.createdBy,
            timestamp: new Date('2026-11-02T12:12:00.000Z'),
          },
        ],
        isDeleted: false,
        createdAt: new Date('2026-11-02T12:08:00.000Z'),
        updatedAt: new Date('2026-11-02T12:15:00.000Z'),
      };

      const mockCompositeDoc: any = {
        _id: compositeId,
        case: caseId,
        witnessDescription: mockWitnessDoc,
        version: 1,
        status: 'draft',
        editingType: 'initial',
        generatedBy: mockCaseDoc.createdBy,
        gcitVector: cVec,
        gcitMask: mVec,
        imageHashes: [
          'aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111',
          'bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222',
        ],
        variations: [
          {
            _id: new Types.ObjectId(),
            variationIndex: 1,
            variationRole: 'standard',
            seed: 1001,
            storageKey: 'case78/v1_var1.png',
            imageUrl: '/uploads/composites/case78/v1_var1.png',
            mimeType: 'image/png',
            width: 64,
            height: 64,
            fileSizeBytes: 4200,
            sha256: 'aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111',
            gcitVector: cVec,
            gcitMask: mVec,
            isSelected: true,
          },
          {
            _id: new Types.ObjectId(),
            variationIndex: 2,
            variationRole: 'standard',
            seed: 1002,
            storageKey: 'case78/v1_var2.png',
            imageUrl: '/uploads/composites/case78/v1_var2.png',
            mimeType: 'image/png',
            width: 64,
            height: 64,
            fileSizeBytes: 4210,
            sha256: 'bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222',
            gcitVector: cVec,
            gcitMask: mVec,
            isSelected: false,
          },
        ],
        selectionHistory: [],
        evidenceTrace: [
          {
            attributeName: 'Hair Color',
            category: 'hair_color',
            effectiveValue: 'black',
            originalValue: 'dark',
            evidenceState: 'explicit',
            originalEvidenceState: 'explicit',
            sourceSpan: 'black hair',
            startChar: 18,
            endChar: 28,
            surroundingContext: 'Suspect had short black hair and no beard.',
            extractedBy: 'bilstm-crf',
            confidence: null,
            isCorrected: true,
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
            attributeName: 'Facial Hair',
            category: 'facial_hair',
            effectiveValue: 'no beard',
            originalValue: 'no beard',
            evidenceState: 'negated',
            originalEvidenceState: 'negated',
            sourceSpan: 'no beard',
            startChar: 33,
            endChar: 41,
            surroundingContext: 'Suspect had short black hair and no beard.',
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
              inclusionReason: 'Inverted polarity on facial_no_beard (+1.0 = No Beard).',
            },
          },
        ],
        generatorMetadata: {
          modelName: 'ConditionalWGANGPGenerator',
          checkpointSha256: 'a1714389d0d1eddad10fae74a736ca41de5e230eada095e6868e5daeeb2a1964',
        },
        isDeleted: false,
        createdAt: new Date('2026-11-02T12:20:00.000Z'),
        updatedAt: new Date('2026-11-02T12:20:00.000Z'),
        save: jest.fn().mockResolvedValue(true),
      };

      jest.spyOn(Case, 'findOne').mockImplementation(() => createChainableQuery(mockCaseDoc));
      jest.spyOn(Case, 'findById').mockImplementation(() => createChainableQuery(mockCaseDoc));
      jest
        .spyOn(WitnessDescription, 'findById')
        .mockImplementation(() => createChainableQuery(mockWitnessDoc));
      jest
        .spyOn(WitnessDescription, 'findOne')
        .mockImplementation(() => createChainableQuery(mockWitnessDoc));
      jest
        .spyOn(Composite, 'findOne')
        .mockImplementation(() => createChainableQuery(mockCompositeDoc));
      jest
        .spyOn(Composite, 'find')
        .mockImplementation(() => createChainableQuery([mockCompositeDoc]));
      jest.spyOn(SecurityAudit, 'create').mockResolvedValue({} as any);

      // Step A: Assigned collaborator inspects Phase 7.0 Witness Trace
      const witnessTrace = await witnessDescriptionService.getWitnessDescriptionTrace(
        witnessId.toString(),
        collaboratorId.toString(),
        'investigator'
      );
      expect(witnessTrace.rawStatement).toBe('Suspect had short black hair and no beard.');
      expect(witnessTrace.evidenceTrace).toHaveLength(2);
      expect(witnessTrace.gcitPreview.conditionVector[0]).toBe(1.0);
      expect(witnessTrace.gcitPreview.conditionVector[13]).toBe(1.0);
      expect(witnessTrace.gcitPreview.conditionedCount).toBe(2);

      // Step B: Assigned collaborator compares variations via Phase 7.1
      const comparison = await compositeService.compareCompositeVariations(
        caseId.toString(),
        collaboratorId.toString(),
        'investigator',
        { compositeId: compositeId.toString() }
      );
      expect(comparison.variations).toHaveLength(2);
      expect(comparison.summary.sameCompositeSession).toBe(true);
      expect(comparison.summary.differingGCITDimensionsCount).toBe(0);
      expect(comparison.dimensionComparison).toHaveLength(20);

      // Step C: Assigned collaborator selects Variation #2 as preferred via Phase 7.1
      await compositeService.selectCompositeVariation(
        caseId.toString(),
        compositeId.toString(),
        2,
        collaboratorId.toString(),
        'investigator',
        'approved',
        'Variation 2 jawline matches witness description best'
      );
      expect(mockCompositeDoc.status).toBe('approved');
      expect(mockCompositeDoc.variations[0].isSelected).toBe(false);
      expect(mockCompositeDoc.variations[1].isSelected).toBe(true);
      expect(mockCompositeDoc.selectionHistory).toHaveLength(1);
      expect(mockCompositeDoc.selectionHistory[0].previousVariationIndex).toBe(1);
      expect(mockCompositeDoc.selectionHistory[0].variationIndex).toBe(2);

      // Step D: Assigned collaborator inspects Phase 7.7 Timeline & 7-Stage Progress
      const mockReportDocs: any[] = [
        {
          _id: reportId,
          reportNumber: 'RPT-780100',
          title: 'Forensic Dossier CASE-780100',
          case: caseId,
          composite: mockCompositeDoc,
          compositeVersion: 1,
          generatedBy: mockCaseDoc.createdBy,
          status: 'finalized',
          sha256: 'cccc3333cccc3333cccc3333cccc3333cccc3333cccc3333cccc3333cccc3333',
          createdAt: new Date('2026-11-02T12:30:00.000Z'),
        },
      ];
      const mockCommentDocs: any[] = [
        {
          _id: commentId,
          case: caseId,
          author: mockCaseDoc.assignedTo[0],
          content: 'Reviewed Variation 2 and finalized dossier.',
          linkedComposite: { _id: compositeId, version: 1, status: 'approved' },
          linkedVariationIndex: 2,
          editHistory: [],
          isDeleted: false,
          createdAt: new Date('2026-11-02T12:28:00.000Z'),
          updatedAt: new Date('2026-11-02T12:28:00.000Z'),
        },
      ];

      jest
        .spyOn(WitnessDescription, 'find')
        .mockImplementation(() => createChainableQuery([mockWitnessDoc]));
      jest.spyOn(Report, 'find').mockImplementation(() => createChainableQuery(mockReportDocs));
      jest
        .spyOn(CaseComment, 'find')
        .mockImplementation(() => createChainableQuery(mockCommentDocs));
      jest.spyOn(SecurityAudit, 'find').mockImplementation(() => createChainableQuery([]));

      const timeline = await caseCollaborationService.getCaseTimeline(
        caseId.toString(),
        collaboratorId.toString(),
        'investigator',
        { order: 'asc' }
      );
      expect(timeline.totalEvents).toBeGreaterThanOrEqual(8);
      const eventTypes = timeline.events.map((e) => e.eventType);
      expect(eventTypes).toContain('CASE_CREATED');
      expect(eventTypes).toContain('WITNESS_STATEMENT_SUBMITTED');
      expect(eventTypes).toContain('WITNESS_NLP_ANALYZED');
      expect(eventTypes).toContain('WITNESS_EVIDENCE_CORRECTED');
      expect(eventTypes).toContain('WITNESS_EVIDENCE_APPROVED');
      expect(eventTypes).toContain('COMPOSITE_GENERATED_SUCCESS');
      expect(eventTypes).toContain('COMPOSITE_VARIATION_SELECTED');
      expect(eventTypes).toContain('CASE_COMMENT_ADDED');
      expect(eventTypes).toContain('REPORT_GENERATED_SUCCESS');

      const progress = await caseCollaborationService.getCaseProgress(
        caseId.toString(),
        collaboratorId.toString(),
        'investigator'
      );
      expect(progress.completedStagesCount).toBe(7);
      expect(progress.completionPercentage).toBe(100);
    });
  });

  // =========================================================================
  // 3. Data Integrity & Application-Level Immutability Enforcement
  // =========================================================================
  describe('3. Data Integrity & Application-Level Immutability Controls', () => {
    it('preserves rawStatement, originalValue, and originalEvidenceState across multiple witness corrections', async () => {
      const caseId = new Types.ObjectId();
      const witnessId = new Types.ObjectId();
      const userId = new Types.ObjectId();

      const mockCaseDoc: any = {
        _id: caseId,
        createdBy: userId,
        assignedTo: [],
        isDeleted: false,
      };

      const originalStatement = 'The suspect had blonde hair and glasses.';
      const mockWitnessDoc: any = {
        _id: witnessId,
        case: caseId,
        rawStatement: originalStatement,
        extractedAttributes: [
          {
            category: 'hair_color',
            attributeName: 'Hair Color',
            value: 'blonde',
            originalValue: 'blonde',
            evidenceState: 'explicit',
            originalEvidenceState: 'explicit',
            sourceText: 'blonde',
            start: 16,
            end: 22,
            isCorrected: false,
          },
        ],
        corrections: [],
        reviewStatus: 'pending',
        isDeleted: false,
      };

      jest.spyOn(Case, 'findOne').mockImplementation(() => createChainableQuery(mockCaseDoc));
      jest.spyOn(Case, 'findById').mockImplementation(() => createChainableQuery(mockCaseDoc));
      jest
        .spyOn(WitnessDescription, 'findById')
        .mockImplementation(() => createChainableQuery(mockWitnessDoc));
      jest
        .spyOn(WitnessDescription, 'findByIdAndUpdate')
        .mockImplementation(((_id: any, update: any) => {
          if (update.$push?.corrections) {
            mockWitnessDoc.corrections.push(update.$push.corrections);
          }
          if (update.$set?.extractedAttributes) {
            mockWitnessDoc.extractedAttributes = update.$set.extractedAttributes;
          }
          if (update.$set?.reviewStatus) {
            mockWitnessDoc.reviewStatus = update.$set.reviewStatus;
          }
          return createChainableQuery(mockWitnessDoc);
        }) as any);

      // First correction: blonde -> brown
      await witnessDescriptionService.addCorrection(
        witnessId.toString(),
        {
          attributeCategory: 'hair_color',
          previousValue: 'blonde',
          correctedValue: 'brown',
          previousEvidenceState: 'explicit',
          correctedEvidenceState: 'explicit',
          reason: 'Witness clarified hair was light brown',
        },
        userId.toString(),
        'investigator'
      );

      // Second correction: brown -> black (uncertain)
      await witnessDescriptionService.addCorrection(
        witnessId.toString(),
        {
          attributeCategory: 'hair_color',
          previousValue: 'brown',
          correctedValue: 'black',
          previousEvidenceState: 'explicit',
          correctedEvidenceState: 'uncertain',
          reason: 'Lighting was dim, witness uncertain between dark brown and black',
        },
        userId.toString(),
        'investigator'
      );

      expect(mockWitnessDoc.rawStatement).toBe(originalStatement);
      expect(mockWitnessDoc.extractedAttributes[0].value).toBe('black');
      expect(mockWitnessDoc.extractedAttributes[0].evidenceState).toBe('uncertain');
      // Original values MUST remain immutable in application logic
      expect(mockWitnessDoc.extractedAttributes[0].originalValue).toBe('blonde');
      expect(mockWitnessDoc.extractedAttributes[0].originalEvidenceState).toBe('explicit');
      expect(mockWitnessDoc.corrections).toHaveLength(2);
    });
  });

  // =========================================================================
  // 4. Multi-Endpoint RBAC & Cross-Case Isolation Enforcement
  // =========================================================================
  describe('4. Security & Authorization Boundaries Across Phases 7.0, 7.1 & 7.7', () => {
    it('blocks unassigned investigators with HTTP 403 across Witness Trace (7.0), Comparison (7.1), and Timeline (7.7)', async () => {
      const caseId = new Types.ObjectId();
      const creatorId = new Types.ObjectId();
      const intruderId = new Types.ObjectId();
      const witnessId = new Types.ObjectId();

      const mockCaseDoc: any = {
        _id: caseId,
        caseNumber: 'CASE-780999',
        createdBy: creatorId,
        assignedTo: [],
        isDeleted: false,
      };

      const mockWitnessDoc: any = {
        _id: witnessId,
        case: caseId,
        rawStatement: 'Restricted statement',
        extractedAttributes: [],
      };

      jest.spyOn(Case, 'findOne').mockImplementation(() => createChainableQuery(mockCaseDoc));
      jest.spyOn(Case, 'findById').mockImplementation(() => createChainableQuery(mockCaseDoc));
      jest
        .spyOn(WitnessDescription, 'findById')
        .mockImplementation(() => createChainableQuery(mockWitnessDoc));
      jest.spyOn(SecurityAudit, 'create').mockResolvedValue({} as any);

      await expect(
        witnessDescriptionService.getWitnessDescriptionTrace(
          witnessId.toString(),
          intruderId.toString(),
          'investigator'
        )
      ).rejects.toThrow(/authorized|assigned|permission|access/i);

      await expect(
        compositeService.compareCompositeVariations(
          caseId.toString(),
          intruderId.toString(),
          'investigator',
          { compositeId: new Types.ObjectId().toString() }
        )
      ).rejects.toThrow(/authorized|assigned|permission|access/i);

      await expect(
        caseCollaborationService.getCaseTimeline(
          caseId.toString(),
          intruderId.toString(),
          'investigator',
          {}
        )
      ).rejects.toThrow(/authorized|assigned|permission|access/i);
    });
  });
});
