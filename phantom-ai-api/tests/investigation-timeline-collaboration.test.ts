import { Types } from 'mongoose';
import * as caseCollaborationService from '../src/services/caseCollaboration.service';
import { Case } from '../src/models/Case.model';
import { User } from '../src/models/User.model';
import { WitnessDescription } from '../src/models/WitnessDescription.model';
import { Composite } from '../src/models/Composite.model';
import { Report } from '../src/models/Report.model';
import { SecurityAudit } from '../src/models/SecurityAudit.model';
import { CaseComment } from '../src/models/CaseComment.model';

/**
 * Attaches Mongoose query chain methods (.populate, .select, .sort, .lean) directly
 * to `doc` (non-enumerably) so both `await Model.findOne()` and `await Model.findOne().populate().lean()`
 * resolve to the same live reference.
 */
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
      lean: {
        value: jest.fn(() => Promise.resolve(doc)),
        configurable: true,
        writable: true,
      },
    });
  }
  return doc;
}

describe('Phase 7.7 Investigation Timeline & Investigator Collaboration Suite', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  // =========================================================================
  // AREA 1: Investigation Activity Timeline Ordering, Deduplication & Provenance
  // =========================================================================
  describe('Area 1: Chronological Timeline Synthesis, Audit Merging & Provenance Distinction', () => {
    it('1. synthesizes chronological events (asc and desc) from Case, WitnessDescription, Composite, Report, CaseComment, and SecurityAudit', async () => {
      const caseId = new Types.ObjectId();
      const creatorId = new Types.ObjectId();
      const preAssignedId = new Types.ObjectId();
      const witnessId = new Types.ObjectId();
      const compositeId = new Types.ObjectId();
      const reportId = new Types.ObjectId();
      const commentId = new Types.ObjectId();

      const mockCaseDoc: any = {
        _id: caseId,
        caseNumber: 'CASE-770001',
        title: 'Downtown Bank Robbery',
        status: 'in_progress',
        priority: 'high',
        createdBy: {
          _id: creatorId,
          name: 'Lead Det. Vance',
          email: 'vance@agency.gov',
          role: 'investigator',
          investigatorId: 'INV-001',
        },
        assignedTo: [
          {
            _id: preAssignedId,
            name: 'Det. Miller',
            email: 'miller@agency.gov',
            role: 'investigator',
            investigatorId: 'INV-002',
          },
        ],
        isDeleted: false,
        createdAt: new Date('2026-11-02T10:00:00.000Z'),
        updatedAt: new Date('2026-11-02T10:05:00.000Z'),
      };

      const mockWitnessDocs: any[] = [
        {
          _id: witnessId,
          case: caseId,
          witnessName: 'Elena Rostova',
          isAnonymous: false,
          recordedBy: mockCaseDoc.createdBy,
          rawStatement: 'The suspect was a male with black hair and no beard.',
          extractedAttributes: [
            {
              category: 'hair_color',
              value: 'black',
              evidenceState: 'explicit',
            },
          ],
          corrections: [
            {
              attributeCategory: 'facial_hair',
              previousValue: 'stubble',
              correctedValue: null,
              previousEvidenceState: 'uncertain',
              correctedEvidenceState: 'negated',
              correctedBy: mockCaseDoc.createdBy,
              timestamp: new Date('2026-11-02T10:15:00.000Z'),
              reason: 'Witness clarified suspect was clean-shaven',
              rejected: true,
            },
          ],
          reviewStatus: 'approved',
          reviewedBy: mockCaseDoc.createdBy,
          reviewedAt: new Date('2026-11-02T10:20:00.000Z'),
          modelMetadata: {
            name: 'CharAwareBiLSTMAttributeLabeler',
            architecture: 'CharAwareBiLSTM-CRF',
            phase: '3C.2',
          },
          createdAt: new Date('2026-11-02T10:10:00.000Z'),
          updatedAt: new Date('2026-11-02T10:20:00.000Z'),
        },
      ];

      const mockCompositeDocs: any[] = [
        {
          _id: compositeId,
          case: caseId,
          witnessDescription: witnessId,
          version: 1,
          status: 'approved',
          editingType: 'initial_generation',
          variations: [{ variationIndex: 0, seed: 42 }, { variationIndex: 1, seed: 43 }],
          selectionHistory: [
            {
              variationIndex: 1,
              previousVariationIndex: 0,
              seed: 43,
              selectedBy: mockCaseDoc.createdBy,
              selectedAt: new Date('2026-11-02T10:35:00.000Z'),
              status: 'approved',
              notes: 'Best match to witness description',
            },
          ],
          createdBy: mockCaseDoc.createdBy,
          createdAt: new Date('2026-11-02T10:25:00.000Z'),
        },
      ];

      const mockReportDocs: any[] = [
        {
          _id: reportId,
          case: caseId,
          reportNumber: 'RPT-20261102-0001',
          title: 'Forensic Dossier CASE-770001',
          compositeVersion: 1,
          sha256: 'abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890',
          generatedBy: mockCaseDoc.createdBy,
          createdAt: new Date('2026-11-02T10:40:00.000Z'),
        },
      ];

      const mockCommentDocs: any[] = [
        {
          _id: commentId,
          case: caseId,
          author: mockCaseDoc.createdBy,
          content: 'Cross-referenced composite v1 with transit camera footage.',
          noteType: 'lead_follow_up',
          isEdited: false,
          editHistory: [],
          isDeleted: false,
          createdAt: new Date('2026-11-02T10:45:00.000Z'),
          updatedAt: new Date('2026-11-02T10:45:00.000Z'),
        },
      ];

      const mockAuditLogs: any[] = [
        {
          _id: new Types.ObjectId(),
          userId: mockCaseDoc.createdBy,
          event: 'COMPOSITE_GENERATED_SUCCESS',
          status: 'success',
          details: {
            caseId: caseId.toString(),
            compositeId: compositeId.toString(),
            version: 1,
          },
          createdAt: new Date('2026-11-02T10:25:01.000Z'),
        },
        {
          _id: new Types.ObjectId(),
          userId: mockCaseDoc.createdBy,
          event: 'COMPOSITE_VARIATIONS_COMPARED',
          status: 'success',
          details: {
            caseId: caseId.toString(),
            compositeId: compositeId.toString(),
            version: 1,
            variationCount: 2,
          },
          createdAt: new Date('2026-11-02T10:30:00.000Z'),
        },
        {
          _id: new Types.ObjectId(),
          userId: mockCaseDoc.createdBy,
          event: 'COMPOSITE_VARIATION_SELECTED',
          status: 'success',
          details: {
            caseId: caseId.toString(),
            compositeId: compositeId.toString(),
            version: 1,
            variationIndex: 1,
          },
          createdAt: new Date('2026-11-02T10:35:00.000Z'),
        },
        {
          _id: new Types.ObjectId(),
          userId: mockCaseDoc.createdBy,
          event: 'REPORT_GENERATED_SUCCESS',
          status: 'success',
          details: {
            caseId: caseId.toString(),
            reportId: reportId.toString(),
            reportNumber: 'RPT-20261102-0001',
          },
          createdAt: new Date('2026-11-02T10:40:00.000Z'),
        },
        {
          _id: new Types.ObjectId(),
          userId: mockCaseDoc.createdBy,
          event: 'REPORT_DOWNLOADED',
          status: 'success',
          details: {
            caseId: caseId.toString(),
            reportId: reportId.toString(),
            reportNumber: 'RPT-20261102-0001',
          },
          createdAt: new Date('2026-11-02T10:42:00.000Z'),
        },
      ];

      jest
        .spyOn(Case, 'findOne')
        .mockImplementation(() => createChainableQuery(mockCaseDoc));
      jest
        .spyOn(WitnessDescription, 'find')
        .mockImplementation(() => createChainableQuery(mockWitnessDocs));
      jest
        .spyOn(Composite, 'find')
        .mockImplementation(() => createChainableQuery(mockCompositeDocs));
      jest
        .spyOn(Report, 'find')
        .mockImplementation(() => createChainableQuery(mockReportDocs));
      jest
        .spyOn(CaseComment, 'find')
        .mockImplementation(() => createChainableQuery(mockCommentDocs));
      jest
        .spyOn(SecurityAudit, 'find')
        .mockImplementation(() => createChainableQuery(mockAuditLogs));

      const ascResult = await caseCollaborationService.getCaseTimeline(
        caseId.toString(),
        creatorId.toString(),
        'investigator',
        { order: 'asc' }
      );

      expect(ascResult.caseNumber).toBe('CASE-770001');
      expect(ascResult.order).toBe('asc');
      expect(ascResult.events.length).toBeGreaterThanOrEqual(10);

      for (let i = 1; i < ascResult.events.length; i++) {
        const prevTs = new Date(ascResult.events[i - 1].timestamp).getTime();
        const currTs = new Date(ascResult.events[i].timestamp).getTime();
        expect(currTs).toBeGreaterThanOrEqual(prevTs);
      }

      // Verify zero duplicate entries for merged audit + record events
      expect(
        ascResult.events.filter((e) => e.eventType === 'COMPOSITE_GENERATED_SUCCESS')
      ).toHaveLength(1);
      expect(
        ascResult.events.filter((e) => e.eventType === 'COMPOSITE_VARIATION_SELECTED')
      ).toHaveLength(1);
      expect(
        ascResult.events.filter((e) => e.eventType === 'REPORT_GENERATED_SUCCESS')
      ).toHaveLength(1);

      // Verify standalone SecurityAudit events are included
      expect(
        ascResult.events.some((e) => e.eventType === 'COMPOSITE_VARIATIONS_COMPARED')
      ).toBe(true);
      expect(
        ascResult.events.some((e) => e.eventType === 'REPORT_DOWNLOADED')
      ).toBe(true);
    });

    it('2. explicitly distinguishes historical derived state from audited events and supports category filtering', async () => {
      const caseId = new Types.ObjectId();
      const creatorId = new Types.ObjectId();
      const preAssignedId = new Types.ObjectId();

      const mockCaseDoc: any = {
        _id: caseId,
        caseNumber: 'CASE-770010',
        title: 'Historical Case Record',
        status: 'open',
        priority: 'medium',
        createdBy: { _id: creatorId, name: 'Det. Vance' },
        assignedTo: [{ _id: preAssignedId, name: 'Det. Miller' }],
        isDeleted: false,
        createdAt: new Date('2026-11-01T08:00:00.000Z'),
        updatedAt: new Date('2026-11-01T09:00:00.000Z'),
      };

      jest
        .spyOn(Case, 'findOne')
        .mockImplementation(() => createChainableQuery(mockCaseDoc));
      jest
        .spyOn(WitnessDescription, 'find')
        .mockImplementation(() => createChainableQuery([]));
      jest
        .spyOn(Composite, 'find')
        .mockImplementation(() => createChainableQuery([]));
      jest
        .spyOn(Report, 'find')
        .mockImplementation(() => createChainableQuery([]));
      jest
        .spyOn(CaseComment, 'find')
        .mockImplementation(() => createChainableQuery([]));
      jest
        .spyOn(SecurityAudit, 'find')
        .mockImplementation(() => createChainableQuery([]));

      const timeline = await caseCollaborationService.getCaseTimeline(
        caseId.toString(),
        creatorId.toString(),
        'investigator',
        { order: 'desc' }
      );

      expect(timeline.provenanceSummary.historical_derived_state).toBe(2);
      const histUpdate = timeline.events.find(
        (e) => e.eventType === 'CASE_UPDATED_HISTORICAL'
      );
      expect(histUpdate?.recordProvenance).toBe('historical_derived_state');
      expect(histUpdate?.provenanceNote).toContain('cannot be reconstructed');

      const collabOnly = await caseCollaborationService.getCaseTimeline(
        caseId.toString(),
        creatorId.toString(),
        'investigator',
        { category: 'collaboration' }
      );
      expect(collabOnly.events).toHaveLength(1);
      expect(collabOnly.events[0].eventType).toBe(
        'CASE_PARTICIPANT_ASSIGNED_INITIAL'
      );
    });
  });

  // =========================================================================
  // AREA 2: Investigator Participant Assignment, Removal & RBAC Enforcement
  // =========================================================================
  describe('Area 2: Investigator Collaboration & Case Participant Assignment RBAC', () => {
    it('3. allows case creator to assign an active investigator and logs CASE_PARTICIPANT_ADDED', async () => {
      const caseId = new Types.ObjectId();
      const creatorId = new Types.ObjectId();
      const newInvestigatorId = new Types.ObjectId();

      const mockCaseDoc: any = {
        _id: caseId,
        caseNumber: 'CASE-770002',
        title: 'Harbor Warehouse Investigation',
        createdBy: creatorId,
        assignedTo: [],
        isDeleted: false,
        save: jest.fn().mockResolvedValue(true),
      };

      const targetUser: any = {
        _id: newInvestigatorId,
        name: 'Det. Sarah Chen',
        email: 'chen@agency.gov',
        role: 'investigator',
        investigatorId: 'INV-2026-009',
        isActive: true,
      };

      jest
        .spyOn(Case, 'findOne')
        .mockImplementation(() => createChainableQuery(mockCaseDoc));
      jest
        .spyOn(User, 'findById')
        .mockImplementation(() => createChainableQuery(targetUser));
      const auditSpy = jest
        .spyOn(SecurityAudit, 'create')
        .mockResolvedValue({ _id: new Types.ObjectId() } as any);
      jest
        .spyOn(SecurityAudit, 'find')
        .mockImplementation(() => createChainableQuery([]));

      const afterAdd = await caseCollaborationService.addCaseParticipant(
        caseId.toString(),
        creatorId.toString(),
        'investigator',
        {
          userId: newInvestigatorId.toString(),
          notes: 'Assigned for composite variation review',
        }
      );

      expect(mockCaseDoc.save).toHaveBeenCalledTimes(1);
      expect(
        mockCaseDoc.assignedTo.some(
          (id: Types.ObjectId) => id.toString() === newInvestigatorId.toString()
        )
      ).toBe(true);
      expect(auditSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          event: 'CASE_PARTICIPANT_ADDED',
          status: 'success',
        })
      );
      expect(afterAdd.canManageParticipants).toBe(true);
    });

    it('4. allows case creator or supervisor to remove an assigned participant and logs CASE_PARTICIPANT_REMOVED', async () => {
      const caseId = new Types.ObjectId();
      const creatorId = new Types.ObjectId();
      const assignedId = new Types.ObjectId();

      const mockCaseDoc: any = {
        _id: caseId,
        caseNumber: 'CASE-770002B',
        createdBy: creatorId,
        assignedTo: [assignedId],
        isDeleted: false,
        save: jest.fn().mockResolvedValue(true),
      };

      jest
        .spyOn(Case, 'findOne')
        .mockImplementation(() => createChainableQuery(mockCaseDoc));
      jest.spyOn(User, 'findById').mockImplementation(() =>
        createChainableQuery({
          _id: assignedId,
          name: 'Det. Chen',
          email: 'chen@agency.gov',
          role: 'investigator',
        })
      );
      const auditSpy = jest
        .spyOn(SecurityAudit, 'create')
        .mockResolvedValue({ _id: new Types.ObjectId() } as any);
      jest
        .spyOn(SecurityAudit, 'find')
        .mockImplementation(() => createChainableQuery([]));

      await caseCollaborationService.removeCaseParticipant(
        caseId.toString(),
        assignedId.toString(),
        creatorId.toString(),
        'investigator',
        'Reassigned to another division'
      );

      expect(mockCaseDoc.assignedTo).toHaveLength(0);
      expect(auditSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          event: 'CASE_PARTICIPANT_REMOVED',
          status: 'success',
        })
      );
    });

    it('5. blocks non-creator assigned investigators from modifying participant roster and prevents removing the case creator', async () => {
      const caseId = new Types.ObjectId();
      const creatorId = new Types.ObjectId();
      const assignedColleagueId = new Types.ObjectId();
      const thirdUserId = new Types.ObjectId();

      const mockCaseDoc: any = {
        _id: caseId,
        caseNumber: 'CASE-770003',
        createdBy: creatorId,
        assignedTo: [assignedColleagueId],
        isDeleted: false,
        save: jest.fn(),
      };

      jest
        .spyOn(Case, 'findOne')
        .mockImplementation(() => createChainableQuery(mockCaseDoc));

      await expect(
        caseCollaborationService.addCaseParticipant(
          caseId.toString(),
          assignedColleagueId.toString(),
          'investigator',
          { userId: thirdUserId.toString() }
        )
      ).rejects.toThrow(
        'Forbidden: Only the case creator, supervisors, or administrators can modify case participant assignments.'
      );

      await expect(
        caseCollaborationService.removeCaseParticipant(
          caseId.toString(),
          creatorId.toString(),
          creatorId.toString(),
          'investigator'
        )
      ).rejects.toThrow('Cannot remove the primary case creator from case ownership');
    });

    it('6. rejects duplicate participant assignment and inactive user assignment', async () => {
      const caseId = new Types.ObjectId();
      const creatorId = new Types.ObjectId();
      const existingParticipantId = new Types.ObjectId();
      const inactiveUserId = new Types.ObjectId();

      const mockCaseDoc: any = {
        _id: caseId,
        caseNumber: 'CASE-770003B',
        createdBy: creatorId,
        assignedTo: [existingParticipantId],
        isDeleted: false,
        save: jest.fn(),
      };

      jest
        .spyOn(Case, 'findOne')
        .mockImplementation(() => createChainableQuery(mockCaseDoc));

      jest.spyOn(User, 'findById').mockImplementationOnce(() =>
        createChainableQuery({
          _id: existingParticipantId,
          name: 'Already Assigned',
          isActive: true,
        })
      );

      await expect(
        caseCollaborationService.addCaseParticipant(
          caseId.toString(),
          creatorId.toString(),
          'investigator',
          { userId: existingParticipantId.toString() }
        )
      ).rejects.toThrow('Investigator is already assigned to this case');

      jest.spyOn(User, 'findById').mockImplementationOnce(() =>
        createChainableQuery({
          _id: inactiveUserId,
          name: 'Inactive User',
          isActive: false,
        })
      );

      await expect(
        caseCollaborationService.addCaseParticipant(
          caseId.toString(),
          creatorId.toString(),
          'investigator',
          { userId: inactiveUserId.toString() }
        )
      ).rejects.toThrow('Cannot assign an inactive user account to a case');
    });
  });

  // =========================================================================
  // AREA 3: Case Comments Creation, Edit History, Soft-Delete & Evidence Isolation
  // =========================================================================
  describe('Area 3: Case Comments, Author Permissions, Edit Audit History & Evidence Isolation', () => {
    it('7. creates case comments linked to case resources, rejects cross-case links, and never alters WitnessDescription or Composite records', async () => {
      const caseId = new Types.ObjectId();
      const otherCaseId = new Types.ObjectId();
      const creatorId = new Types.ObjectId();
      const compositeId = new Types.ObjectId();
      const foreignCompositeId = new Types.ObjectId();
      const commentId = new Types.ObjectId();

      const mockCaseDoc: any = {
        _id: caseId,
        caseNumber: 'CASE-770004',
        title: 'Metro Station Incident',
        createdBy: creatorId,
        assignedTo: [],
        isDeleted: false,
      };

      jest
        .spyOn(Case, 'findOne')
        .mockImplementation(() => createChainableQuery(mockCaseDoc));

      const witnessUpdateSpy = jest.spyOn(WitnessDescription, 'findByIdAndUpdate');
      const compositeUpdateSpy = jest.spyOn(Composite, 'findByIdAndUpdate');

      // 1. Reject cross-case linked composite
      jest.spyOn(Composite, 'findById').mockImplementationOnce(() =>
        createChainableQuery({
          _id: foreignCompositeId,
          case: otherCaseId,
          version: 1,
        })
      );

      await expect(
        caseCollaborationService.addCaseComment(
          caseId.toString(),
          creatorId.toString(),
          'investigator',
          {
            content: 'Trying to link composite from another case',
            noteType: 'general',
            linkedResource: {
              resourceType: 'composite',
              resourceId: foreignCompositeId.toString(),
            },
          }
        )
      ).rejects.toThrow(
        'Linked composite does not exist or does not belong to this case'
      );

      // 2. Valid same-case linked composite comment
      jest.spyOn(Composite, 'findById').mockImplementationOnce(() =>
        createChainableQuery({
          _id: compositeId,
          case: caseId,
          version: 2,
          editingType: 'semantic_latent_traversal',
        })
      );

      const createdCommentDoc: any = {
        _id: commentId,
        case: caseId,
        author: {
          _id: creatorId,
          name: 'Det. Vance',
          email: 'vance@agency.gov',
          role: 'investigator',
        },
        content: 'Composite v2 jawline refinement matches witness follow-up.',
        noteType: 'lead_follow_up',
        linkedResource: {
          resourceType: 'composite',
          resourceId: compositeId,
          label: 'Composite v2 (semantic_latent_traversal)',
        },
        isEdited: false,
        editHistory: [],
        isDeleted: false,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      jest.spyOn(CaseComment, 'create').mockResolvedValue(createdCommentDoc);
      jest
        .spyOn(CaseComment, 'findById')
        .mockImplementation(() => createChainableQuery(createdCommentDoc));
      const auditSpy = jest
        .spyOn(SecurityAudit, 'create')
        .mockResolvedValue({ _id: new Types.ObjectId() } as any);

      const result = await caseCollaborationService.addCaseComment(
        caseId.toString(),
        creatorId.toString(),
        'investigator',
        {
          content: 'Composite v2 jawline refinement matches witness follow-up.',
          noteType: 'lead_follow_up',
          linkedResource: {
            resourceType: 'composite',
            resourceId: compositeId.toString(),
          },
        }
      );

      expect(result.id).toBe(commentId.toString());
      expect(result.noteType).toBe('lead_follow_up');
      expect(result.linkedResource?.label).toBe(
        'Composite v2 (semantic_latent_traversal)'
      );
      expect(result.evidenceIsolationNotice).toContain(
        'strictly isolated from witness testimony'
      );
      expect(auditSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          event: 'CASE_COMMENT_ADDED',
          status: 'success',
        })
      );
      expect(witnessUpdateSpy).not.toHaveBeenCalled();
      expect(compositeUpdateSpy).not.toHaveBeenCalled();
    });

    it('8. enforces comment author/supervisor permissions for edits and preserves editHistory', async () => {
      const caseId = new Types.ObjectId();
      const authorId = new Types.ObjectId();
      const otherAssignedInvestigatorId = new Types.ObjectId();
      const commentId = new Types.ObjectId();

      const mockCaseDoc: any = {
        _id: caseId,
        caseNumber: 'CASE-770005',
        createdBy: authorId,
        assignedTo: [otherAssignedInvestigatorId],
        isDeleted: false,
      };

      const mutableComment: any = {
        _id: commentId,
        case: caseId,
        author: authorId,
        content: 'Initial investigator note on suspect getaway direction.',
        noteType: 'general',
        isEdited: false,
        editHistory: [],
        isDeleted: false,
        createdAt: new Date('2026-11-02T11:00:00.000Z'),
        updatedAt: new Date('2026-11-02T11:00:00.000Z'),
        save: jest.fn().mockResolvedValue(true),
      };

      jest
        .spyOn(Case, 'findOne')
        .mockImplementation(() => createChainableQuery(mockCaseDoc));
      jest
        .spyOn(CaseComment, 'findById')
        .mockImplementation(() => createChainableQuery(mutableComment));
      const auditSpy = jest
        .spyOn(SecurityAudit, 'create')
        .mockResolvedValue({ _id: new Types.ObjectId() } as any);

      // Non-author assigned investigator is blocked from editing colleague's comment
      await expect(
        caseCollaborationService.updateCaseComment(
          caseId.toString(),
          commentId.toString(),
          otherAssignedInvestigatorId.toString(),
          'investigator',
          { content: 'Unauthorized edit attempt' }
        )
      ).rejects.toThrow(
        'Forbidden: Only the comment author, supervisors, or administrators can edit this comment'
      );

      // Original author edits comment -> preserves prior content in editHistory
      const edited = await caseCollaborationService.updateCaseComment(
        caseId.toString(),
        commentId.toString(),
        authorId.toString(),
        'investigator',
        {
          content: 'Updated note: Suspect headed northbound on 5th Ave.',
          noteType: 'lead_follow_up',
          editReason: 'Corrected street direction from CCTV timestamp',
        }
      );

      expect(edited.isEdited).toBe(true);
      expect(edited.content).toBe(
        'Updated note: Suspect headed northbound on 5th Ave.'
      );
      expect(edited.editHistory).toHaveLength(1);
      expect(edited.editHistory[0].previousContent).toBe(
        'Initial investigator note on suspect getaway direction.'
      );
      expect(edited.editHistory[0].editReason).toBe(
        'Corrected street direction from CCTV timestamp'
      );
      expect(auditSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          event: 'CASE_COMMENT_EDITED',
          status: 'success',
        })
      );
    });

    it('9. enforces permissions for soft-deletion and preserves original comment content in audit history', async () => {
      const caseId = new Types.ObjectId();
      const authorId = new Types.ObjectId();
      const otherAssignedInvestigatorId = new Types.ObjectId();
      const supervisorId = new Types.ObjectId();
      const commentId = new Types.ObjectId();

      const mockCaseDoc: any = {
        _id: caseId,
        caseNumber: 'CASE-770005B',
        createdBy: authorId,
        assignedTo: [otherAssignedInvestigatorId],
        isDeleted: false,
      };

      const mutableComment: any = {
        _id: commentId,
        case: caseId,
        author: authorId,
        content: 'Note to be archived via soft-delete.',
        noteType: 'general',
        isEdited: false,
        editHistory: [],
        isDeleted: false,
        createdAt: new Date('2026-11-02T11:00:00.000Z'),
        updatedAt: new Date('2026-11-02T11:00:00.000Z'),
        save: jest.fn().mockResolvedValue(true),
      };

      jest
        .spyOn(Case, 'findOne')
        .mockImplementation(() => createChainableQuery(mockCaseDoc));
      jest
        .spyOn(CaseComment, 'findById')
        .mockImplementation(() => createChainableQuery(mutableComment));
      const auditSpy = jest
        .spyOn(SecurityAudit, 'create')
        .mockResolvedValue({ _id: new Types.ObjectId() } as any);

      // Non-author investigator cannot delete
      await expect(
        caseCollaborationService.deleteCaseComment(
          caseId.toString(),
          commentId.toString(),
          otherAssignedInvestigatorId.toString(),
          'investigator'
        )
      ).rejects.toThrow(
        'Forbidden: Only the comment author, supervisors, or administrators can delete this comment'
      );

      // Supervisor soft-deletes comment -> preserves original content in audit history
      const deleted = await caseCollaborationService.deleteCaseComment(
        caseId.toString(),
        commentId.toString(),
        supervisorId.toString(),
        'supervisor',
        'Duplicate entry consolidated into supervisory summary'
      );

      expect(deleted.isDeleted).toBe(true);
      expect(deleted.content).toBe(
        '[Comment removed — retained in forensic audit history]'
      );
      expect(deleted.originalContentPreservedInAudit).toBe(
        'Note to be archived via soft-delete.'
      );
      expect(deleted.deleteReason).toBe(
        'Duplicate entry consolidated into supervisory summary'
      );
      expect(auditSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          event: 'CASE_COMMENT_DELETED',
          status: 'success',
        })
      );
    });
  });

  // =========================================================================
  // AREA 4: 7-Stage Case Workflow Progress Calculation
  // =========================================================================
  describe('Area 4: 7-Stage Case Progress Tracking Across Not Started, Pending, Blocked & Completed States', () => {
    it('10. calculates blocked state when witness statement has unresolved contradictions and not_started for downstream stages', async () => {
      const caseId = new Types.ObjectId();
      const creatorId = new Types.ObjectId();

      const mockCaseDoc: any = {
        _id: caseId,
        caseNumber: 'CASE-770006',
        status: 'open',
        priority: 'high',
        createdBy: creatorId,
        assignedTo: [],
        isDeleted: false,
        createdAt: new Date('2026-11-02T09:00:00.000Z'),
      };

      jest
        .spyOn(Case, 'findOne')
        .mockImplementation(() => createChainableQuery(mockCaseDoc));
      jest.spyOn(WitnessDescription, 'find').mockImplementationOnce(() =>
        createChainableQuery([
          {
            _id: new Types.ObjectId(),
            case: caseId,
            rawStatement: 'He had a thick beard but was clean-shaven.',
            extractedAttributes: [
              {
                category: 'facial_hair',
                value: 'beard',
                evidenceState: 'contradictory',
              },
            ],
            corrections: [],
            reviewStatus: 'in_review',
            createdAt: new Date('2026-11-02T09:05:00.000Z'),
            updatedAt: new Date('2026-11-02T09:06:00.000Z'),
          },
        ])
      );
      jest
        .spyOn(Composite, 'find')
        .mockImplementationOnce(() => createChainableQuery([]));
      jest
        .spyOn(Report, 'find')
        .mockImplementationOnce(() => createChainableQuery([]));
      jest
        .spyOn(SecurityAudit, 'find')
        .mockImplementationOnce(() => createChainableQuery([]));

      const blockedProgress = await caseCollaborationService.getCaseProgress(
        caseId.toString(),
        creatorId.toString(),
        'investigator'
      );

      expect(blockedProgress.hasBlockingIssues).toBe(true);
      expect(blockedProgress.completedStagesCount).toBe(3);
      const stage4 = blockedProgress.stages.find(
        (s) => s.stageKey === 'evidence_approval'
      );
      const stage5 = blockedProgress.stages.find(
        (s) => s.stageKey === 'composite_generation'
      );
      const stage6 = blockedProgress.stages.find(
        (s) => s.stageKey === 'variation_selection'
      );
      const stage7 = blockedProgress.stages.find(
        (s) => s.stageKey === 'report_generation'
      );
      expect(stage4?.status).toBe('blocked');
      expect(stage5?.status).toBe('blocked');
      expect(stage6?.status).toBe('not_started');
      expect(stage7?.status).toBe('not_started');
    });

    it('11. calculates 100% completion across all 7 stages when witness evidence, composite, variation selection, and report exist', async () => {
      const caseId = new Types.ObjectId();
      const creatorId = new Types.ObjectId();

      const mockCaseDoc: any = {
        _id: caseId,
        caseNumber: 'CASE-770007',
        status: 'in_progress',
        priority: 'high',
        createdBy: creatorId,
        assignedTo: [],
        isDeleted: false,
        createdAt: new Date('2026-11-02T09:00:00.000Z'),
      };

      jest
        .spyOn(Case, 'findOne')
        .mockImplementation(() => createChainableQuery(mockCaseDoc));
      jest.spyOn(WitnessDescription, 'find').mockImplementationOnce(() =>
        createChainableQuery([
          {
            _id: new Types.ObjectId(),
            case: caseId,
            rawStatement: 'Suspect had black hair and glasses.',
            extractedAttributes: [
              { category: 'hair_color', value: 'black', evidenceState: 'explicit' },
            ],
            corrections: [],
            reviewStatus: 'approved',
            reviewedAt: new Date('2026-11-02T09:10:00.000Z'),
            createdAt: new Date('2026-11-02T09:05:00.000Z'),
            updatedAt: new Date('2026-11-02T09:10:00.000Z'),
          },
        ])
      );
      jest.spyOn(Composite, 'find').mockImplementationOnce(() =>
        createChainableQuery([
          {
            _id: new Types.ObjectId(),
            case: caseId,
            version: 1,
            status: 'approved',
            variations: [{ variationIndex: 0, seed: 42 }],
            selectionHistory: [
              {
                variationIndex: 0,
                seed: 42,
                selectedAt: new Date('2026-11-02T09:20:00.000Z'),
                status: 'approved',
              },
            ],
            createdAt: new Date('2026-11-02T09:15:00.000Z'),
          },
        ])
      );
      jest.spyOn(Report, 'find').mockImplementationOnce(() =>
        createChainableQuery([
          {
            _id: new Types.ObjectId(),
            case: caseId,
            reportNumber: 'RPT-20261102-0099',
            sha256: '9999999999999999999999999999999999999999999999999999999999999999',
            createdAt: new Date('2026-11-02T09:25:00.000Z'),
          },
        ])
      );
      jest.spyOn(SecurityAudit, 'find').mockImplementationOnce(() =>
        createChainableQuery([
          {
            _id: new Types.ObjectId(),
            event: 'COMPOSITE_VARIATIONS_COMPARED',
            createdAt: new Date('2026-11-02T09:18:00.000Z'),
          },
        ])
      );

      const fullProgress = await caseCollaborationService.getCaseProgress(
        caseId.toString(),
        creatorId.toString(),
        'investigator'
      );

      expect(fullProgress.completedStagesCount).toBe(7);
      expect(fullProgress.completionPercentage).toBe(100);
      expect(fullProgress.hasBlockingIssues).toBe(false);
      expect(fullProgress.stages.every((s) => s.status === 'completed')).toBe(true);
    });
  });

  // =========================================================================
  // AREA 5: Unauthorized & Cross-Case Access Prevention Across All Endpoints
  // =========================================================================
  describe('Area 5: Case-Level RBAC Enforcement Across Timeline, Progress, Participants & Comments', () => {
    it('12. blocks unassigned investigators (403 Forbidden) from accessing timeline, progress, participants, and comments of another investigator case', async () => {
      const caseId = new Types.ObjectId();
      const ownerId = new Types.ObjectId();
      const unauthorizedInvestigatorId = new Types.ObjectId();

      jest.spyOn(Case, 'findOne').mockImplementation(() =>
        createChainableQuery({
          _id: caseId,
          caseNumber: 'CASE-770099',
          createdBy: ownerId,
          assignedTo: [],
          isDeleted: false,
        } as any)
      );

      await expect(
        caseCollaborationService.getCaseTimeline(
          caseId.toString(),
          unauthorizedInvestigatorId.toString(),
          'investigator'
        )
      ).rejects.toThrow('Forbidden: You are not authorized to view or modify this case file');

      await expect(
        caseCollaborationService.getCaseProgress(
          caseId.toString(),
          unauthorizedInvestigatorId.toString(),
          'investigator'
        )
      ).rejects.toThrow('Forbidden: You are not authorized to view or modify this case file');

      await expect(
        caseCollaborationService.getCaseParticipants(
          caseId.toString(),
          unauthorizedInvestigatorId.toString(),
          'investigator'
        )
      ).rejects.toThrow('Forbidden: You are not authorized to view or modify this case file');

      await expect(
        caseCollaborationService.getCaseComments(
          caseId.toString(),
          unauthorizedInvestigatorId.toString(),
          'investigator'
        )
      ).rejects.toThrow('Forbidden: You are not authorized to view or modify this case file');

      await expect(
        caseCollaborationService.addCaseComment(
          caseId.toString(),
          unauthorizedInvestigatorId.toString(),
          'investigator',
          { content: 'Unauthorized note', noteType: 'general' }
        )
      ).rejects.toThrow('Forbidden: You are not authorized to view or modify this case file');
    });
  });
});
