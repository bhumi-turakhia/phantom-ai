import { Types } from 'mongoose';
import * as authService from '../src/services/auth.service';
import * as caseService from '../src/services/case.service';
import { assertCaseAccess } from '../src/services/witnessDescription.service';
import { getDashboardOverview } from '../src/services/dashboard.service';
import { User } from '../src/models/User.model';
import { Case } from '../src/models/Case.model';
import { Composite } from '../src/models/Composite.model';
import { WitnessDescription } from '../src/models/WitnessDescription.model';
import { Report } from '../src/models/Report.model';
import { CaseComment } from '../src/models/CaseComment.model';
import { SecurityAudit } from '../src/models/SecurityAudit.model';
import { GenerationJob } from '../src/models/GenerationJob.model';
import { mlClient } from '../src/services/mlClient.service';
import { ApiError } from '../src/utils/apiResponse';

jest.mock('../src/services/caseCollaboration.service', () => ({
  getCaseTimeline: jest.fn().mockResolvedValue({ events: [] }),
}));

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

describe('PHANTOM AI — Investigator Identity, Case Access Control and Lifecycle Suite', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  // =========================================================================
  // 1. Investigator Identity & Profile Management
  // =========================================================================
  describe('1. Investigator Identity & Profile Management', () => {
    const mockUserId = new Types.ObjectId().toString();

    it('successfully updates permitted investigator profile fields (name, badge, department)', async () => {
      const mockUserDoc = {
        _id: new Types.ObjectId(mockUserId),
        name: 'Agent Bhumi',
        email: 'bhumi@agency.gov',
        role: 'investigator',
        investigatorId: 'INV-12A1-4B7F97F4',
        badgeNumber: 'BDG-001',
        department: 'Forensic Reconstruction',
        isActive: true,
        save: jest.fn().mockResolvedValue(true),
      };

      jest.spyOn(User, 'findById').mockReturnValue({
        select: jest.fn().mockResolvedValue(mockUserDoc),
      } as any);
      jest.spyOn(SecurityAudit, 'create').mockResolvedValue({} as any);

      const result = await authService.updateUserProfile(mockUserId, {
        name: 'Agent Bhumi Turakhia',
        badgeNumber: 'BDG-999',
        department: 'Biometric Intelligence',
      });

      expect(mockUserDoc.name).toBe('Agent Bhumi Turakhia');
      expect(mockUserDoc.badgeNumber).toBe('BDG-999');
      expect(mockUserDoc.department).toBe('Biometric Intelligence');
      expect(mockUserDoc.save).toHaveBeenCalled();
      expect(SecurityAudit.create).toHaveBeenCalledWith(
        expect.objectContaining({
          event: 'PROFILE_UPDATED',
          status: 'success',
        })
      );
      expect(result.name).toBe('Agent Bhumi Turakhia');
    });

    it('safely ignores or rejects modification to role, email, or investigatorId in profile update', async () => {
      const mockUserDoc = {
        _id: new Types.ObjectId(mockUserId),
        name: 'Agent Bhumi',
        email: 'bhumi@agency.gov',
        role: 'investigator',
        investigatorId: 'INV-12A1-4B7F97F4',
        badgeNumber: 'BDG-001',
        department: 'Forensic',
        isActive: true,
        save: jest.fn().mockResolvedValue(true),
      };

      jest.spyOn(User, 'findById').mockReturnValue({
        select: jest.fn().mockResolvedValue(mockUserDoc),
      } as any);
      jest.spyOn(SecurityAudit, 'create').mockResolvedValue({} as any);

      await authService.updateUserProfile(mockUserId, {
        name: 'Agent Bhumi',
        ...({ role: 'admin', email: 'hacker@agency.gov', investigatorId: 'INV-HAX' } as any),
      });

      expect(mockUserDoc.role).toBe('investigator');
      expect(mockUserDoc.email).toBe('bhumi@agency.gov');
      expect(mockUserDoc.investigatorId).toBe('INV-12A1-4B7F97F4');
    });

    it('rejects password update if current password verification fails', async () => {
      const mockUserDoc = {
        _id: new Types.ObjectId(mockUserId),
        comparePassword: jest.fn().mockResolvedValue(false),
        save: jest.fn(),
      };

      jest.spyOn(User, 'findById').mockReturnValue({
        select: jest.fn().mockResolvedValue(mockUserDoc),
      } as any);

      await expect(
        authService.updateUserProfile(mockUserId, {
          currentPassword: 'WrongPassword123!',
          newPassword: 'NewPassword123!',
        })
      ).rejects.toThrow(ApiError);
    });
  });

  // =========================================================================
  // 2. Multi-Investigator Access Control & Isolation (assertCaseAccess)
  // =========================================================================
  describe('2. Multi-Investigator Access Control & Case Isolation', () => {
    const creatorId = new Types.ObjectId().toString();
    const collaboratorId = new Types.ObjectId().toString();
    const readOnlyCollabId = new Types.ObjectId().toString();
    const strangerId = new Types.ObjectId().toString();
    const caseId = new Types.ObjectId().toString();

    const baseCaseDoc = {
      _id: new Types.ObjectId(caseId),
      caseNumber: 'CASE-777',
      title: 'Isolated Evidence Case',
      status: 'open',
      isDeleted: false,
      createdBy: new Types.ObjectId(creatorId),
      assignedTo: [new Types.ObjectId(creatorId), new Types.ObjectId(collaboratorId), new Types.ObjectId(readOnlyCollabId)],
      collaborators: [
        {
          user: new Types.ObjectId(collaboratorId),
          permission: 'editor',
          addedAt: new Date(),
          addedBy: new Types.ObjectId(creatorId),
        },
        {
          user: new Types.ObjectId(readOnlyCollabId),
          permission: 'read_only',
          addedAt: new Date(),
          addedBy: new Types.ObjectId(creatorId),
        },
      ],
    };

    it('grants full read, edit, and manage permissions to the case creator', async () => {
      jest.spyOn(Case, 'findOne').mockResolvedValue(baseCaseDoc as any);

      await expect(assertCaseAccess(caseId, creatorId, 'investigator', 'read')).resolves.not.toThrow();
      await expect(assertCaseAccess(caseId, creatorId, 'investigator', 'edit')).resolves.not.toThrow();
      await expect(assertCaseAccess(caseId, creatorId, 'investigator', 'manage')).resolves.not.toThrow();
    });

    it('strictly forbids unassigned Investigator B from accessing Investigator A private case (403)', async () => {
      jest.spyOn(Case, 'findOne').mockResolvedValue(baseCaseDoc as any);

      await expect(assertCaseAccess(caseId, strangerId, 'investigator', 'read')).rejects.toThrow(
        /Forbidden: You are not authorized to view or modify this case file/
      );
      await expect(assertCaseAccess(caseId, strangerId, 'investigator', 'edit')).rejects.toThrow(
        /Forbidden: You are not authorized to view or modify this case file/
      );
    });

    it('permits supervisors and admins global supervisory access across all cases', async () => {
      jest.spyOn(Case, 'findOne').mockResolvedValue(baseCaseDoc as any);

      await expect(assertCaseAccess(caseId, strangerId, 'supervisor', 'read')).resolves.not.toThrow();
      await expect(assertCaseAccess(caseId, strangerId, 'admin', 'manage')).resolves.not.toThrow();
    });

    it('permits read_only collaborator for read access, but blocks edit and manage with 403', async () => {
      jest.spyOn(Case, 'findOne').mockResolvedValue(baseCaseDoc as any);

      // Read is allowed
      await expect(assertCaseAccess(caseId, readOnlyCollabId, 'investigator', 'read')).resolves.not.toThrow();

      // Edit is blocked
      await expect(assertCaseAccess(caseId, readOnlyCollabId, 'investigator', 'edit')).rejects.toThrow(
        /Forbidden: Read-only participants cannot alter evidence/
      );

      // Manage is blocked
      await expect(assertCaseAccess(caseId, readOnlyCollabId, 'investigator', 'manage')).rejects.toThrow(
        /Forbidden: Only the case creator, a supervisor, or an administrator can manage/
      );
    });

    it('permits editor collaborator for edit access, but blocks manage', async () => {
      jest.spyOn(Case, 'findOne').mockResolvedValue(baseCaseDoc as any);

      await expect(assertCaseAccess(caseId, collaboratorId, 'investigator', 'edit')).resolves.not.toThrow();
      await expect(assertCaseAccess(caseId, collaboratorId, 'investigator', 'manage')).rejects.toThrow(
        /Forbidden: Only the case creator, a supervisor, or an administrator can manage/
      );
    });

    it('blocks editing operations on archived cases', async () => {
      const archivedCaseDoc = { ...baseCaseDoc, status: 'archived' };
      jest.spyOn(Case, 'findOne').mockResolvedValue(archivedCaseDoc as any);

      await expect(assertCaseAccess(caseId, creatorId, 'investigator', 'edit')).rejects.toThrow(
        /Forbidden: This case has been archived and is read-only/
      );
    });
  });

  // =========================================================================
  // 3. Case Lifecycle Management (Resolve, Reopen, Archive, Restore, Delete)
  // =========================================================================
  describe('3. Case Lifecycle Management', () => {
    const creatorId = new Types.ObjectId().toString();
    const caseId = new Types.ObjectId().toString();

    it('resolves an open case, sets resolvedAt & resolvedBy, and creates CASE_RESOLVED audit log', async () => {
      const mockCase: any = {
        _id: new Types.ObjectId(caseId),
        caseNumber: 'CASE-001',
        title: 'Suspect Composite Investigation',
        status: 'open',
        isDeleted: false,
        createdBy: new Types.ObjectId(creatorId),
        assignedTo: [new Types.ObjectId(creatorId)],
        collaborators: [],
        save: jest.fn().mockResolvedValue(true),
      };

      jest.spyOn(Case, 'findOne').mockResolvedValue(mockCase as any);
      jest.spyOn(SecurityAudit, 'create').mockResolvedValue({} as any);

      const resolved = await caseService.resolveCase(
        caseId,
        {
          summary: 'Suspect positively identified through composite match; warrant served.',
          reason: 'Positive identification',
        },
        creatorId,
        'investigator'
      );

      expect(mockCase.status).toBe('resolved');
      expect(mockCase.resolutionSummary).toBe('Suspect positively identified through composite match; warrant served.');
      expect(mockCase.resolvedBy?.toString()).toBe(creatorId);
      expect(mockCase.resolvedAt).toBeInstanceOf(Date);
      expect(mockCase.save).toHaveBeenCalled();
      expect(SecurityAudit.create).toHaveBeenCalledWith(
        expect.objectContaining({
          event: 'CASE_RESOLVED',
          status: 'success',
        })
      );
      expect(resolved.status).toBe('resolved');
    });

    it('rejects resolving an already resolved or archived case', async () => {
      const alreadyResolvedCase = {
        _id: new Types.ObjectId(caseId),
        caseNumber: 'CASE-001',
        status: 'resolved',
        isDeleted: false,
        createdBy: new Types.ObjectId(creatorId),
        assignedTo: [new Types.ObjectId(creatorId)],
        save: jest.fn(),
      };

      jest.spyOn(Case, 'findOne').mockResolvedValue(alreadyResolvedCase as any);

      await expect(
        caseService.resolveCase(
          caseId,
          { summary: 'Attempt duplicate resolve' },
          creatorId,
          'investigator'
        )
      ).rejects.toThrow(ApiError);
    });

    it('reopens a resolved case, resets status to open, and creates CASE_REOPENED audit log', async () => {
      const resolvedCase: any = {
        _id: new Types.ObjectId(caseId),
        caseNumber: 'CASE-001',
        status: 'resolved',
        isDeleted: false,
        createdBy: new Types.ObjectId(creatorId),
        assignedTo: [new Types.ObjectId(creatorId)],
        save: jest.fn().mockResolvedValue(true),
      };

      jest.spyOn(Case, 'findOne').mockResolvedValue(resolvedCase as any);
      jest.spyOn(SecurityAudit, 'create').mockResolvedValue({} as any);

      const reopened = await caseService.reopenCase(
        caseId,
        { reason: 'New forensic evidence received from secondary witness.' },
        creatorId,
        'investigator'
      );

      expect(resolvedCase.status).toBe('open');
      expect(resolvedCase.reopenedBy?.toString()).toBe(creatorId);
      expect(resolvedCase.reopenReason).toBe('New forensic evidence received from secondary witness.');
      expect(SecurityAudit.create).toHaveBeenCalledWith(
        expect.objectContaining({
          event: 'CASE_REOPENED',
          status: 'success',
        })
      );
      expect(reopened.status).toBe('open');
    });

    it('archives an active or resolved case and creates CASE_ARCHIVED audit log', async () => {
      const activeCase: any = {
        _id: new Types.ObjectId(caseId),
        caseNumber: 'CASE-001',
        status: 'in_progress',
        isDeleted: false,
        createdBy: new Types.ObjectId(creatorId),
        assignedTo: [new Types.ObjectId(creatorId)],
        save: jest.fn().mockResolvedValue(true),
      };

      jest.spyOn(Case, 'findOne').mockResolvedValue(activeCase as any);
      jest.spyOn(SecurityAudit, 'create').mockResolvedValue({} as any);

      const archived = await caseService.archiveCase(
        caseId,
        { reason: 'Statute of limitations audit review.' },
        creatorId,
        'investigator'
      );

      expect(activeCase.status).toBe('archived');
      expect(activeCase.archivedBy?.toString()).toBe(creatorId);
      expect(SecurityAudit.create).toHaveBeenCalledWith(
        expect.objectContaining({
          event: 'CASE_ARCHIVED',
          status: 'success',
        })
      );
      expect(archived.status).toBe('archived');
    });

    it('restores an archived case back to active workflow and creates CASE_RESTORED audit log', async () => {
      const archivedCase: any = {
        _id: new Types.ObjectId(caseId),
        caseNumber: 'CASE-001',
        status: 'archived',
        isDeleted: false,
        createdBy: new Types.ObjectId(creatorId),
        assignedTo: [new Types.ObjectId(creatorId)],
        save: jest.fn().mockResolvedValue(true),
      };

      jest.spyOn(Case, 'findOne').mockResolvedValue(archivedCase as any);
      jest.spyOn(SecurityAudit, 'create').mockResolvedValue({} as any);

      const restored = await caseService.restoreCase(
        caseId,
        { reason: 'Cold case reopened by federal task force.' },
        creatorId,
        'investigator'
      );

      expect(archivedCase.status).toBe('open');
      expect(archivedCase.restoredBy?.toString()).toBe(creatorId);
      expect(SecurityAudit.create).toHaveBeenCalledWith(
        expect.objectContaining({
          event: 'CASE_RESTORED',
          status: 'success',
        })
      );
      expect(restored.status).toBe('open');
    });

    it('soft deletes a case, sets isDeleted=true, preserves forensic data, and creates CASE_DELETED audit log', async () => {
      const targetCase: any = {
        _id: new Types.ObjectId(caseId),
        caseNumber: 'CASE-001',
        title: 'Safely Soft Deleted Case',
        status: 'open',
        isDeleted: false,
        createdBy: new Types.ObjectId(creatorId),
        assignedTo: [new Types.ObjectId(creatorId)],
        save: jest.fn().mockResolvedValue(true),
      };

      jest.spyOn(Case, 'findOne').mockResolvedValue(targetCase as any);
      jest.spyOn(SecurityAudit, 'create').mockResolvedValue({} as any);

      await caseService.deleteCase(
        caseId,
        creatorId,
        'investigator',
        { reason: 'Duplicate case record merged into primary file.' }
      );

      expect(targetCase.isDeleted).toBe(true);
      expect(targetCase.deletedBy?.toString()).toBe(creatorId);
      expect(targetCase.deletionReason).toBe('Duplicate case record merged into primary file.');
      expect(SecurityAudit.create).toHaveBeenCalledWith(
        expect.objectContaining({
          event: 'CASE_DELETED',
          status: 'success',
        })
      );
    });

    it('blocks non-owner regular investigator from deleting a case (403)', async () => {
      const unauthorizedUserId = new Types.ObjectId().toString();
      const targetCase = {
        _id: new Types.ObjectId(caseId),
        caseNumber: 'CASE-001',
        status: 'open',
        isDeleted: false,
        createdBy: new Types.ObjectId(creatorId),
        assignedTo: [new Types.ObjectId(creatorId), new Types.ObjectId(unauthorizedUserId)],
        collaborators: [{ user: new Types.ObjectId(unauthorizedUserId), permission: 'editor' }],
        save: jest.fn(),
      };

      jest.spyOn(Case, 'findOne').mockResolvedValue(targetCase as any);

      await expect(
        caseService.deleteCase(caseId, unauthorizedUserId, 'investigator', {})
      ).rejects.toThrow(ApiError);
    });
  });

  // =========================================================================
  // 4. Dashboard Telemetry Alignment
  // =========================================================================
  describe('4. Dashboard Telemetry Alignment & Filter Consistency', () => {
    it('strictly excludes resolved, archived, and soft-deleted cases from active count', async () => {
      jest.spyOn(Case, 'countDocuments').mockResolvedValue(10 as any);
      jest.spyOn(Composite, 'countDocuments').mockResolvedValue(12 as any);
      jest.spyOn(Composite, 'find').mockReturnValue(createChainableQuery([]) as any);
      jest.spyOn(WitnessDescription, 'countDocuments').mockResolvedValue(8 as any);
      jest.spyOn(Report, 'countDocuments').mockResolvedValue(3 as any);
      jest.spyOn(CaseComment, 'countDocuments').mockResolvedValue(5 as any);
      jest.spyOn(User, 'countDocuments').mockResolvedValue(2 as any);
      jest.spyOn(User, 'findById').mockReturnValue(createChainableQuery({ name: 'Agent Bhumi', role: 'investigator' }) as any);
      jest.spyOn(SecurityAudit, 'countDocuments').mockResolvedValue(40 as any);
      jest.spyOn(mlClient, 'getModelStatus').mockResolvedValue({ isOnline: true } as any);

      // Return 4 accessible cases, where 2 are open, 1 is in_progress, and 1 is resolved
      jest.spyOn(Case, 'find').mockReturnValue(
        createChainableQuery([
          {
            _id: new Types.ObjectId(),
            caseNumber: 'CASE-ACTIVE-01',
            title: 'Active Case 1',
            status: 'open',
            createdAt: new Date(),
          },
          {
            _id: new Types.ObjectId(),
            caseNumber: 'CASE-ACTIVE-02',
            title: 'Active Case 2',
            status: 'in_progress',
            createdAt: new Date(),
          },
          {
            _id: new Types.ObjectId(),
            caseNumber: 'CASE-ACTIVE-03',
            title: 'Active Case 3',
            status: 'under_review',
            createdAt: new Date(),
          },
          {
            _id: new Types.ObjectId(),
            caseNumber: 'CASE-RESOLVED-01',
            title: 'Resolved Case',
            status: 'resolved',
            createdAt: new Date(),
          },
        ])
      );

      jest.spyOn(GenerationJob, 'find').mockReturnValue(createChainableQuery([]));
      jest.spyOn(SecurityAudit, 'find').mockReturnValue(createChainableQuery([]));

      const overview = await getDashboardOverview(new Types.ObjectId().toString());

      // activeCasesCount must be exactly 3 (excluding the resolved case)
      expect(overview.metrics.activeCases.value).toBe(3);
      expect(overview.metrics.databaseRecords.breakdown.cases).toBe(10);
    });
  });

  // =========================================================================
  // 5. Case List View Filters (getAllCases)
  // =========================================================================
  describe('5. Case List View Filters (getAllCases)', () => {
    const mockUserId = new Types.ObjectId().toString();

    it('applies correct query filter for view=active (excludes resolved and archived)', async () => {
      let capturedFilter: any = null;
      jest.spyOn(Case, 'find').mockImplementation(((filter: any) => {
        capturedFilter = filter;
        return createChainableQuery([]);
      }) as any);
      jest.spyOn(Case, 'countDocuments').mockResolvedValue(0 as any);

      await caseService.getAllCases({ view: 'active' }, mockUserId, 'investigator');

      expect(capturedFilter.isDeleted).toBe(false);
      expect(capturedFilter.status).toEqual({
        $in: ['open', 'in_progress', 'under_review'],
      });
    });

    it('applies correct query filter for view=resolved', async () => {
      let capturedFilter: any = null;
      jest.spyOn(Case, 'find').mockImplementation(((filter: any) => {
        capturedFilter = filter;
        return createChainableQuery([]);
      }) as any);
      jest.spyOn(Case, 'countDocuments').mockResolvedValue(0 as any);

      await caseService.getAllCases({ view: 'resolved' }, mockUserId, 'investigator');

      expect(capturedFilter.isDeleted).toBe(false);
      expect(capturedFilter.status).toEqual({
        $in: ['resolved', 'closed'],
      });
    });

    it('applies correct query filter for view=archived', async () => {
      let capturedFilter: any = null;
      jest.spyOn(Case, 'find').mockImplementation(((filter: any) => {
        capturedFilter = filter;
        return createChainableQuery([]);
      }) as any);
      jest.spyOn(Case, 'countDocuments').mockResolvedValue(0 as any);

      await caseService.getAllCases({ view: 'archived' }, mockUserId, 'investigator');

      expect(capturedFilter.isDeleted).toBe(false);
      expect(capturedFilter.status).toBe('archived');
    });
  });
});
