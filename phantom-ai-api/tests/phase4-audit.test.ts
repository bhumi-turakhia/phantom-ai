import mongoose, { Types } from 'mongoose';
import { User } from '../src/models/User.model';
import { Case } from '../src/models/Case.model';
import { WitnessDescription } from '../src/models/WitnessDescription.model';
import { env } from '../src/config/env';
import * as witnessDescriptionService from '../src/services/witnessDescription.service';
import { ApiError } from '../src/utils/apiResponse';

describe('Phase 4.1: Final Post-Integration Audit & Case Authorization Suite', () => {
  let investigator1Id: Types.ObjectId;
  let investigator2Id: Types.ObjectId;
  let adminId: Types.ObjectId;
  let restrictedCaseId: Types.ObjectId;
  let openCaseId: Types.ObjectId;
  let unassignedCaseId: Types.ObjectId;
  let witnessDescId: Types.ObjectId;
  let unassignedWitnessDescId: Types.ObjectId;

  beforeAll(async () => {
    if (mongoose.connection.readyState === 0) {
      await mongoose.connect(env.MONGODB_URI);
    }

    // Create Investigator 1
    const inv1 = await User.create({
      name: 'Detective One',
      email: `det1.${Date.now()}@agency.gov`,
      password: 'SamplePassword123!',
      role: 'investigator',
      badgeNumber: '1001',
    });
    investigator1Id = inv1._id;

    // Create Investigator 2 (unassigned to restricted case)
    const inv2 = await User.create({
      name: 'Detective Two',
      email: `det2.${Date.now()}@agency.gov`,
      password: 'SamplePassword123!',
      role: 'investigator',
      badgeNumber: '1002',
    });
    investigator2Id = inv2._id;

    // Create Admin
    const admin = await User.create({
      name: 'Supervisor Admin',
      email: `admin.${Date.now()}@agency.gov`,
      password: 'SamplePassword123!',
      role: 'admin',
      badgeNumber: '0001',
    });
    adminId = admin._id;

    // Restricted Case: created by and assigned ONLY to Investigator 1
    const rCase = await Case.create({
      caseNumber: `RESTRICTED-${Date.now().toString().slice(-6)}`,
      title: 'Restricted Forensic Homicide File',
      createdBy: investigator1Id,
      assignedTo: [investigator1Id],
    });
    restrictedCaseId = rCase._id;

    // Open Case: assigned to Investigator 2
    const oCase = await Case.create({
      caseNumber: `OPEN-${Date.now().toString().slice(-6)}`,
      title: 'Open Burglary Investigation',
      createdBy: investigator2Id,
      assignedTo: [investigator2Id],
    });
    openCaseId = oCase._id;

    // Unassigned Case: created by Investigator 1 with NO assignees (assignedTo: [])
    const uCase = await Case.create({
      caseNumber: `UNASSIGNED-${Date.now().toString().slice(-6)}`,
      title: 'Unassigned Pending Case',
      createdBy: investigator1Id,
      assignedTo: [],
    });
    unassignedCaseId = uCase._id;

    // Witness Description on Unassigned Case
    const uDoc = await WitnessDescription.create({
      case: unassignedCaseId,
      recordedBy: investigator1Id,
      rawStatement: 'Witness saw a suspect in a blue jacket.',
      extractedAttributes: [],
      reviewStatus: 'pending',
    });
    unassignedWitnessDescId = uDoc._id;

    // Witness Description on Restricted Case
    const originalStatement = 'Witness observed a tall male with dark curly hair and a scar on his left cheek.';
    const wDoc = await WitnessDescription.create({
      case: restrictedCaseId,
      recordedBy: investigator1Id,
      rawStatement: originalStatement,
      extractedAttributes: [
        {
          category: 'hair',
          value: 'dark curly hair',
          sourceText: 'dark curly hair',
          start: 35,
          end: 51,
          evidenceState: 'explicit',
          extractedBy: 'char_bilstm_crf_phase3c2',
          confidence: 90,
        },
      ],
      reviewStatus: 'in_review',
    });
    witnessDescId = wDoc._id;
  });

  afterAll(async () => {
    // Clean up test documents
    await User.deleteMany({ _id: { $in: [investigator1Id, investigator2Id, adminId] } });
    await Case.deleteMany({ _id: { $in: [restrictedCaseId, openCaseId, unassignedCaseId] } });
    await WitnessDescription.deleteMany({ _id: { $in: [witnessDescId, unassignedWitnessDescId] } });
    await mongoose.connection.close();
  });

  test('1. Case Authorization: Unauthorized investigator cannot add corrections to restricted case', async () => {
    // Investigator 2 attempts to add a correction to Investigator 1's restricted case
    await expect(
      witnessDescriptionService.addCorrection(
        witnessDescId.toString(),
        {
          attributeCategory: 'hair',
          correctedValue: 'blonde hair',
          reason: 'Unauthorized attempt to modify testimony',
        },
        investigator2Id.toString(),
        'investigator'
      )
    ).rejects.toThrow(ApiError);

    try {
      await witnessDescriptionService.addCorrection(
        witnessDescId.toString(),
        {
          attributeCategory: 'hair',
          correctedValue: 'blonde hair',
          reason: 'Unauthorized attempt to modify testimony',
        },
        investigator2Id.toString(),
        'investigator'
      );
    } catch (err: any) {
      expect(err.statusCode).toBe(403);
      expect(err.message).toMatch(/not authorized/i);
    }
  });

  test('2. Case Authorization: Unauthorized investigator cannot approve restricted case analysis', async () => {
    await expect(
      witnessDescriptionService.updateApprovalStatus(
        witnessDescId.toString(),
        'approved',
        investigator2Id.toString(),
        'investigator'
      )
    ).rejects.toThrow(ApiError);

    try {
      await witnessDescriptionService.updateApprovalStatus(
        witnessDescId.toString(),
        'approved',
        investigator2Id.toString(),
        'investigator'
      );
    } catch (err: any) {
      expect(err.statusCode).toBe(403);
    }
  });

  test('3. Case Authorization: Assigned investigator can successfully record corrections', async () => {
    const updated = await witnessDescriptionService.addCorrection(
      witnessDescId.toString(),
      {
        attributeCategory: 'hair',
        previousValue: 'dark curly hair',
        correctedValue: 'short black curly hair',
        previousEvidenceState: 'explicit',
        correctedEvidenceState: 'explicit',
        reason: 'Witness clarified on second interview that hair was jet black',
      },
      investigator1Id.toString(),
      'investigator'
    );

    expect(updated).toBeDefined();
    expect(updated.corrections.length).toBe(1);
    expect(updated.corrections[0].attributeCategory).toBe('hair');
    expect(updated.corrections[0].correctedValue).toBe('short black curly hair');
    expect(updated.corrections[0].correctedBy._id.toString()).toBe(investigator1Id.toString());
    expect(updated.corrections[0].reason).toBe('Witness clarified on second interview that hair was jet black');

    // Verify original statement was NOT modified or overwritten
    expect(updated.rawStatement).toBe('Witness observed a tall male with dark curly hair and a scar on his left cheek.');
  });

  test('4. Case Authorization: Admin / Supervisor can override and approve analysis across cases', async () => {
    const approved = await witnessDescriptionService.updateApprovalStatus(
      witnessDescId.toString(),
      'approved',
      adminId.toString(),
      'admin'
    );

    expect(approved).toBeDefined();
    expect(approved.reviewStatus).toBe('approved');
    expect(approved.reviewedBy).toBeDefined();
    expect(approved.reviewedAt).toBeDefined();
  });

  test('5. Correction persistence: corrections and audit history persist upon retrieval', async () => {
    const doc = await witnessDescriptionService.getWitnessDescriptionById(witnessDescId.toString());
    expect(doc).toBeDefined();
    expect(doc.corrections.length).toBe(1);
    expect(doc.corrections[0].correctedValue).toBe('short black curly hair');
    expect(doc.reviewStatus).toBe('approved');
  });

  test('6. Unassigned Case Authorization: Unassigned investigator cannot access/modify case with empty assignedTo', async () => {
    // Investigator 2 (not creator, not assigned) attempts to add a correction to Investigator 1's unassigned case
    await expect(
      witnessDescriptionService.addCorrection(
        unassignedWitnessDescId.toString(),
        {
          attributeCategory: 'clothing',
          correctedValue: 'red jacket',
          reason: 'Unauthorized modification of unassigned case',
        },
        investigator2Id.toString(),
        'investigator'
      )
    ).rejects.toThrow(ApiError);

    try {
      await witnessDescriptionService.addCorrection(
        unassignedWitnessDescId.toString(),
        {
          attributeCategory: 'clothing',
          correctedValue: 'red jacket',
          reason: 'Unauthorized modification of unassigned case',
        },
        investigator2Id.toString(),
        'investigator'
      );
    } catch (err: any) {
      expect(err.statusCode).toBe(403);
      expect(err.message).toMatch(/not authorized to view or modify this case/i);
    }
  });

  test('7. Unassigned Case Authorization: Creator CAN access and modify case with empty assignedTo', async () => {
    const updated = await witnessDescriptionService.addCorrection(
      unassignedWitnessDescId.toString(),
      {
        attributeCategory: 'clothing',
        correctedValue: 'blue windbreaker jacket',
        reason: 'Creator clarification on unassigned case',
      },
      investigator1Id.toString(),
      'investigator'
    );

    expect(updated).toBeDefined();
    expect(updated.corrections.length).toBe(1);
    expect(updated.corrections[0].correctedValue).toBe('blue windbreaker jacket');
  });

  test('8. Unassigned Case Authorization: Admin / Supervisor CAN override and modify case with empty assignedTo', async () => {
    const approved = await witnessDescriptionService.updateApprovalStatus(
      unassignedWitnessDescId.toString(),
      'approved',
      adminId.toString(),
      'admin'
    );

    expect(approved).toBeDefined();
    expect(approved.reviewStatus).toBe('approved');
  });
});

