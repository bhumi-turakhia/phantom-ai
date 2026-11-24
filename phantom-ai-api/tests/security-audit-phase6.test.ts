import request from 'supertest';
import mongoose from 'mongoose';
import app from '../src/app';
import { User } from '../src/models/User.model';
import { Case } from '../src/models/Case.model';
import { WitnessDescription } from '../src/models/WitnessDescription.model';
import { Session } from '../src/models/Session.model';
import { env } from '../src/config/env';
import * as caseService from '../src/services/case.service';
import * as witnessDescriptionService from '../src/services/witnessDescription.service';
import jwt from 'jsonwebtoken';
import crypto from 'crypto';

describe('Phase 6.0: Production Security, RBAC & Deployment Readiness Audit Suite', () => {
  let invAUser: any;
  let invBUser: any;
  let supervisorUser: any;
  let invAToken: string;
  let invBToken: string;
  let caseA: any;
  let witnessA: any;

  const createTestSessionToken = async (user: any): Promise<string> => {
    const rawToken = jwt.sign(
      {
        id: user._id.toString(),
        investigatorId: user.investigatorId,
        role: user.role,
        email: user.email,
        name: user.name,
      },
      env.JWT_SECRET,
      { expiresIn: '1h' }
    );
    const tokenHash = crypto.createHash('sha256').update(rawToken).digest('hex');
    await Session.create({
      userId: user._id,
      investigatorId: user.investigatorId,
      tokenHash,
      isValid: true,
      expiresAt: new Date(Date.now() + 3600000),
    });
    return rawToken;
  };

  beforeAll(async () => {
    if (mongoose.connection.readyState === 0) {
      await mongoose.connect(env.MONGODB_URI);
    }

    // 1. Create Investigator A
    invAUser = await User.create({
      investigatorId: `INV-AUD-${Date.now().toString().slice(-6)}-A`,
      name: 'Audited Detective A',
      email: `det.a.${Date.now()}@agency.internal`,
      password: 'AuditSecurePassword123!',
      role: 'investigator',
      badgeNumber: '8801',
      isActive: true,
    });
    invAToken = await createTestSessionToken(invAUser);

    // 2. Create Investigator B
    invBUser = await User.create({
      investigatorId: `INV-AUD-${Date.now().toString().slice(-6)}-B`,
      name: 'Audited Detective B',
      email: `det.b.${Date.now()}@agency.internal`,
      password: 'AuditSecurePassword123!',
      role: 'investigator',
      badgeNumber: '8802',
      isActive: true,
    });
    invBToken = await createTestSessionToken(invBUser);

    // 3. Create Supervisor
    supervisorUser = await User.create({
      investigatorId: `SUP-AUD-${Date.now().toString().slice(-6)}`,
      name: 'Supervisory Agent',
      email: `sup.${Date.now()}@agency.internal`,
      password: 'AuditSecurePassword123!',
      role: 'supervisor',
      badgeNumber: '8800',
      isActive: true,
    });

    // 4. Create Case A assigned strictly to Investigator A
    caseA = await Case.create({
      caseNumber: `AUDIT-CASE-${Date.now().toString().slice(-5)}`,
      title: 'Restricted Forensic Homicide Dossier',
      description: 'Confidential investigation file with restricted investigator access',
      status: 'open',
      priority: 'high',
      createdBy: invAUser._id,
      assignedTo: [invAUser._id],
    });

    // 5. Create Witness statement on Case A
    witnessA = await WitnessDescription.create({
      case: caseA._id,
      witnessName: 'Protected Witness',
      recordedBy: invAUser._id,
      rawStatement: 'Male suspect in his 30s with short black hair and bushy eyebrows.',
      gender: 'male',
      build: 'average',
      confidenceLevel: 85,
    });
  });

  afterAll(async () => {
    await User.deleteMany({ email: /@agency\.internal$/ });
    await Case.deleteMany({ caseNumber: /^AUDIT-CASE-/ });
    if (caseA) {
      await WitnessDescription.deleteMany({ case: caseA._id });
      await Session.deleteMany({ userId: { $in: [invAUser._id, invBUser._id, supervisorUser._id] } });
    }
    await mongoose.connection.close();
  });

  describe('1. Case-Level Authorization & IDOR Defenses (CWE-639 / CWE-862)', () => {
    it('1.1. Investigator A can access their own case by ID', async () => {
      const caseDoc = await caseService.getCaseById(caseA._id.toString(), invAUser._id.toString(), 'investigator');
      expect(caseDoc).toBeDefined();
      expect(caseDoc.caseNumber).toBe(caseA.caseNumber);
    });

    it('1.2. Investigator B is forbidden from accessing Investigator A case by ID (403)', async () => {
      await expect(
        caseService.getCaseById(caseA._id.toString(), invBUser._id.toString(), 'investigator')
      ).rejects.toThrow(/Forbidden/);
    });

    it('1.3. Investigator B is forbidden from modifying Investigator A case (403)', async () => {
      await expect(
        caseService.updateCase(
          caseA._id.toString(),
          { title: 'Unauthorized Modification By Investigator B' },
          invBUser._id.toString(),
          'investigator'
        )
      ).rejects.toThrow(/Forbidden/);
    });

    it('1.4. Investigator B is forbidden from creating witness descriptions on Investigator A case (403)', async () => {
      await expect(
        witnessDescriptionService.createWitnessDescription(
          {
            case: caseA._id.toString(),
            rawStatement: 'Rogue statement from unassigned investigator.',
          } as any,
          invBUser._id.toString(),
          'investigator'
        )
      ).rejects.toThrow(/Forbidden/);
    });

    it('1.5. Investigator B is forbidden from reading witness descriptions of Investigator A case (403)', async () => {
      await expect(
        witnessDescriptionService.getWitnessDescriptionsByCase(
          caseA._id.toString(),
          {},
          invBUser._id.toString(),
          'investigator'
        )
      ).rejects.toThrow(/Forbidden/);
    });

    it('1.6. Investigator B is forbidden from retrieving specific witness description by ID (403)', async () => {
      await expect(
        witnessDescriptionService.getWitnessDescriptionById(
          witnessA._id.toString(),
          invBUser._id.toString(),
          'investigator'
        )
      ).rejects.toThrow(/Forbidden/);
    });

    it('1.7. Supervisor has global oversight and CAN access Investigator A case and witness statements', async () => {
      const caseDoc = await caseService.getCaseById(caseA._id.toString(), supervisorUser._id.toString(), 'supervisor');
      expect(caseDoc).toBeDefined();

      const witnessDoc = await witnessDescriptionService.getWitnessDescriptionById(
        witnessA._id.toString(),
        supervisorUser._id.toString(),
        'supervisor'
      );
      expect(witnessDoc).toBeDefined();
      expect(witnessDoc.witnessName).toBe('Protected Witness');
    });
  });

  describe('2. HTTP API Route Protection & RBAC Enforcement', () => {
    it('2.1. GET /api/v1/cases/:id returns 403 when accessed by unauthorized investigator', async () => {
      const res = await request(app)
        .get(`/api/v1/cases/${caseA._id}`)
        .set('Authorization', `Bearer ${invBToken}`);

      expect(res.status).toBe(403);
      expect(res.body.message).toMatch(/Forbidden/);
    });

    it('2.2. PATCH /api/v1/cases/:id returns 403 when unauthorized investigator attempts update', async () => {
      const res = await request(app)
        .patch(`/api/v1/cases/${caseA._id}`)
        .set('Authorization', `Bearer ${invBToken}`)
        .send({ title: 'Tampered Title' });

      expect(res.status).toBe(403);
      expect(res.body.message).toMatch(/Forbidden/);
    });

    it('2.3. DELETE /api/v1/cases/:id rejects unauthorized investigator with 403 (requires case creator or supervisor/admin)', async () => {
      const res = await request(app)
        .delete(`/api/v1/cases/${caseA._id}`)
        .set('Authorization', `Bearer ${invBToken}`);

      expect(res.status).toBe(403);
      expect(res.body.message).toMatch(/Forbidden/);
    });

    it('2.4. Unauthenticated request to protected route returns 401', async () => {
      const res = await request(app).get('/api/v1/cases');
      expect(res.status).toBe(401);
      expect(res.body.message).toMatch(/Authentication required/);
    });
  });

  describe('3. Session Revocation and Account Deactivation', () => {
    it('3.1. Immediate access denial when session is invalidated in database', async () => {
      // Invalidate session for Investigator A
      await Session.updateOne({ userId: invAUser._id }, { $set: { isValid: false } });

      const res = await request(app)
        .get(`/api/v1/cases/${caseA._id}`)
        .set('Authorization', `Bearer ${invAToken}`);

      expect(res.status).toBe(401);
      expect(res.body.message).toMatch(/Session is invalid or has been revoked/);

      // Restore session validity for remaining tests
      await Session.updateOne({ userId: invAUser._id }, { $set: { isValid: true } });
    });

    it('3.2. Deactivated investigator account is immediately blocked with 403 on active tokens', async () => {
      await User.updateOne({ _id: invAUser._id }, { $set: { isActive: false } });

      const res = await request(app)
        .get(`/api/v1/cases/${caseA._id}`)
        .set('Authorization', `Bearer ${invAToken}`);

      expect(res.status).toBe(403);
      expect(res.body.message).toMatch(/Account has been deactivated/);

      // Restore active status
      await User.updateOne({ _id: invAUser._id }, { $set: { isActive: true } });
    });
  });

  describe('4. Input Validation & Body Size Limits (DoS Protection)', () => {
    it('4.1. Large payloads exceeding 10kb body limit are rejected with 413 Payload Too Large', async () => {
      const hugeDescription = 'A'.repeat(15 * 1024); // 15 KB payload

      const res = await request(app)
        .post('/api/v1/cases')
        .set('Authorization', `Bearer ${invAToken}`)
        .send({
          caseNumber: 'OVERSIZE-01',
          title: 'Payload Limit Test',
          description: hugeDescription,
        });

      expect(res.status).toBe(413);
    });

    it('4.2. Invalid ObjectId syntax in URL params is rejected with 400 Bad Request', async () => {
      const res = await request(app)
        .get('/api/v1/cases/not-a-valid-objectid')
        .set('Authorization', `Bearer ${invAToken}`);

      expect(res.status).toBe(400);
    });
  });
});
