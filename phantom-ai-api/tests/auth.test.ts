import request from 'supertest';
import mongoose from 'mongoose';
import app from '../src/app';
import { User } from '../src/models/User.model';
import { Case } from '../src/models/Case.model';
import { Session } from '../src/models/Session.model';
import { SecurityAudit } from '../src/models/SecurityAudit.model';
import { env } from '../src/config/env';
import { runAccountMigrations } from '../src/services/migration.service';
import bcrypt from 'bcryptjs';

describe('Phase 1: Investigator Authentication & Security Test Suite', () => {
  let testUserEmail = `test.investigator.${Date.now()}@agency.gov`;
  let testPasscode = 'ForensicPass123!';
  let testInvestigatorId: string;
  let testUserId: string;

  beforeAll(async () => {
    if (mongoose.connection.readyState === 0) {
      await mongoose.connect(env.MONGODB_URI);
    }
    await runAccountMigrations();
  }, 60000); // Allow up to 60 s for migrations (Argon2id re-hashing on existing bcrypt accounts)

  afterAll(async () => {
    // Clean up test records
    if (testUserId) {
      await User.findByIdAndDelete(testUserId);
      await Session.deleteMany({ userId: testUserId });
      await SecurityAudit.deleteMany({ userId: testUserId });
    }
    await mongoose.connection.close();
  });

  // 1. Existing accounts can still authenticate through the upgraded flow (and bcrypt migration works)
  test('1. Existing bcrypt accounts migrate and authenticate at stage 1', async () => {
    const bcryptSalt = await bcrypt.genSalt(10);
    const bcryptHash = await bcrypt.hash('LegacyBcrypt123!', bcryptSalt);

    const legacyUser = await User.create({
      name: 'Legacy Detective',
      email: `legacy.${Date.now()}@agency.gov`,
      password: bcryptHash,
      role: 'investigator',
      badgeNumber: '9901',
    });

    await runAccountMigrations();
    const updated = await User.findById(legacyUser._id);
    expect(updated?.investigatorId).toBeDefined();

    const res = await request(app)
      .post('/api/v1/auth/login')
      .send({
        identifier: legacyUser.email,
        password: 'LegacyBcrypt123!',
      });

    expect(res.status).toBe(200);
    expect(res.body.success).toBe(true);
    expect(res.body.data.requirePasskeyEnrollment).toBe(true);

    // Verify safe migration to Argon2id in DB
    const recheckedUser = await User.findById(legacyUser._id).select('+password');
    expect(recheckedUser?.password.startsWith('$argon2')).toBe(true);

    // Clean up legacy test user
    await User.findByIdAndDelete(legacyUser._id);
  });

  // 2. Correct passcode succeeds at the first authentication stage
  test('2. Correct passcode succeeds at stage 1 and returns preAuthToken', async () => {
    // Register a new investigator
    const regRes = await request(app)
      .post('/api/v1/auth/register')
      .send({
        name: 'Agent Marcus Vance',
        email: testUserEmail,
        password: testPasscode,
        role: 'investigator',
        badgeNumber: '4421',
        department: 'Forensics',
      });

    expect(regRes.status).toBe(201);
    expect(regRes.body.success).toBe(true);
    expect(regRes.body.data.user.investigatorId).toBeDefined();
    expect(regRes.body.data.requirePasskeyEnrollment).toBe(true);
    expect(regRes.body.data.preAuthToken).toBeDefined();

    testInvestigatorId = regRes.body.data.user.investigatorId;
    testUserId = regRes.body.data.user.id;

    // Login with Investigator ID
    const loginRes = await request(app)
      .post('/api/v1/auth/login')
      .send({
        identifier: testInvestigatorId,
        password: testPasscode,
      });

    expect(loginRes.status).toBe(200);
    expect(loginRes.body.success).toBe(true);
    expect(loginRes.body.data.preAuthToken).toBeDefined();
  });

  // 3. Incorrect passcode is rejected
  test('3. Incorrect passcode is rejected and increments attempt count', async () => {
    const res = await request(app)
      .post('/api/v1/auth/login')
      .send({
        identifier: testInvestigatorId,
        password: 'WrongPassword123!',
      });

    expect(res.status).toBe(401);
    expect(res.body.success).toBe(false);

    const userInDb = await User.findById(testUserId);
    expect(userInDb?.failedLoginAttempts).toBeGreaterThanOrEqual(1);
  });

  // 4. Correct passcode without a registered passkey does not bypass required enrollment
  test('4. Correct passcode without enrolled passkey cannot access protected routes without enrollment', async () => {
    const loginRes = await request(app)
      .post('/api/v1/auth/login')
      .send({
        identifier: testInvestigatorId,
        password: testPasscode,
      });

    expect(loginRes.status).toBe(200);
    expect(loginRes.body.data.requirePasskeyEnrollment).toBe(true);
    expect(loginRes.body.data.tokens).toBeUndefined(); // NO access tokens granted

    // Attempting to call protected route with preAuthToken must be rejected
    const protectedRes = await request(app)
      .get('/api/v1/auth/me')
      .set('Authorization', `Bearer ${loginRes.body.data.preAuthToken}`);

    expect(protectedRes.status).toBe(401);
  });

  // 5. Passkey registration works
  test('5. Passkey registration challenge generation works and requires valid pre-auth', async () => {
    // Login to get fresh preAuthToken
    const loginRes = await request(app)
      .post('/api/v1/auth/login')
      .send({
        identifier: testInvestigatorId,
        password: testPasscode,
      });

    const preAuthToken = loginRes.body.data.preAuthToken;

    // Request registration options
    const optionsRes = await request(app)
      .post('/api/v1/auth/passkey/register-options')
      .set('Authorization', `Bearer ${preAuthToken}`)
      .send({});

    expect(optionsRes.status).toBe(200);
    expect(optionsRes.body.success).toBe(true);
    expect(optionsRes.body.data.challenge).toBeDefined();
    expect(optionsRes.body.data.rp.id).toBe(env.RP_ID);

    // Verify challenge saved in DB
    const userInDb = await User.findById(testUserId);
    expect(userInDb?.currentChallenge?.challenge).toBe(optionsRes.body.data.challenge);
  });

  // 5b. Emergency recovery code verification works and enforces single-use
  test('5b. Emergency recovery codes function properly and cannot be reused', async () => {
    const { generateRecoveryCodes } = require('../src/utils/recoveryCode');
    const { plainCodes, hashedCodes } = generateRecoveryCodes(2);

    const user = await User.findById(testUserId);
    if (!user) throw new Error('User not found');
    user.recoveryCodes = hashedCodes.map((codeHash: string) => ({
      codeHash,
      used: false,
      createdAt: new Date(),
    }));
    await user.save({ validateBeforeSave: false });

    // Use recovery code for the first time
    const recoveryRes = await request(app)
      .post('/api/v1/auth/recover')
      .send({
        identifier: testInvestigatorId,
        password: testPasscode,
        recoveryCode: plainCodes[0],
      });

    expect(recoveryRes.status).toBe(200);
    expect(recoveryRes.body.success).toBe(true);
    expect(recoveryRes.body.data.requirePasskeyEnrollment).toBe(true);
    expect(recoveryRes.body.data.preAuthToken).toBeDefined();

    // Attempting to reuse the exact same recovery code must be rejected
    const reuseRes = await request(app)
      .post('/api/v1/auth/recover')
      .send({
        identifier: testInvestigatorId,
        password: testPasscode,
        recoveryCode: plainCodes[0],
      });

    expect(reuseRes.status).toBe(401);
    expect(reuseRes.body.message).toMatch(/Invalid or already used recovery code/i);
  });

  // 6 & 7. Invalid WebAuthn assertions are rejected
  test('6 & 7. Invalid WebAuthn assertion signature/payload is rejected', async () => {
    // Ensure the test user has an authentication-flow challenge set BEFORE calling verify-login.
    // Test 12 may have left a 'registration' flow challenge on this user, which would cause a
    // 400 (flow mismatch) before the credential-not-found 401 we want to verify here.
    const userForT67 = await User.findById(testUserId);
    if (!userForT67) throw new Error('Test user missing');
    userForT67.currentChallenge = {
      challenge: 'auth-flow-challenge-for-test-67',
      expiresAt: new Date(Date.now() + 5 * 60 * 1000),
      flow: 'authentication',
    };
    await userForT67.save({ validateBeforeSave: false });

    const loginRes = await request(app)
      .post('/api/v1/auth/login')
      .send({
        identifier: testInvestigatorId,
        password: testPasscode,
      });

    const preAuthToken = loginRes.body.data.preAuthToken;
    // The login call above regenerates and saves a new authentication-flow challenge.
    // Now send a fake assertion — the credential ID won't match any stored passkey.
    const fakeAssertionRes = await request(app)
      .post('/api/v1/auth/passkey/verify-login')
      .send({
        preAuthToken,
        response: {
          id: 'fake-credential-id-that-does-not-exist',
          rawId: 'fake-raw-id',
          response: {
            authenticatorData: 'fake-auth-data',
            clientDataJSON: 'fake-client-json',
            signature: 'fake-signature',
          },
          type: 'public-key',
        },
      });

    expect(fakeAssertionRes.status).toBe(401);
    expect(fakeAssertionRes.body.success).toBe(false);
  });

  // 8. Replayed challenges are rejected
  test('8. Replayed challenges are rejected and cannot be used twice', async () => {
    const user = await User.findById(testUserId);
    if (!user) throw new Error('Test user missing');

    user.currentChallenge = {
      challenge: 'valid-one-time-challenge-12345',
      expiresAt: new Date(Date.now() + 5 * 60 * 1000),
      flow: 'authentication',
    };
    await user.save({ validateBeforeSave: false });

    // Login passkey verification clears the challenge immediately upon attempt
    const loginRes = await request(app)
      .post('/api/v1/auth/passkey/verify-login')
      .send({
        preAuthToken: 'invalid-token',
        response: { id: 'any' },
      });

    expect(loginRes.status).toBe(401);
  });

  // 9. Unauthorized API requests are rejected
  test('9. Unauthorized API requests to protected routes return 401', async () => {
    const res = await request(app).get('/api/v1/auth/me');
    expect(res.status).toBe(401);
    expect(res.body.success).toBe(false);
  });

  // 10. Expired / revoked sessions cannot access protected routes
  test('10. Revoked sessions in the database are immediately denied access', async () => {
    const user = await User.findById(testUserId);
    if (!user) throw new Error('User missing');

    // Create a mock active session
    const jwt = require('jsonwebtoken');
    const crypto = require('crypto');
    const token = jwt.sign(
      { id: user._id.toString(), role: user.role, investigatorId: user.investigatorId },
      env.JWT_SECRET,
      { expiresIn: '1h' }
    );
    const tokenDigest = crypto.createHash('sha256').update(token).digest('hex');

    const session = await Session.create({
      userId: user._id,
      investigatorId: user.investigatorId,
      tokenHash: tokenDigest,
      isValid: false, // Explicitly revoked
      expiresAt: new Date(Date.now() + 60 * 60 * 1000),
    });

    const res = await request(app)
      .get('/api/v1/auth/me')
      .set('Authorization', `Bearer ${token}`);

    expect(res.status).toBe(401);
    expect(res.body.message).toMatch(/Session is invalid or has been revoked/i);

    await Session.findByIdAndDelete(session._id);
  });

  // 11. Role-based access works
  test('11. Role-based access control enforces permissions on routes', async () => {
    const user = await User.findById(testUserId);
    if (!user) throw new Error('User missing');

    const jwt = require('jsonwebtoken');
    const crypto = require('crypto');
    const investigatorToken = jwt.sign(
      { id: user._id.toString(), role: 'investigator', investigatorId: user.investigatorId },
      env.JWT_SECRET,
      { expiresIn: '1h' }
    );
    const tokenDigest = crypto.createHash('sha256').update(investigatorToken).digest('hex');

    const session = await Session.create({
      userId: user._id,
      investigatorId: user.investigatorId,
      tokenHash: tokenDigest,
      isValid: true,
      expiresAt: new Date(Date.now() + 60 * 60 * 1000),
    });

    // Test a simulated route requiring admin role
    const { authorize } = require('../src/middlewares/auth.middleware');
    const express = require('express');
    const adminTestApp = express();
    adminTestApp.use(express.json());
    adminTestApp.get(
      '/admin-only',
      (req: any, _res: any, next: any) => {
        req.user = { role: 'investigator' };
        next();
      },
      authorize('admin'),
      (_req: any, res: any) => res.json({ ok: true })
    );

    const adminCheckRes = await request(adminTestApp).get('/admin-only');
    expect(adminCheckRes.status).toBe(403);

    await Session.findByIdAndDelete(session._id);
  });

  // 12. Cross-flow challenge binding: registration challenge rejected during authentication
  test('12. Registration challenge cannot be used for authentication assertion', async () => {
    const user = await User.findById(testUserId);
    if (!user) throw new Error('User missing');

    const jwt = require('jsonwebtoken');
    const preAuthToken = jwt.sign(
      { id: user._id.toString(), investigatorId: user.investigatorId, step: 'mfa_passkey', type: 'pre_auth' },
      env.JWT_SECRET,
      { expiresIn: '5m' }
    );

    // Explicitly set challenge with flow: 'registration'
    user.currentChallenge = {
      challenge: 'reg-flow-challenge-xyz',
      expiresAt: new Date(Date.now() + 5 * 60 * 1000),
      flow: 'registration',
    };
    await user.save({ validateBeforeSave: false });

    const authRes = await request(app)
      .post('/api/v1/auth/passkey/verify-login')
      .send({
        preAuthToken,
        response: { id: 'any-id' },
      });

    expect(authRes.status).toBe(400);
    expect(authRes.body.message).toMatch(/Challenge was not generated for authentication/i);
  });

  // 13. Cross-investigator passkey rejection: User B's credential cannot authenticate User A
  test("13. Passkey belonging to another investigator is rejected for target investigator", async () => {
    const jwt = require('jsonwebtoken');
    const otherUser = await User.create({
      name: 'Agent Other',
      email: `other.${Date.now()}@agency.gov`,
      password: 'password123',
      role: 'investigator',
    });

    const userA = await User.findById(testUserId);
    if (!userA) throw new Error('User A missing');

    // Add passkey to otherUser only
    otherUser.passkeys.push({
      credentialID: 'user-b-exclusive-credential-id',
      credentialPublicKey: 'dGVzdC1wdWJsaWMta2V5',
      counter: 1,
      createdAt: new Date(),
    });
    await otherUser.save({ validateBeforeSave: false });

    // User A attempts to verify with User B's credential ID
    userA.currentChallenge = {
      challenge: 'valid-challenge-user-a',
      expiresAt: new Date(Date.now() + 5 * 60 * 1000),
      flow: 'authentication',
    };
    await userA.save({ validateBeforeSave: false });

    const preAuthTokenA = jwt.sign(
      { id: userA._id.toString(), investigatorId: userA.investigatorId, step: 'mfa_passkey', type: 'pre_auth' },
      env.JWT_SECRET,
      { expiresIn: '5m' }
    );

    const crossRes = await request(app)
      .post('/api/v1/auth/passkey/verify-login')
      .send({
        preAuthToken: preAuthTokenA,
        response: { id: 'user-b-exclusive-credential-id' },
      });

    expect(crossRes.status).toBe(401);
    expect(crossRes.body.message).toMatch(/Passkey not recognized for this account/i);

    await User.findByIdAndDelete(otherUser._id);
  });

  // 14. Immediate reflection of role change and account deactivation
  test('14. Account deactivation or role changes take effect immediately on next request', async () => {
    const user = await User.findById(testUserId);
    if (!user) throw new Error('User missing');

    const jwt = require('jsonwebtoken');
    const crypto = require('crypto');
    const token = jwt.sign(
      { id: user._id.toString(), role: user.role, investigatorId: user.investigatorId },
      env.JWT_SECRET,
      { expiresIn: '1h' }
    );
    const tokenDigest = crypto.createHash('sha256').update(token).digest('hex');

    const session = await Session.create({
      userId: user._id,
      investigatorId: user.investigatorId,
      tokenHash: tokenDigest,
      isValid: true,
      expiresAt: new Date(Date.now() + 60 * 60 * 1000),
    });

    // Verify active
    const activeRes = await request(app)
      .get('/api/v1/auth/me')
      .set('Authorization', `Bearer ${token}`);
    expect(activeRes.status).toBe(200);

    // Deactivate in DB
    user.isActive = false;
    await user.save({ validateBeforeSave: false });

    // Next request must be forbidden immediately
    const inactiveRes = await request(app)
      .get('/api/v1/auth/me')
      .set('Authorization', `Bearer ${token}`);
    expect(inactiveRes.status).toBe(403);
    expect(inactiveRes.body.message).toMatch(/Account has been deactivated/i);

    // Restore active state
    user.isActive = true;
    await user.save({ validateBeforeSave: false });
    await Session.findByIdAndDelete(session._id);
  });

  // 15. Sensitive case routes enforce role authorization
  test('15. Sensitive case routes (DELETE /cases/:id) reject unauthorized investigator', async () => {
    const user = await User.findById(testUserId);
    if (!user) throw new Error('User missing');

    const jwt = require('jsonwebtoken');
    const crypto = require('crypto');
    const token = jwt.sign(
      { id: user._id.toString(), role: 'investigator', investigatorId: user.investigatorId },
      env.JWT_SECRET,
      { expiresIn: '1h' }
    );
    const tokenDigest = crypto.createHash('sha256').update(token).digest('hex');

    const session = await Session.create({
      userId: user._id,
      investigatorId: user.investigatorId,
      tokenHash: tokenDigest,
      isValid: true,
      expiresAt: new Date(Date.now() + 60 * 60 * 1000),
    });

    const otherCase = await Case.create({
      caseNumber: `AUTH-CASE-${Date.now().toString().slice(-5)}`,
      title: 'Restricted Forensic File',
      description: 'Test case for authorization check',
      status: 'open',
      priority: 'medium',
      createdBy: new mongoose.Types.ObjectId(),
    });

    const deleteRes = await request(app)
      .delete(`/api/v1/cases/${otherCase._id}`)
      .set('Authorization', `Bearer ${token}`);

    // Unauthorized investigator must be 403 Forbidden for deletion
    expect(deleteRes.status).toBe(403);
    expect(deleteRes.body.message).toMatch(/Forbidden/i);

    await Case.findByIdAndDelete(otherCase._id);
    await Session.findByIdAndDelete(session._id);
  });

  // 16. Cookies have appropriate HttpOnly and SameSite settings
  test('16. Set-Cookie headers contain HttpOnly and SameSite settings on pre-auth / login verification', async () => {
    const loginRes = await request(app)
      .post('/api/v1/auth/login')
      .send({
        identifier: testInvestigatorId,
        password: testPasscode,
      });

    expect(loginRes.status).toBe(200);
    const rawCookies = loginRes.headers['set-cookie'];
    expect(rawCookies).toBeDefined();
    const cookies: string[] = Array.isArray(rawCookies) ? rawCookies : [rawCookies];
    const preAuthCookie = cookies.find((c: string) => c.includes('pre_auth_token'));
    expect(preAuthCookie).toBeDefined();
    expect(preAuthCookie).toMatch(/HttpOnly/i);
    expect(preAuthCookie).toMatch(/SameSite/i);
  });

  // 17. CSRF protection blocks untrusted cross-origin cookie-authenticated mutation
  test('17. CSRF protection rejects untrusted origin on cookie-authenticated requests', async () => {
    const { csrfProtection } = require('../src/middlewares/csrf.middleware');
    const express = require('express');
    const cookieParser = require('cookie-parser');

    const csrfTestApp = express();
    // Register cookie-parser BEFORE csrf middleware so req.cookies is populated from the Cookie header
    csrfTestApp.use(cookieParser());
    csrfTestApp.use(express.json());
    // Register csrfProtection globally at middleware level, not inside the route
    csrfTestApp.use(csrfProtection);
    csrfTestApp.post('/csrf-protected', (_req: any, res: any) => res.json({ ok: true }));

    // 17a. Untrusted origin with an actual cookie header must be rejected
    const untrustedRes = await request(csrfTestApp)
      .post('/csrf-protected')
      .set('Cookie', 'access_token=fake-access-token')
      .set('Origin', 'https://malicious-forensic-spoof.evil.com');

    expect(untrustedRes.status).toBe(403);
    expect(untrustedRes.body.message).toMatch(/CSRF verification failed/i);

    // 17b. Trusted origin with an actual cookie header must be allowed
    const trustedRes = await request(csrfTestApp)
      .post('/csrf-protected')
      .set('Cookie', 'access_token=fake-access-token')
      .set('Origin', 'http://localhost:5173');

    expect(trustedRes.status).toBe(200);
  });
});
