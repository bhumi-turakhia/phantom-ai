import request from 'supertest';
import mongoose from 'mongoose';
import fs from 'fs';
import path from 'path';
import app from '../src/app';
import { User } from '../src/models/User.model';
import { Case } from '../src/models/Case.model';
import { env } from '../src/config/env';
import {
  seedDatabase,
  validatePasscodePolicy,
  INSECURE_DEFAULT_PASSWORDS,
} from '../src/seed';
import {
  csrfProtection,
  generateCsrfToken,
  getTrustedOrigins,
  isOriginTrusted,
} from '../src/middlewares/csrf.middleware';
import express from 'express';
import cookieParser from 'cookie-parser';

describe('Phase 1 Final Security Hardening Test Suite', () => {
  beforeAll(async () => {
    if (mongoose.connection.readyState === 0) {
      await mongoose.connect(env.MONGODB_URI);
    }
  });

  afterAll(async () => {
    // Clean up temporary test seed records
    await User.deleteMany({ email: /.*@hardening-test\.gov$/ });
    await User.deleteMany({ email: 'seed-prod-test@hardening-test.com' });
    await mongoose.connection.close();
  });

  describe('TASK 1: Seed Credentials & Administrator Initialization Security', () => {
    test('1.1. No hardcoded default credentials in seed source files', () => {
      const seedSource = fs.readFileSync(path.join(__dirname, '../src/seed.ts'), 'utf8');
      // Must not assign ForensicDev2026!Secure as a fallback or default
      expect(seedSource).not.toMatch(/initialPasscode\s*=\s*['"]ForensicDev2026!Secure['"]/);
      expect(seedSource).not.toMatch(/default.*['"]ForensicDev2026!Secure['"]/i);

      // Verify that known compromised passwords are explicitly blacklisted
      expect(INSECURE_DEFAULT_PASSWORDS.has('ForensicDev2026!Secure')).toBe(true);
      expect(() => validatePasscodePolicy('ForensicDev2026!Secure', false)).toThrow(
        /Refusing insecure configuration/i
      );
    });

    test('1.2. Production seed refuses insecure configuration (short, weak, or default passwords)', () => {
      // Short password (< 12 chars in production)
      expect(() => validatePasscodePolicy('Short1!', true)).toThrow(
        /must be at least 12 characters/i
      );

      // Missing special character or digits in production
      expect(() => validatePasscodePolicy('AllLettersNoSpecialsOrDigits', true)).toThrow(
        /must contain uppercase, lowercase, digit, and special/i
      );

      // Default insecure password
      expect(() => validatePasscodePolicy('ForensicDev2026!Secure', true)).toThrow(
        /known default or compromised/i
      );
    });

    test('1.3. Cryptographically random one-time credential generated when no passcode configured', async () => {
      const email = `test.random.${Date.now()}@hardening-test.gov`;
      const result = await seedDatabase({
        email,
        silent: true,
        nodeEnv: 'development',
      });

      expect(result.action).toBe('created');
      expect(result.oneTimePasscodeGenerated).toBe(true);
      expect(result.oneTimePasscode).toBeDefined();
      expect(result.oneTimePasscode!.length).toBeGreaterThanOrEqual(24);

      // Verify stored record is hashed with Argon2id and never stored in plaintext
      const createdUser = await User.findOne({ email }).select('+password');
      expect(createdUser).toBeDefined();
      expect(createdUser?.password).not.toBe(result.oneTimePasscode);
      expect(createdUser?.password.startsWith('$argon2id$')).toBe(true);
    });

    test('1.4. Re-running seed preserves existing user records and does not overwrite credentials', async () => {
      const email = `test.preserve.${Date.now()}@hardening-test.gov`;
      const initial = await seedDatabase({
        email,
        passcode: 'PreservePasscode2026!@#',
        silent: true,
        nodeEnv: 'development',
      });

      expect(initial.action).toBe('created');

      const userBefore = await User.findOne({ email }).select('+password');
      const passwordHashBefore = userBefore?.password;

      // Re-run seed with no passcode specified: should preserve unchanged
      const secondRun = await seedDatabase({
        email,
        silent: true,
        nodeEnv: 'development',
      });

      expect(secondRun.action).toBe('verified_unchanged');

      const userAfter = await User.findOne({ email }).select('+password');
      expect(userAfter?.password).toBe(passwordHashBefore);
    });

    test('1.5. Safe procedure to update passcode and deactivate/reactivate account', async () => {
      const email = `test.deactivate.${Date.now()}@hardening-test.gov`;
      await seedDatabase({
        email,
        passcode: 'InitialPass2026!@#',
        silent: true,
        nodeEnv: 'development',
      });

      // Deactivate account safely
      const deactResult = await seedDatabase({
        email,
        deactivateAccount: true,
        silent: true,
      });
      expect(deactResult.action).toBe('deactivated');
      const deactivatedUser = await User.findOne({ email });
      expect(deactivatedUser?.isActive).toBe(false);

      // Reactivate account
      const reactResult = await seedDatabase({
        email,
        reactivateAccount: true,
        silent: true,
      });
      expect(reactResult.action).toBe('reactivated');
      const reactivatedUser = await User.findOne({ email });
      expect(reactivatedUser?.isActive).toBe(true);

      // Update passcode securely with Argon2id
      const updateResult = await seedDatabase({
        email,
        passcode: 'UpdatedSecurePasscode2026!$%',
        silent: true,
        nodeEnv: 'development',
      });
      expect(updateResult.action).toBe('updated_passcode');
      const updatedUser = await User.findOne({ email }).select('+password');
      expect(updatedUser?.password.startsWith('$argon2id$')).toBe(true);
    });
  });

  describe('TASK 2: CSRF & Origin Validation Hardening', () => {
    let testApp: express.Express;

    beforeAll(() => {
      testApp = express();
      testApp.use(cookieParser());
      testApp.use(express.json());
      testApp.use(csrfProtection);
      testApp.post('/api/test-mutation', (_req, res) => res.json({ success: true }));
      testApp.get('/api/test-safe', (_req, res) => res.json({ success: true }));
    });

    test('2.1. Untrusted Origin rejected on cookie-authenticated requests', async () => {
      const res = await request(testApp)
        .post('/api/test-mutation')
        .set('Cookie', 'access_token=valid-session-token')
        .set('Origin', 'https://attacker-domain.evil.com');

      expect(res.status).toBe(403);
      expect(res.body.message).toMatch(/CSRF verification failed: Untrusted request origin/i);
    });

    test('2.2. Untrusted Origin rejected on public state-changing requests (Login-CSRF protection)', async () => {
      const res = await request(app)
        .post('/api/v1/auth/login')
        .set('Origin', 'https://malicious-login-spoof.evil.com')
        .send({
          identifier: 'test@agency.gov',
          password: 'SomePassword123!',
        });

      expect(res.status).toBe(403);
      expect(res.body.message).toMatch(/CSRF verification failed: Untrusted request origin/i);
    });

    test('2.3. Malformed and null Origins rejected', async () => {
      const resMalformed = await request(testApp)
        .post('/api/test-mutation')
        .set('Cookie', 'access_token=valid-session-token')
        .set('Origin', 'invalid://not-a-valid-url');

      expect(resMalformed.status).toBe(403);
      expect(resMalformed.body.message).toMatch(/CSRF verification failed/i);

      const resNull = await request(testApp)
        .post('/api/test-mutation')
        .set('Cookie', 'access_token=valid-session-token')
        .set('Origin', 'null');

      expect(resNull.status).toBe(403);
      expect(resNull.body.message).toMatch(/CSRF verification failed/i);
    });

    test('2.4. Missing Origin rejected on cookie-authenticated requests', async () => {
      const res = await request(testApp)
        .post('/api/test-mutation')
        .set('Cookie', 'access_token=valid-session-token');

      expect(res.status).toBe(403);
      expect(res.body.message).toMatch(/Missing trusted origin on cookie-authenticated request/i);
    });

    test('2.5. Custom CSRF header cannot be accepted as sole proof of trust from arbitrary cross-origin request', async () => {
      // Untrusted origin + custom header: MUST be rejected
      const resUntrustedWithHeader = await request(testApp)
        .post('/api/test-mutation')
        .set('Cookie', 'access_token=valid-session-token')
        .set('Origin', 'https://attacker.evil.com')
        .set('X-Phantom-CSRF', '1')
        .set('X-Requested-With', 'XMLHttpRequest');

      expect(resUntrustedWithHeader.status).toBe(403);
      expect(resUntrustedWithHeader.body.message).toMatch(/Untrusted request origin/i);

      // Missing origin + custom header: MUST be rejected
      const resMissingOriginWithHeader = await request(testApp)
        .post('/api/test-mutation')
        .set('Cookie', 'access_token=valid-session-token')
        .set('X-Phantom-CSRF', '1');

      expect(resMissingOriginWithHeader.status).toBe(403);
      expect(resMissingOriginWithHeader.body.message).toMatch(/Missing trusted origin/i);
    });

    test('2.6. Missing CSRF token rejected where required (csrf_token cookie present without header)', async () => {
      const res = await request(testApp)
        .post('/api/test-mutation')
        .set('Cookie', 'access_token=valid-session-token; csrf_token=active-csrf-token-123')
        .set('Origin', 'http://localhost:5173');

      expect(res.status).toBe(403);
      expect(res.body.message).toMatch(/Missing CSRF token/i);
    });

    test('2.7. Invalid CSRF token rejected (mismatched header value)', async () => {
      const res = await request(testApp)
        .post('/api/test-mutation')
        .set('Cookie', 'access_token=valid-session-token; csrf_token=correct-csrf-token-456')
        .set('Origin', 'http://localhost:5173')
        .set('X-CSRF-Token', 'wrong-manipulated-csrf-token');

      expect(res.status).toBe(403);
      expect(res.body.message).toMatch(/Invalid CSRF token/i);
    });

    test('2.8. Valid CSRF token accepted (matching header and cookie with trusted origin)', async () => {
      const token = generateCsrfToken();
      const res = await request(testApp)
        .post('/api/test-mutation')
        .set('Cookie', `access_token=valid-session-token; csrf_token=${token}`)
        .set('Origin', 'http://localhost:5173')
        .set('X-CSRF-Token', token);

      expect(res.status).toBe(200);
      expect(res.body.success).toBe(true);
    });

    test('2.9. Bearer-only authenticated requests proceed according to actual authentication mechanism', async () => {
      const res = await request(testApp)
        .post('/api/test-mutation')
        .set('Authorization', 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dummy');

      expect(res.status).toBe(200);
      expect(res.body.success).toBe(true);
    });

    test('2.10. Preflight OPTIONS requests are handled cleanly without CSRF rejection', async () => {
      const res = await request(app)
        .options('/api/v1/cases')
        .set('Origin', 'http://localhost:5173')
        .set('Access-Control-Request-Method', 'POST');

      expect([200, 204]).toContain(res.status);
    });

    test('2.11. Credentialed CORS does not accept arbitrary origins', async () => {
      const res = await request(app)
        .options('/api/v1/cases')
        .set('Origin', 'https://unauthorized-origin.evil.com')
        .set('Access-Control-Request-Method', 'POST');

      // The Access-Control-Allow-Origin header must NOT reflect the untrusted origin or wildcard
      expect(res.headers['access-control-allow-origin']).toBeUndefined();
    });

    test('2.12. CSRF token issuance endpoint returns signed token and sets cookie', async () => {
      const res = await request(app).get('/api/v1/auth/csrf-token');

      expect(res.status).toBe(200);
      expect(res.body.success).toBe(true);
      expect(res.body.data.csrfToken).toBeDefined();

      const rawCookies = res.headers['set-cookie'];
      expect(rawCookies).toBeDefined();
      const cookies = Array.isArray(rawCookies) ? rawCookies : [rawCookies];
      const csrfCookie = cookies.find((c: string) => c.includes('csrf_token'));
      expect(csrfCookie).toBeDefined();
      // Non-HttpOnly for client script accessibility
      expect(csrfCookie).not.toMatch(/HttpOnly/i);
      expect(csrfCookie).toMatch(/SameSite/i);
    });

    test('2.13. Trusted origin extraction and normalization validates allowlist correctly', () => {
      const origins = getTrustedOrigins();
      expect(origins.length).toBeGreaterThan(0);
      expect(origins).toContain('http://localhost:5173');
      expect(isOriginTrusted('http://localhost:5173', origins)).toBe(true);
      expect(isOriginTrusted('http://localhost:5173/some/path', origins)).toBe(true);
      expect(isOriginTrusted('https://evil.com', origins)).toBe(false);
      expect(isOriginTrusted(null, origins)).toBe(false);
      expect(isOriginTrusted('null', origins)).toBe(false);
    });
  });

  describe('TASK 3: Duplicate Mongoose Index Verification', () => {
    test('3.1. Case schema has exactly one unique index definition on caseNumber', () => {
      const schemaIndexes = Case.schema.indexes();
      const caseNumberIndexes = schemaIndexes.filter(([indexKey]) => indexKey.caseNumber !== undefined);

      // Exactly 1 index definition on caseNumber in schema indexes
      expect(caseNumberIndexes.length).toBe(1);
      expect(caseNumberIndexes[0][0]).toEqual({ caseNumber: 1 });
      expect(caseNumberIndexes[0][1]?.unique).toBe(true);
    });

    test('3.2. Database preserves unique constraint on caseNumber without errors', async () => {
      const testCaseNumber = `CASE-HARDENING-${Date.now()}`;
      const firstCase = await Case.create({
        caseNumber: testCaseNumber,
        title: 'Original Evidence Case',
        createdBy: new mongoose.Types.ObjectId(),
      });

      expect(firstCase).toBeDefined();

      // Attempt duplicate insertion
      await expect(
        Case.create({
          caseNumber: testCaseNumber,
          title: 'Duplicate Collision Case',
          createdBy: new mongoose.Types.ObjectId(),
        })
      ).rejects.toThrow();

      await Case.findByIdAndDelete(firstCase._id);
    });
  });
});
