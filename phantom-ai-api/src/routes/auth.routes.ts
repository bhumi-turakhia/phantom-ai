import { Router } from 'express';
import {
  register,
  login,
  getPasskeyRegisterOptions,
  verifyPasskeyRegister,
  verifyPasskeyLogin,
  recover,
  listPasskeys,
  removePasskey,
  refreshToken,
  logout,
  getMe,
  updateProfile,
  getCsrfToken,
} from '../controllers/auth.controller';
import { validate } from '../middlewares/validate.middleware';
import { authenticate, authenticatePreAuth } from '../middlewares/auth.middleware';
import {
  registerSchema,
  loginSchema,
  passkeyVerifyLoginSchema,
  passkeyVerifyRegisterSchema,
  recoverySchema,
  removePasskeySchema,
  refreshTokenSchema,
  updateProfileSchema,
} from '../validations/auth.validation';

const router = Router();

/**
 * @route   POST /api/v1/auth/register
 * @desc    Register a new investigator account (requires follow-up passkey enrollment)
 * @access  Public
 */
router.post('/register', validate(registerSchema), register);

/**
 * @route   POST /api/v1/auth/login
 * @desc    Verify passcode and initiate WebAuthn MFA / Enrollment challenge
 * @access  Public
 */
router.post('/login', validate(loginSchema), login);

/**
 * @route   POST /api/v1/auth/passkey/register-options
 * @desc    Generate WebAuthn registration options for Touch ID / platform authenticator
 * @access  PreAuth / Authenticated
 */
router.post('/passkey/register-options', authenticatePreAuth, getPasskeyRegisterOptions);

/**
 * @route   POST /api/v1/auth/passkey/verify-register
 * @desc    Verify cryptographic passkey registration and issue session + recovery codes
 * @access  PreAuth / Authenticated
 */
router.post(
  '/passkey/verify-register',
  authenticatePreAuth,
  validate(passkeyVerifyRegisterSchema),
  verifyPasskeyRegister
);

/**
 * @route   POST /api/v1/auth/passkey/verify-login
 * @desc    Verify cryptographic passkey assertion and issue authenticated investigator session
 * @access  Public (requires preAuthToken)
 */
router.post(
  '/passkey/verify-login',
  validate(passkeyVerifyLoginSchema),
  verifyPasskeyLogin
);

/**
 * @route   POST /api/v1/auth/recover
 * @desc    Emergency account recovery via single-use hashed recovery code
 * @access  Public
 */
router.post('/recover', validate(recoverySchema), recover);

/**
 * @route   GET /api/v1/auth/passkeys
 * @desc    List enrolled passkeys for current investigator
 * @access  Private
 */
router.get('/passkeys', authenticate, listPasskeys);

/**
 * @route   DELETE /api/v1/auth/passkeys/:credentialId
 * @desc    Remove an enrolled passkey with passcode re-verification
 * @access  Private
 */
router.delete(
  '/passkeys/:credentialId',
  authenticate,
  validate(removePasskeySchema),
  removePasskey
);

/**
 * @route   POST /api/v1/auth/refresh-token
 * @desc    Refresh session access token
 * @access  Public (requires refresh token)
 */
router.post('/refresh-token', validate(refreshTokenSchema), refreshToken);

/**
 * @route   POST /api/v1/auth/logout
 * @desc    Terminate investigator session and clear cookies
 * @access  Private
 */
router.post('/logout', authenticate, logout);

/**
 * @route   GET /api/v1/auth/csrf-token
 * @desc    Issue CSRF token and set double-submit cookie
 * @access  Public
 */
router.get('/csrf-token', getCsrfToken);

/**
 * @route   GET /api/v1/auth/me
 * @desc    Get current authenticated investigator details
 * @access  Private
 */
router.get('/me', authenticate, getMe);

/**
 * @route   PUT /api/v1/auth/profile
 * @desc    Update permitted investigator profile attributes
 * @access  Private
 */
router.put('/profile', authenticate, validate(updateProfileSchema), updateProfile);

export default router;
