import jwt, { SignOptions } from 'jsonwebtoken';
import crypto from 'crypto';
import {
  generateRegistrationOptions,
  verifyRegistrationResponse,
  generateAuthenticationOptions,
  verifyAuthenticationResponse,
} from '@simplewebauthn/server';
import { User, IUser, UserRole, IPasskeyCredential } from '../models/User.model';
import { Session, ISession } from '../models/Session.model';
import { SecurityAudit } from '../models/SecurityAudit.model';
import { env } from '../config/env';
import { ApiError } from '../utils/apiResponse';
import { generateInvestigatorId, generateRecoveryCodes, verifyRecoveryCode } from '../utils/recoveryCode';
import type { RegisterInput, LoginInput, RecoveryInput, UpdateProfileInput } from '../validations/auth.validation';

interface AuthTokens {
  accessToken: string;
  refreshToken: string;
}

export interface SafeUser {
  id: string;
  investigatorId: string;
  name: string;
  email: string;
  role: UserRole;
  badgeNumber?: string;
  department?: string;
  hasPasskey: boolean;
  passkeyCount: number;
  isActive: boolean;
  lastLogin?: Date;
  createdAt: Date;
  updatedAt: Date;
}

export interface AuthSessionResult {
  user: SafeUser;
  tokens: AuthTokens;
  sessionId: string;
  recoveryCodes?: string[];
}

const getExpectedOrigins = (): string[] => {
  return env.WEBAUTHN_ORIGIN.split(',').map((o) => o.trim());
};

export const sanitizeUser = (user: IUser): SafeUser => ({
  id: user._id.toString(),
  investigatorId: user.investigatorId,
  name: user.name,
  email: user.email,
  role: user.role,
  badgeNumber: user.badgeNumber,
  department: user.department,
  hasPasskey: (user.passkeys && user.passkeys.length > 0) || false,
  passkeyCount: user.passkeys?.length || 0,
  isActive: user.isActive,
  lastLogin: user.lastLogin,
  createdAt: user.createdAt,
  updatedAt: user.updatedAt,
});

const hashToken = (token: string): string => {
  return crypto.createHash('sha256').update(token).digest('hex');
};

const generateAccessToken = (userId: string, role: UserRole, investigatorId: string): string => {
  return jwt.sign(
    { id: userId, role, investigatorId },
    env.JWT_SECRET,
    { expiresIn: env.JWT_EXPIRES_IN } as SignOptions
  );
};

const generateRefreshToken = (userId: string): string => {
  return jwt.sign(
    { id: userId, type: 'refresh' },
    env.JWT_SECRET,
    { expiresIn: '30d' } as SignOptions
  );
};

export const createInvestigatorSession = async (
  user: IUser,
  ipAddress?: string,
  userAgent?: string
): Promise<{ tokens: AuthTokens; session: ISession }> => {
  const accessToken = generateAccessToken(user._id.toString(), user.role, user.investigatorId);
  const refreshToken = generateRefreshToken(user._id.toString());
  const tokenDigest = hashToken(accessToken);

  // Expires in 7 days or matching JWT_EXPIRES_IN
  const expiresAt = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000);

  const session = await Session.create({
    userId: user._id,
    investigatorId: user.investigatorId,
    tokenHash: tokenDigest,
    ipAddress,
    userAgent,
    isValid: true,
    expiresAt,
  });

  return {
    tokens: { accessToken, refreshToken },
    session,
  };
};

export const registerUser = async (
  input: RegisterInput,
  ipAddress?: string,
  userAgent?: string
): Promise<{ user: SafeUser; preAuthToken: string; requirePasskeyEnrollment: true }> => {
  const existing = await User.findOne({ email: input.email.toLowerCase() });
  if (existing) {
    throw new ApiError('An account with this email already exists', 409);
  }

  // Generate a non-guessable investigator ID
  let assignedId = generateInvestigatorId(input.badgeNumber);
  while (await User.findOne({ investigatorId: assignedId })) {
    assignedId = generateInvestigatorId();
  }

  const user = await User.create({
    investigatorId: assignedId,
    name: input.name,
    email: input.email,
    password: input.password,
    role: input.role ?? 'investigator',
    badgeNumber: input.badgeNumber,
    department: input.department,
  });

  await SecurityAudit.create({
    userId: user._id,
    investigatorId: user.investigatorId,
    email: user.email,
    event: 'PASSKEY_REGISTER_INITIATED',
    status: 'success',
    ipAddress,
    userAgent,
    details: { message: 'Investigator registered, awaiting initial passkey enrollment' },
  });

  // Short-lived token for initial passkey enrollment
  const preAuthToken = jwt.sign(
    { id: user._id.toString(), investigatorId: user.investigatorId, step: 'enroll_passkey', type: 'pre_auth' },
    env.JWT_SECRET,
    { expiresIn: '15m' }
  );

  return {
    user: sanitizeUser(user),
    preAuthToken,
    requirePasskeyEnrollment: true,
  };
};

export const loginPasscode = async (
  input: LoginInput,
  ipAddress?: string,
  userAgent?: string
): Promise<{
  requirePasskey?: boolean;
  requirePasskeyEnrollment?: boolean;
  preAuthToken: string;
  options?: any;
  user: { investigatorId: string; name: string; email: string };
}> => {
  const identifier = (input.identifier || input.email || '').trim().toLowerCase();

  const user = await User.findOne({
    $or: [
      { email: identifier },
      { investigatorId: identifier.toUpperCase() },
    ],
  }).select('+password');

  if (!user) {
    await SecurityAudit.create({
      event: 'LOGIN_PASSCODE_FAILURE',
      status: 'failure',
      ipAddress,
      userAgent,
      details: { identifier, reason: 'User not found' },
    });
    throw new ApiError('Invalid Investigator ID/Email or Passcode', 401);
  }

  if (user.isLocked()) {
    const remainingMinutes = Math.ceil(
      (user.lockUntil!.getTime() - Date.now()) / (60 * 1000)
    );
    await SecurityAudit.create({
      userId: user._id,
      investigatorId: user.investigatorId,
      email: user.email,
      event: 'LOGIN_LOCKED_OUT',
      status: 'warning',
      ipAddress,
      userAgent,
      details: { remainingMinutes },
    });
    throw new ApiError(
      `Account locked due to consecutive failed attempts. Try again in ${remainingMinutes} minutes.`,
      423
    );
  }

  if (!user.isActive) {
    throw new ApiError('Your investigator account has been deactivated. Contact an administrator.', 403);
  }

  const { valid, needsRehash } = await user.comparePassword(input.password);

  if (!valid) {
    await user.incLoginAttempts();
    await SecurityAudit.create({
      userId: user._id,
      investigatorId: user.investigatorId,
      email: user.email,
      event: 'LOGIN_PASSCODE_FAILURE',
      status: 'failure',
      ipAddress,
      userAgent,
      details: { attempts: user.failedLoginAttempts + 1 },
    });
    throw new ApiError('Invalid Investigator ID/Email or Passcode', 401);
  }

  // Safe migration to Argon2id if hash was legacy bcrypt
  if (needsRehash) {
    user.password = input.password;
    await user.save();
  }

  await user.resetLoginAttempts();

  await SecurityAudit.create({
    userId: user._id,
    investigatorId: user.investigatorId,
    email: user.email,
    event: 'LOGIN_PASSCODE_SUCCESS',
    status: 'success',
    ipAddress,
    userAgent,
  });

  const hasEnrolledPasskeys = user.passkeys && user.passkeys.length > 0;

  if (hasEnrolledPasskeys) {
    // Generate WebAuthn authentication options
    const options = await generateAuthenticationOptions({
      rpID: env.RP_ID,
      allowCredentials: user.passkeys.map((pk) => ({
        id: pk.credentialID,
        transports: (pk.transports || []) as any,
      })),
      userVerification: 'preferred',
    });

    user.currentChallenge = {
      challenge: options.challenge,
      expiresAt: new Date(Date.now() + 5 * 60 * 1000),
      flow: 'authentication',
    };
    await user.save({ validateBeforeSave: false });

    const preAuthToken = jwt.sign(
      { id: user._id.toString(), investigatorId: user.investigatorId, step: 'mfa_passkey', type: 'pre_auth' },
      env.JWT_SECRET,
      { expiresIn: '5m' }
    );

    return {
      requirePasskey: true,
      preAuthToken,
      options,
      user: {
        investigatorId: user.investigatorId,
        name: user.name,
        email: user.email,
      },
    };
  } else {
    // Passkey enrollment required: passcode alone does NOT grant access
    const preAuthToken = jwt.sign(
      { id: user._id.toString(), investigatorId: user.investigatorId, step: 'enroll_passkey', type: 'pre_auth' },
      env.JWT_SECRET,
      { expiresIn: '15m' }
    );

    return {
      requirePasskeyEnrollment: true,
      preAuthToken,
      user: {
        investigatorId: user.investigatorId,
        name: user.name,
        email: user.email,
      },
    };
  }
};

export const getPasskeyRegistrationOptions = async (
  userId: string
): Promise<any> => {
  const user = await User.findById(userId);
  if (!user) {
    throw new ApiError('Investigator account not found', 404);
  }

  const options = await generateRegistrationOptions({
    rpName: env.RP_NAME,
    rpID: env.RP_ID,
    userID: Buffer.from(user._id.toString()),
    userName: user.email,
    userDisplayName: `${user.name} (${user.investigatorId})`,
    attestationType: 'none',
    excludeCredentials: user.passkeys.map((pk) => ({
      id: pk.credentialID,
      transports: (pk.transports || []) as any,
    })),
    authenticatorSelection: {
      residentKey: 'preferred',
      userVerification: 'preferred',
    },
  });

  user.currentChallenge = {
    challenge: options.challenge,
    expiresAt: new Date(Date.now() + 5 * 60 * 1000),
    flow: 'registration',
  };
  await user.save({ validateBeforeSave: false });

  return options;
};

export const verifyPasskeyRegistration = async (
  userId: string,
  response: any,
  deviceName?: string,
  ipAddress?: string,
  userAgent?: string
): Promise<AuthSessionResult> => {
  const user = await User.findById(userId);
  if (!user) {
    throw new ApiError('Investigator account not found', 404);
  }

  if (!user.currentChallenge || !user.currentChallenge.challenge) {
    throw new ApiError('No active passkey registration challenge found or challenge expired.', 400);
  }

  if (user.currentChallenge.expiresAt.getTime() < Date.now()) {
    throw new ApiError('Registration challenge has expired. Please retry enrollment.', 400);
  }

  if (user.currentChallenge.flow !== 'registration') {
    user.currentChallenge = undefined;
    await user.save({ validateBeforeSave: false });
    throw new ApiError('Invalid challenge flow: Challenge was not generated for registration.', 400);
  }

  const expectedChallenge = user.currentChallenge.challenge;
  user.currentChallenge = undefined; // One-time use to prevent challenge reuse

  let verification: any;
  try {
    verification = await verifyRegistrationResponse({
      response,
      expectedChallenge,
      expectedOrigin: getExpectedOrigins(),
      expectedRPID: env.RP_ID,
      requireUserVerification: false,
    });
  } catch (err: any) {
    await SecurityAudit.create({
      userId: user._id,
      investigatorId: user.investigatorId,
      email: user.email,
      event: 'LOGIN_PASSKEY_FAILURE',
      status: 'failure',
      ipAddress,
      userAgent,
      details: { error: err.message, context: 'registration' },
    });
    throw new ApiError(`Passkey registration verification failed: ${err.message}`, 400);
  }

  const { verified, registrationInfo } = verification;
  if (!verified || !registrationInfo) {
    throw new ApiError('Passkey verification failed.', 400);
  }

  const { credential, credentialDeviceType, credentialBackedUp } = registrationInfo;

  const newPasskey: IPasskeyCredential = {
    credentialID: credential.id,
    credentialPublicKey: Buffer.from(credential.publicKey).toString('base64url'),
    counter: credential.counter,
    transports: (response.response?.transports as string[]) || [],
    deviceType: credentialDeviceType,
    backedUp: credentialBackedUp,
    name: deviceName || 'Device Platform Authenticator',
    createdAt: new Date(),
  };

  user.passkeys.push(newPasskey);

  // Generate single-use recovery codes
  const { plainCodes, hashedCodes } = generateRecoveryCodes(6);
  user.recoveryCodes = hashedCodes.map((codeHash) => ({
    codeHash,
    used: false,
    createdAt: new Date(),
  }));

  user.lastLogin = new Date();
  await user.save({ validateBeforeSave: false });

  await SecurityAudit.create({
    userId: user._id,
    investigatorId: user.investigatorId,
    email: user.email,
    event: 'PASSKEY_REGISTER_SUCCESS',
    status: 'success',
    ipAddress,
    userAgent,
    details: {
      credentialId: credential.id,
      deviceType: credentialDeviceType,
      backedUp: credentialBackedUp,
    },
  });

  const { tokens, session } = await createInvestigatorSession(user, ipAddress, userAgent);

  return {
    user: sanitizeUser(user),
    tokens,
    sessionId: session._id.toString(),
    recoveryCodes: plainCodes,
  };
};

export const verifyPasskeyLogin = async (
  preAuthToken: string,
  response: any,
  ipAddress?: string,
  userAgent?: string
): Promise<AuthSessionResult> => {
  let decoded: any;
  try {
    decoded = jwt.verify(preAuthToken, env.JWT_SECRET);
  } catch (err) {
    throw new ApiError('Pre-authentication token has expired or is invalid. Please restart login.', 401);
  }

  const user = await User.findById(decoded.id);
  if (!user) {
    throw new ApiError('Investigator account not found.', 404);
  }

  if (!user.currentChallenge || !user.currentChallenge.challenge) {
    throw new ApiError('No pending challenge or challenge has already been used.', 400);
  }

  if (user.currentChallenge.expiresAt.getTime() < Date.now()) {
    throw new ApiError('Authentication challenge expired. Please retry.', 400);
  }

  if (user.currentChallenge.flow !== 'authentication') {
    user.currentChallenge = undefined;
    await user.save({ validateBeforeSave: false });
    throw new ApiError('Invalid challenge flow: Challenge was not generated for authentication.', 400);
  }

  const expectedChallenge = user.currentChallenge.challenge;
  // Clear challenge immediately to prevent reuse / replay
  user.currentChallenge = undefined;
  await user.save({ validateBeforeSave: false });

  const passkey = user.passkeys.find((pk) => pk.credentialID === response.id);
  if (!passkey) {
    await SecurityAudit.create({
      userId: user._id,
      investigatorId: user.investigatorId,
      email: user.email,
      event: 'LOGIN_PASSKEY_FAILURE',
      status: 'failure',
      ipAddress,
      userAgent,
      details: { reason: 'Unrecognized credential ID', id: response.id },
    });
    throw new ApiError('Passkey not recognized for this account.', 401);
  }

  let verification: any;
  try {
    verification = await verifyAuthenticationResponse({
      response,
      expectedChallenge,
      expectedOrigin: getExpectedOrigins(),
      expectedRPID: env.RP_ID,
      credential: {
        id: passkey.credentialID,
        publicKey: Buffer.from(passkey.credentialPublicKey, 'base64url'),
        counter: passkey.counter,
        transports: (passkey.transports || []) as any,
      },
      requireUserVerification: false,
    });
  } catch (err: any) {
    await SecurityAudit.create({
      userId: user._id,
      investigatorId: user.investigatorId,
      email: user.email,
      event: 'LOGIN_PASSKEY_FAILURE',
      status: 'failure',
      ipAddress,
      userAgent,
      details: { error: err.message },
    });
    throw new ApiError(`WebAuthn verification failed: ${err.message}`, 401);
  }

  const { verified, authenticationInfo } = verification;
  if (!verified) {
    throw new ApiError('Passkey authentication could not be cryptographically verified.', 401);
  }

  // Update authenticator counter to prevent clone/replay attacks
  passkey.counter = authenticationInfo.newCounter;
  user.lastLogin = new Date();
  await user.save({ validateBeforeSave: false });

  await SecurityAudit.create({
    userId: user._id,
    investigatorId: user.investigatorId,
    email: user.email,
    event: 'LOGIN_PASSKEY_SUCCESS',
    status: 'success',
    ipAddress,
    userAgent,
    details: {
      credentialId: passkey.credentialID,
      newCounter: authenticationInfo.newCounter,
    },
  });

  const { tokens, session } = await createInvestigatorSession(user, ipAddress, userAgent);

  return {
    user: sanitizeUser(user),
    tokens,
    sessionId: session._id.toString(),
  };
};

export const recoverAccount = async (
  input: RecoveryInput,
  ipAddress?: string,
  userAgent?: string
): Promise<{ preAuthToken: string; requirePasskeyEnrollment: true; message: string }> => {
  const identifier = (input.identifier || input.email || '').trim().toLowerCase();

  const user = await User.findOne({
    $or: [{ email: identifier }, { investigatorId: identifier.toUpperCase() }],
  }).select('+password');

  if (!user) {
    throw new ApiError('Invalid account credentials or recovery code', 401);
  }

  const { valid } = await user.comparePassword(input.password);
  if (!valid) {
    throw new ApiError('Invalid account credentials or recovery code', 401);
  }

  // Find matching unused recovery code
  const matchingCodeIndex = user.recoveryCodes.findIndex(
    (rc) => !rc.used && verifyRecoveryCode(input.recoveryCode, rc.codeHash)
  );

  if (matchingCodeIndex === -1) {
    await SecurityAudit.create({
      userId: user._id,
      investigatorId: user.investigatorId,
      email: user.email,
      event: 'RECOVERY_CODE_FAILED',
      status: 'failure',
      ipAddress,
      userAgent,
      details: { reason: 'Invalid or already used recovery code' },
    });
    throw new ApiError('Invalid or already used recovery code', 401);
  }

  // Single-use enforcement: mark code used immediately
  user.recoveryCodes[matchingCodeIndex].used = true;
  user.recoveryCodes[matchingCodeIndex].usedAt = new Date();
  await user.save({ validateBeforeSave: false });

  await SecurityAudit.create({
    userId: user._id,
    investigatorId: user.investigatorId,
    email: user.email,
    event: 'RECOVERY_CODE_USED',
    status: 'warning',
    ipAddress,
    userAgent,
    details: { message: 'Emergency single-use recovery code consumed. Prompting passkey re-enrollment.' },
  });

  // Return a pre-auth token specifically for mandatory passkey re-enrollment
  const preAuthToken = jwt.sign(
    { id: user._id.toString(), investigatorId: user.investigatorId, step: 'enroll_passkey', type: 'pre_auth' },
    env.JWT_SECRET,
    { expiresIn: '15m' }
  );

  return {
    preAuthToken,
    requirePasskeyEnrollment: true,
    message: 'Recovery code verified. You must now re-enroll a device passkey to secure your account.',
  };
};

export const listPasskeys = async (userId: string): Promise<any[]> => {
  const user = await User.findById(userId);
  if (!user) {
    throw new ApiError('User not found', 404);
  }

  return (user.passkeys || []).map((pk) => ({
    credentialID: pk.credentialID,
    name: pk.name,
    deviceType: pk.deviceType,
    backedUp: pk.backedUp,
    createdAt: pk.createdAt,
  }));
};

export const removePasskey = async (
  userId: string,
  credentialId: string,
  passwordConfirm: string,
  ipAddress?: string,
  userAgent?: string
): Promise<void> => {
  const user = await User.findById(userId).select('+password');
  if (!user) {
    throw new ApiError('User not found', 404);
  }

  const { valid } = await user.comparePassword(passwordConfirm);
  if (!valid) {
    throw new ApiError('Incorrect passcode confirmation. Passkey deletion aborted.', 401);
  }

  if (user.passkeys.length <= 1) {
    throw new ApiError('Cannot remove the only enrolled passkey. Add a new passkey before deleting this one.', 400);
  }

  user.passkeys = user.passkeys.filter((pk) => pk.credentialID !== credentialId);
  await user.save({ validateBeforeSave: false });

  await SecurityAudit.create({
    userId: user._id,
    investigatorId: user.investigatorId,
    email: user.email,
    event: 'PASSKEY_DELETED',
    status: 'warning',
    ipAddress,
    userAgent,
    details: { removedCredentialId: credentialId },
  });
};

export const refreshAccessToken = async (
  refreshToken: string,
  ipAddress?: string,
  userAgent?: string
): Promise<{ accessToken: string }> => {
  if (!refreshToken) {
    throw new ApiError('Refresh token is required', 400);
  }

  let decoded: { id: string; type: string };
  try {
    decoded = jwt.verify(refreshToken, env.JWT_SECRET) as { id: string; type: string };
  } catch (err) {
    if (err instanceof jwt.TokenExpiredError) {
      throw new ApiError('Refresh token has expired. Please log in again.', 401);
    }
    throw new ApiError('Invalid refresh token', 401);
  }

  if (decoded.type !== 'refresh') {
    throw new ApiError('Invalid token type', 401);
  }

  const user = await User.findById(decoded.id);
  if (!user || !user.isActive) {
    throw new ApiError('Investigator account not found or inactive', 401);
  }

  const accessToken = generateAccessToken(user._id.toString(), user.role, user.investigatorId);

  await SecurityAudit.create({
    userId: user._id,
    investigatorId: user.investigatorId,
    email: user.email,
    event: 'TOKEN_REFRESH',
    status: 'success',
    ipAddress,
    userAgent,
  });

  return { accessToken };
};

export const logoutUser = async (userId: string, sessionId?: string): Promise<void> => {
  if (sessionId) {
    await Session.findByIdAndUpdate(sessionId, { isValid: false, revokedAt: new Date() });
  } else {
    // Invalidate all active sessions for this user
    await Session.updateMany({ userId, isValid: true }, { isValid: false, revokedAt: new Date() });
  }

  await SecurityAudit.create({
    userId: userId as any,
    event: 'LOGOUT',
    status: 'success',
  });
};

export const getCurrentUser = async (userId: string): Promise<SafeUser> => {
  const user = await User.findById(userId);
  if (!user) {
    throw new ApiError('Investigator account not found', 404);
  }

  if (!user.isActive) {
    throw new ApiError('Account has been deactivated', 403);
  }

  return sanitizeUser(user);
};

export const updateUserProfile = async (
  userId: string,
  input: UpdateProfileInput,
  ipAddress?: string,
  userAgent?: string
): Promise<SafeUser> => {
  const user = await User.findById(userId).select('+password');
  if (!user) {
    throw new ApiError('Investigator account not found', 404);
  }

  if (!user.isActive) {
    throw new ApiError('Account has been deactivated', 403);
  }

  // Update permitted fields
  if (input.name && input.name.trim()) {
    user.name = input.name.trim();
  }
  if (input.badgeNumber !== undefined) {
    user.badgeNumber = input.badgeNumber.trim();
  }
  if (input.department !== undefined) {
    user.department = input.department.trim();
  }

  // Handle password update
  if (input.newPassword) {
    if (!input.currentPassword) {
      throw new ApiError('Current passcode is required to update passcode', 400);
    }
    const { valid } = await user.comparePassword(input.currentPassword);
    if (!valid) {
      throw new ApiError('Current passcode verification failed', 401);
    }
    user.password = input.newPassword;
  }

  await user.save();

  await SecurityAudit.create({
    userId: user._id,
    event: 'PROFILE_UPDATED',
    status: 'success',
    ipAddress,
    userAgent,
    details: {
      name: user.name,
      badgeNumber: user.badgeNumber,
      department: user.department,
      passwordChanged: Boolean(input.newPassword),
    },
  });

  return sanitizeUser(user);
};
