import api, { setAccessToken } from './api';
import { startRegistration, startAuthentication } from '@simplewebauthn/browser';

export interface InvestigatorUser {
  id: string;
  investigatorId: string;
  name: string;
  email: string;
  role: 'admin' | 'investigator' | 'supervisor' | 'analyst';
  badgeNumber?: string;
  department?: string;
  hasPasskey: boolean;
  passkeyCount: number;
}

export interface PreAuthResult {
  requirePasskey?: boolean;
  requirePasskeyEnrollment?: boolean;
  preAuthToken: string;
  options?: any;
  user: {
    investigatorId: string;
    name: string;
    email: string;
  };
}

export interface AuthSessionResponse {
  user: InvestigatorUser;
  tokens: {
    accessToken: string;
    refreshToken: string;
  };
  sessionId: string;
  recoveryCodes?: string[];
}

/**
 * Step 1: Initial passcode verification.
 * Returns either WebAuthn assertion challenge or registration requirement.
 */
export const loginPasscode = async (
  identifier: string,
  passcode: string
): Promise<PreAuthResult> => {
  const response = await api.post('/auth/login', {
    identifier,
    password: passcode,
  });

  return response.data.data;
};

/**
 * Step 2A: Complete WebAuthn Passkey Assertion (Touch ID / Face ID / Platform Key).
 */
export const authenticateWithPasskey = async (
  preAuthToken: string,
  options: any
): Promise<AuthSessionResponse> => {
  // Trigger platform authenticator (Touch ID, Windows Hello, etc.)
  const assertionResponse = await startAuthentication({ optionsJSON: options });

  const response = await api.post('/auth/passkey/verify-login', {
    preAuthToken,
    response: assertionResponse,
  });

  const sessionData: AuthSessionResponse = response.data.data;
  if (sessionData.tokens?.accessToken) {
    setAccessToken(sessionData.tokens.accessToken);
  }

  return sessionData;
};

/**
 * Step 2B: Register a new device passkey (Touch ID / Platform Authenticator).
 */
export const registerPasskey = async (
  preAuthToken: string,
  deviceName?: string
): Promise<AuthSessionResponse> => {
  // Fetch registration challenge options
  const optionsRes = await api.post(
    '/auth/passkey/register-options',
    {},
    {
      headers: { Authorization: `Bearer ${preAuthToken}` },
    }
  );

  const options = optionsRes.data.data;

  // Trigger platform authenticator to create key pair
  const attestationResponse = await startRegistration({ optionsJSON: options });

  // Verify on backend
  const verifyRes = await api.post(
    '/auth/passkey/verify-register',
    {
      response: attestationResponse,
      deviceName: deviceName || 'Device Platform Authenticator',
    },
    {
      headers: { Authorization: `Bearer ${preAuthToken}` },
    }
  );

  const sessionData: AuthSessionResponse = verifyRes.data.data;
  if (sessionData.tokens?.accessToken) {
    setAccessToken(sessionData.tokens.accessToken);
  }

  return sessionData;
};

/**
 * Register a new investigator account.
 */
export const register = async (
  name: string,
  email: string,
  passcode: string,
  badgeNumber?: string,
  department?: string
): Promise<{ user: InvestigatorUser; preAuthToken: string }> => {
  const response = await api.post('/auth/register', {
    name,
    email,
    password: passcode,
    badgeNumber,
    department,
  });

  return response.data.data;
};

/**
 * Emergency account recovery using single-use recovery code.
 */
export const recoverAccount = async (
  identifier: string,
  passcode: string,
  recoveryCode: string
): Promise<{ preAuthToken: string; message: string }> => {
  const response = await api.post('/auth/recover', {
    identifier,
    password: passcode,
    recoveryCode,
  });

  return response.data.data;
};

export const listPasskeys = async (): Promise<any[]> => {
  const response = await api.get('/auth/passkeys');
  return response.data.data;
};

export const removePasskey = async (
  credentialId: string,
  passcode: string
): Promise<void> => {
  await api.delete(`/auth/passkeys/${credentialId}`, {
    data: { password: passcode },
  });
};

export const getCurrentUser = async (): Promise<InvestigatorUser> => {
  const response = await api.get('/auth/me');
  return response.data.data;
};

export interface UpdateProfileInput {
  name?: string;
  badgeNumber?: string;
  department?: string;
  currentPassword?: string;
  newPassword?: string;
}

export const updateProfile = async (data: UpdateProfileInput): Promise<InvestigatorUser> => {
  const response = await api.put('/auth/profile', data);
  const updatedUser = response.data.data;
  if (typeof window !== 'undefined') {
    localStorage.setItem('user', JSON.stringify(updatedUser));
    window.dispatchEvent(new CustomEvent('phantom:user_updated', { detail: updatedUser }));
  }
  return updatedUser;
};

export const logout = async (): Promise<void> => {
  try {
    await api.post('/auth/logout');
  } finally {
    setAccessToken(null);
    if (typeof window !== 'undefined') {
      localStorage.removeItem('user');
      localStorage.removeItem('access_token');
      localStorage.removeItem('role');
      window.dispatchEvent(new CustomEvent('phantom:user_updated', { detail: null }));
    }
  }
};
