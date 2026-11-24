import { Request, Response } from 'express';
import { asyncHandler } from '../utils/asyncHandler';
import { ApiResponse } from '../utils/apiResponse';
import * as authService from '../services/auth.service';
import { isProduction, env } from '../config/env';
import { generateCsrfToken, setCsrfCookie } from '../middlewares/csrf.middleware';

const setAuthCookies = (res: Response, accessToken: string, refreshToken?: string) => {
  const isSecure = isProduction;
  const sameSite = isProduction ? ('strict' as const) : ('lax' as const);

  res.cookie('access_token', accessToken, {
    httpOnly: true,
    secure: isSecure,
    sameSite,
    maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days
    domain: env.COOKIE_DOMAIN,
    path: '/',
  });

  if (refreshToken) {
    res.cookie('refresh_token', refreshToken, {
      httpOnly: true,
      secure: isSecure,
      sameSite,
      maxAge: 30 * 24 * 60 * 60 * 1000, // 30 days
      domain: env.COOKIE_DOMAIN,
      path: '/',
    });
  }

  // Issue double-submit CSRF cookie
  const csrfToken = generateCsrfToken();
  setCsrfCookie(res, csrfToken);
};

const clearAuthCookies = (res: Response) => {
  res.clearCookie('access_token', { domain: env.COOKIE_DOMAIN, path: '/' });
  res.clearCookie('refresh_token', { domain: env.COOKIE_DOMAIN, path: '/' });
  res.clearCookie('pre_auth_token', { domain: env.COOKIE_DOMAIN, path: '/' });
  res.clearCookie('csrf_token', { domain: env.COOKIE_DOMAIN, path: '/' });
};

export const getCsrfToken = asyncHandler(async (_req: Request, res: Response) => {
  const token = generateCsrfToken();
  setCsrfCookie(res, token);
  return ApiResponse.success(res, 'CSRF token issued successfully', { csrfToken: token }, 200);
});

export const register = asyncHandler(async (req: Request, res: Response) => {
  const { name, email, password, role, badgeNumber, department } = req.body;
  const ipAddress = req.ip || req.socket.remoteAddress;
  const userAgent = req.headers['user-agent'];

  const result = await authService.registerUser(
    { name, email, password, role, badgeNumber, department },
    ipAddress,
    userAgent
  );

  res.cookie('pre_auth_token', result.preAuthToken, {
    httpOnly: true,
    secure: isProduction,
    sameSite: isProduction ? 'strict' : 'lax',
    maxAge: 15 * 60 * 1000,
    path: '/',
    domain: env.COOKIE_DOMAIN,
  });

  const csrfToken = generateCsrfToken();
  setCsrfCookie(res, csrfToken);

  return ApiResponse.success(
    res,
    'Investigator account created. Device passkey enrollment required to finalize security setup.',
    result,
    201
  );
});

export const login = asyncHandler(async (req: Request, res: Response) => {
  const { identifier, email, password } = req.body;
  const ipAddress = req.ip || req.socket.remoteAddress;
  const userAgent = req.headers['user-agent'];

  const result = await authService.loginPasscode(
    { identifier: identifier || email, email, password },
    ipAddress,
    userAgent
  );

  res.cookie('pre_auth_token', result.preAuthToken, {
    httpOnly: true,
    secure: isProduction,
    sameSite: isProduction ? 'strict' : 'lax',
    maxAge: 15 * 60 * 1000,
    path: '/',
    domain: env.COOKIE_DOMAIN,
  });

  const csrfToken = generateCsrfToken();
  setCsrfCookie(res, csrfToken);

  const message = result.requirePasskey
    ? 'Passcode verified. Please complete device user verification (Passkey/Touch ID).'
    : 'Passcode verified. Initial passkey enrollment required for this investigator account.';

  return ApiResponse.success(res, message, result, 200);
});

export const getPasskeyRegisterOptions = asyncHandler(async (req: Request, res: Response) => {
  const userId = (req as any).preAuthUser?.id || (req as any).user?.id;
  const options = await authService.getPasskeyRegistrationOptions(userId);

  return ApiResponse.success(res, 'Passkey registration options generated', options, 200);
});

export const verifyPasskeyRegister = asyncHandler(async (req: Request, res: Response) => {
  const userId = (req as any).preAuthUser?.id || (req as any).user?.id;
  const { response, deviceName } = req.body;
  const ipAddress = req.ip || req.socket.remoteAddress;
  const userAgent = req.headers['user-agent'];

  const result = await authService.verifyPasskeyRegistration(
    userId,
    response,
    deviceName,
    ipAddress,
    userAgent
  );

  setAuthCookies(res, result.tokens.accessToken, result.tokens.refreshToken);
  res.clearCookie('pre_auth_token', { domain: env.COOKIE_DOMAIN, path: '/' });

  return ApiResponse.success(
    res,
    'Passkey successfully registered and verified. Investigator authenticated.',
    result,
    200
  );
});

export const verifyPasskeyLogin = asyncHandler(async (req: Request, res: Response) => {
  const { preAuthToken, response } = req.body;
  const token = preAuthToken || req.cookies?.pre_auth_token;
  const ipAddress = req.ip || req.socket.remoteAddress;
  const userAgent = req.headers['user-agent'];

  const result = await authService.verifyPasskeyLogin(token, response, ipAddress, userAgent);

  setAuthCookies(res, result.tokens.accessToken, result.tokens.refreshToken);
  res.clearCookie('pre_auth_token', { domain: env.COOKIE_DOMAIN, path: '/' });

  return ApiResponse.success(
    res,
    'Cryptographic assertion verified. Investigator authenticated.',
    result,
    200
  );
});

export const recover = asyncHandler(async (req: Request, res: Response) => {
  const { identifier, email, password, recoveryCode } = req.body;
  const ipAddress = req.ip || req.socket.remoteAddress;
  const userAgent = req.headers['user-agent'];

  const result = await authService.recoverAccount(
    { identifier: identifier || email, email, password, recoveryCode },
    ipAddress,
    userAgent
  );

  res.cookie('pre_auth_token', result.preAuthToken, {
    httpOnly: true,
    secure: isProduction,
    sameSite: isProduction ? 'strict' : 'lax',
    maxAge: 15 * 60 * 1000,
    path: '/',
    domain: env.COOKIE_DOMAIN,
  });

  const csrfToken = generateCsrfToken();
  setCsrfCookie(res, csrfToken);

  return ApiResponse.success(res, result.message, result, 200);
});

export const listPasskeys = asyncHandler(async (req: Request, res: Response) => {
  const userId = (req as any).user?.id;
  const passkeys = await authService.listPasskeys(userId);

  return ApiResponse.success(res, 'Registered passkeys retrieved', passkeys, 200);
});

export const removePasskey = asyncHandler(async (req: Request, res: Response) => {
  const userId = (req as any).user?.id;
  const { credentialId } = req.params;
  const { password } = req.body;
  const ipAddress = req.ip || req.socket.remoteAddress;
  const userAgent = req.headers['user-agent'];

  await authService.removePasskey(userId, credentialId, password, ipAddress, userAgent);

  return ApiResponse.success(res, 'Passkey successfully removed', null, 200);
});

export const refreshToken = asyncHandler(async (req: Request, res: Response) => {
  const token = req.body.refreshToken || req.cookies?.refresh_token;
  const ipAddress = req.ip || req.socket.remoteAddress;
  const userAgent = req.headers['user-agent'];

  const result = await authService.refreshAccessToken(token, ipAddress, userAgent);
  setAuthCookies(res, result.accessToken);

  return ApiResponse.success(res, 'Session token refreshed successfully', result, 200);
});

export const logout = asyncHandler(async (req: Request, res: Response) => {
  const userId = (req as any).user?.id;
  const sessionId = (req as any).user?.sessionId;

  await authService.logoutUser(userId, sessionId);
  clearAuthCookies(res);

  return ApiResponse.success(res, 'Logout successful. Session terminated.', null, 200);
});

export const getMe = asyncHandler(async (req: Request, res: Response) => {
  const userId = (req as any).user?.id;
  const user = await authService.getCurrentUser(userId);

  return ApiResponse.success(res, 'Investigator profile fetched successfully', user, 200);
});

export const updateProfile = asyncHandler(async (req: Request, res: Response) => {
  const userId = (req as any).user?.id;
  const ipAddress = req.ip || req.socket.remoteAddress;
  const userAgent = req.headers['user-agent'];

  const updatedUser = await authService.updateUserProfile(userId, req.body, ipAddress, userAgent);

  return ApiResponse.success(res, 'Investigator profile updated successfully', updatedUser, 200);
});
