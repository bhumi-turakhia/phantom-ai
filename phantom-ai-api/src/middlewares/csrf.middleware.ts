import { Request, Response, NextFunction } from 'express';
import crypto from 'crypto';
import { env, isProduction } from '../config/env';
import { ApiResponse } from '../utils/apiResponse';

const SAFE_METHODS = new Set(['GET', 'HEAD', 'OPTIONS']);

/**
 * Returns normalized list of allowed origins from system configuration.
 * Disallows wildcard (*) configurations for credentialed requests.
 */
export const getTrustedOrigins = (): string[] => {
  const configured = [
    env.CORS_ORIGIN,
    env.WEBAUTHN_ORIGIN,
    'http://localhost:5173',
    'http://localhost:5001',
    'http://127.0.0.1:5173',
    'http://127.0.0.1:5001',
  ];

  const origins = new Set<string>();
  for (const item of configured) {
    if (!item) continue;
    for (const part of item.split(',')) {
      const trimmed = part.trim();
      if (!trimmed || trimmed === '*') continue;
      try {
        origins.add(new URL(trimmed).origin);
      } catch {
        // Discard invalid URLs
      }
    }
  }
  return Array.from(origins);
};

/**
 * Validates candidate against trusted origin allowlist.
 */
export const isOriginTrusted = (
  candidate: string | undefined | null,
  trustedOrigins: string[]
): boolean => {
  if (!candidate || candidate === 'null') return false;
  try {
    const candidateUrl = new URL(candidate);
    return trustedOrigins.includes(candidateUrl.origin);
  } catch {
    return false;
  }
};

/**
 * Generates a cryptographically strong synchronizer / double-submit CSRF token.
 */
export const generateCsrfToken = (): string => {
  const randomPart = crypto.randomBytes(24).toString('base64url');
  const hmac = crypto.createHmac('sha256', env.JWT_SECRET).update(randomPart).digest('base64url');
  return `${randomPart}.${hmac}`;
};

/**
 * Safely compares two tokens in constant time to prevent timing attacks.
 */
export const compareTokensSafe = (a: string, b: string): boolean => {
  try {
    const bufA = Buffer.from(a);
    const bufB = Buffer.from(b);
    if (bufA.length !== bufB.length) return false;
    return crypto.timingSafeEqual(bufA, bufB);
  } catch {
    return false;
  }
};

/**
 * Sets double-submit CSRF cookie accessible to client-side scripts.
 */
export const setCsrfCookie = (res: Response, token: string): void => {
  res.cookie('csrf_token', token, {
    httpOnly: false, // Must be readable by client script to submit in X-CSRF-Token header
    secure: isProduction,
    sameSite: isProduction ? 'strict' : 'lax',
    path: '/',
    domain: env.COOKIE_DOMAIN,
    maxAge: 7 * 24 * 60 * 60 * 1000,
  });
};

/**
 * Validates state-changing requests to protect against Cross-Site Request Forgery (CSRF).
 * Enforces defense-in-depth:
 * 1. Origin and Referer validation against explicit trusted-origin allowlist.
 * 2. Rejection of untrusted or malformed origins.
 * 3. Double-submit / synchronizer token validation when CSRF tokens are present/required.
 * 4. Pure Bearer-token requests without ambient cookies are handled appropriately.
 * 5. Custom CSRF headers cannot be accepted as sole proof of trust from arbitrary cross-origin requests.
 */
export const csrfProtection = (
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  // Safe HTTP methods do not mutate state and are exempt
  if (SAFE_METHODS.has(req.method)) {
    return next();
  }

  const trustedOrigins = getTrustedOrigins();
  const rawOrigin = req.headers.origin as string | undefined;
  const rawReferer = req.headers.referer as string | undefined;

  // 1. If Origin header is sent on a state-changing request, it MUST be valid and trusted
  if (rawOrigin !== undefined) {
    if (!isOriginTrusted(rawOrigin, trustedOrigins)) {
      ApiResponse.error(res, 'CSRF verification failed: Untrusted request origin.', 403);
      return;
    }
  }

  // 2. If Origin was omitted but Referer is present, validate Referer against trusted origins
  if (rawOrigin === undefined && rawReferer !== undefined) {
    if (!isOriginTrusted(rawReferer, trustedOrigins)) {
      ApiResponse.error(res, 'CSRF verification failed: Untrusted request origin.', 403);
      return;
    }
  }

  // Determine authentication mechanisms on this request
  const hasCookieAuth = !!(
    req.cookies &&
    (req.cookies.access_token || req.cookies.refresh_token || req.cookies.pre_auth_token)
  );
  const hasAuthHeader = !!req.headers.authorization;

  // Pure Bearer-token requests without ambient cookies are not subject to ambient browser CSRF
  if (!hasCookieAuth && hasAuthHeader) {
    return next();
  }

  // 3. For cookie-authenticated requests, enforce strict origin verification & CSRF token checks
  if (hasCookieAuth) {
    const effectiveOrigin = rawOrigin || rawReferer;

    if (!effectiveOrigin) {
      ApiResponse.error(
        res,
        'CSRF verification failed: Missing trusted origin on cookie-authenticated request.',
        403
      );
      return;
    }

    if (!isOriginTrusted(effectiveOrigin, trustedOrigins)) {
      ApiResponse.error(res, 'CSRF verification failed: Untrusted request origin.', 403);
      return;
    }

    // Double-submit token validation
    const cookieToken = req.cookies?.csrf_token || req.cookies?.['XSRF-TOKEN'];
    const headerToken =
      (req.headers['x-csrf-token'] as string | undefined) ||
      (req.headers['x-xsrf-token'] as string | undefined) ||
      (req.headers['x-phantom-csrf'] as string | undefined);

    if (cookieToken) {
      if (!headerToken) {
        ApiResponse.error(res, 'CSRF verification failed: Missing CSRF token.', 403);
        return;
      }

      if (!compareTokensSafe(headerToken, cookieToken)) {
        ApiResponse.error(res, 'CSRF verification failed: Invalid CSRF token.', 403);
        return;
      }
    } else if (headerToken) {
      // Header provided without corresponding CSRF cookie
      ApiResponse.error(res, 'CSRF verification failed: Missing CSRF cookie.', 403);
      return;
    }
  }

  next();
};

/**
 * Route-level middleware to strictly enforce CSRF synchronizer / double-submit token.
 */
export const requireCsrfToken = (
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  const cookieToken = req.cookies?.csrf_token || req.cookies?.['XSRF-TOKEN'];
  const headerToken =
    (req.headers['x-csrf-token'] as string | undefined) ||
    (req.headers['x-xsrf-token'] as string | undefined) ||
    (req.headers['x-phantom-csrf'] as string | undefined);

  if (!cookieToken || !headerToken) {
    ApiResponse.error(res, 'CSRF verification failed: Missing CSRF token.', 403);
    return;
  }

  if (!compareTokensSafe(headerToken, cookieToken)) {
    ApiResponse.error(res, 'CSRF verification failed: Invalid CSRF token.', 403);
    return;
  }

  next();
};
