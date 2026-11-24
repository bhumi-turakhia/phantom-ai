import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import crypto from 'crypto';
import { env } from '../config/env';
import { User, UserRole } from '../models/User.model';
import { Session } from '../models/Session.model';
import { ApiResponse } from '../utils/apiResponse';

export interface AuthenticatedUser {
  id: string;
  investigatorId: string;
  role: UserRole;
  email: string;
  name: string;
  sessionId?: string;
}

export interface PreAuthUser {
  id: string;
  investigatorId: string;
  step: 'mfa_passkey' | 'enroll_passkey' | 'recovery_enroll';
}

const hashToken = (token: string): string => {
  return crypto.createHash('sha256').update(token).digest('hex');
};

export const authenticate = async (
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    let token: string | undefined;

    const authHeader = req.headers.authorization;
    if (authHeader && authHeader.startsWith('Bearer ')) {
      token = authHeader.split(' ')[1];
    } else if (req.cookies && req.cookies.access_token) {
      token = req.cookies.access_token;
    }

    if (!token) {
      ApiResponse.error(res, 'Authentication required. No session or token provided.', 401);
      return;
    }

    let decoded: any;
    try {
      decoded = jwt.verify(token, env.JWT_SECRET);
    } catch (err) {
      if (err instanceof jwt.TokenExpiredError) {
        ApiResponse.error(res, 'Session token has expired. Please log in again.', 401);
        return;
      }
      if (err instanceof jwt.JsonWebTokenError) {
        ApiResponse.error(res, 'Invalid session token. Please log in again.', 401);
        return;
      }
      ApiResponse.error(res, 'Token verification failed', 401);
      return;
    }

    if (decoded.type === 'pre_auth') {
      ApiResponse.error(res, 'Incomplete authentication. Passkey verification required.', 401);
      return;
    }

    // Check session in database if sessionId or tokenHash exists
    const tokenDigest = hashToken(token);
    const session = await Session.findOne({
      userId: decoded.id,
      isValid: true,
      $or: [{ tokenHash: tokenDigest }, { tokenHash: decoded.sessionHash || 'legacy' }],
    });

    if (!session || session.expiresAt.getTime() <= Date.now()) {
      ApiResponse.error(res, 'Session is invalid or has been revoked. Please re-authenticate.', 401);
      return;
    }

    const user = await User.findById(decoded.id).select('-password');
    if (!user) {
      ApiResponse.error(res, 'Investigator account no longer exists', 401);
      return;
    }

    if (!user.isActive) {
      ApiResponse.error(res, 'Account has been deactivated. Contact an administrator.', 403);
      return;
    }

    (req as any).user = {
      id: user._id.toString(),
      investigatorId: user.investigatorId,
      role: user.role,
      email: user.email,
      name: user.name,
      sessionId: session._id.toString(),
    };

    next();
  } catch (err) {
    next(err);
  }
};

/**
 * Middleware for pre-auth steps (WebAuthn registration or assertion before full session).
 */
export const authenticatePreAuth = async (
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    let token: string | undefined;

    const authHeader = req.headers.authorization;
    if (authHeader && authHeader.startsWith('Bearer ')) {
      token = authHeader.split(' ')[1];
    } else if (req.body && req.body.preAuthToken) {
      token = req.body.preAuthToken;
    } else if (req.cookies && req.cookies.pre_auth_token) {
      token = req.cookies.pre_auth_token;
    }

    // Also allow already fully authenticated users (e.g. adding an extra passkey from profile)
    if (token) {
      try {
        const decoded: any = jwt.verify(token, env.JWT_SECRET);
        const user = await User.findById(decoded.id);
        if (!user || !user.isActive) {
          ApiResponse.error(res, 'User invalid or deactivated', 401);
          return;
        }

        (req as any).preAuthUser = {
          id: user._id.toString(),
          investigatorId: user.investigatorId,
          step: decoded.step || 'mfa_passkey',
        };
        (req as any).user = {
          id: user._id.toString(),
          investigatorId: user.investigatorId,
          role: user.role,
          email: user.email,
          name: user.name,
        };
        return next();
      } catch (err) {
        ApiResponse.error(res, 'Invalid or expired pre-authentication token.', 401);
        return;
      }
    }

    ApiResponse.error(res, 'Pre-authentication token is required.', 401);
  } catch (err) {
    next(err);
  }
};

/**
 * Authorize specific investigator roles.
 */
export const authorize =
  (...roles: UserRole[]) =>
  (req: Request, res: Response, next: NextFunction): void => {
    const user = (req as any).user as AuthenticatedUser | undefined;

    if (!user) {
      ApiResponse.error(res, 'Unauthorized', 401);
      return;
    }

    if (!roles.includes(user.role)) {
      ApiResponse.error(
        res,
        `Access denied. Role '${user.role}' lacks required permissions for this forensic operation.`,
        403
      );
      return;
    }

    next();
  };
