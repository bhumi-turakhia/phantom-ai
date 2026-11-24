import { Request, Response, NextFunction } from 'express';
import { ApiError, ApiResponse } from '../utils/apiResponse';
import { logger } from '../config/logger';
import { isProduction } from '../config/env';

export const notFoundHandler = (
  req: Request,
  res: Response,
  _next: NextFunction
): void => {
  ApiResponse.error(
    res,
    `Route not found: ${req.method} ${req.originalUrl}`,
    404
  );
};

export const errorHandler = (
  err: Error | ApiError,
  req: Request,
  res: Response,
  _next: NextFunction
): void => {
  let statusCode = 500;
  let message = 'Internal Server Error';

  if (err instanceof ApiError) {
    statusCode = err.statusCode;
    message = err.message;
  } else if (err.name === 'ValidationError') {
    statusCode = 400;
    message = err.message;
  } else if (err.name === 'CastError') {
    statusCode = 400;
    message = 'Invalid resource identifier';
  } else if (err.name === 'JsonWebTokenError') {
    statusCode = 401;
    message = 'Invalid token';
  } else if (err.name === 'TokenExpiredError') {
    statusCode = 401;
    message = 'Token expired';
  } else if ((err as { code?: number }).code === 11000) {
    statusCode = 409;
    message = 'Duplicate resource value';
  } else if ((err as any).type === 'entity.too.large' || (err as any).status === 413 || (err as any).statusCode === 413) {
    statusCode = 413;
    message = 'Payload too large. Request body exceeds maximum allowed size.';
  } else if (err.message) {
    message = err.message;
  }

  // Security Hardening: In production mode, never leak raw exception messages or database internals to clients on 500s
  if (isProduction && statusCode >= 500 && !(err instanceof ApiError)) {
    message = 'Internal Server Error';
  }

  if (statusCode >= 500) {
    logger.error(
      `${req.method} ${req.originalUrl} - ${err.message}${
        err.stack ? `\n${err.stack}` : ''
      }`
    );
  } else {
    logger.warn(`${req.method} ${req.originalUrl} - ${message}`);
  }

  ApiResponse.error(
    res,
    message,
    statusCode,
    !isProduction ? err.stack : undefined
  );
};