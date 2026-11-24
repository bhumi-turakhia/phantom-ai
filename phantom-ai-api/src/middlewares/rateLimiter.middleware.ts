import rateLimit from 'express-rate-limit';
import { Request, Response } from 'express';
import { env } from '../config/env';
import { ApiResponse } from '../utils/apiResponse';

export const apiRateLimiter = rateLimit({
  windowMs: env.RATE_LIMIT_WINDOW_MS,
  max: env.NODE_ENV === 'development' || env.NODE_ENV === 'test' ? 5000 : env.RATE_LIMIT_MAX_REQUESTS,
  standardHeaders: true,
  legacyHeaders: false,
  message: 'Too many requests, please try again later.',
  handler: (_req: Request, res: Response) => {
    return ApiResponse.error(
      res,
      'Too many requests, please try again later.',
      429
    );
  },
});
