import express, { Application, Request, Response } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import compression from 'compression';
import cookieParser from 'cookie-parser';
import {
  csrfProtection,
  getTrustedOrigins,
  isOriginTrusted,
} from './middlewares/csrf.middleware';
import { isProduction } from './config/env';
import { stream } from './config/logger';
import { apiRateLimiter } from './middlewares/rateLimiter.middleware';
import { errorHandler, notFoundHandler } from './middlewares/error.middleware';
import routes from './routes';

const app: Application = express();

app.use(helmet());

const trustedOrigins = getTrustedOrigins();

// Secure CORS configuration: Never accept wildcard origins with credentials
app.use(
  cors({
    origin: (requestOrigin, callback) => {
      // Allow non-browser requests or internal server-to-server calls without an Origin header
      if (!requestOrigin) {
        return callback(null, true);
      }
      if (isOriginTrusted(requestOrigin, trustedOrigins)) {
        return callback(null, true);
      }
      // Untrusted origins rejected: omit Access-Control-Allow-Origin
      return callback(null, false);
    },
    credentials: true,
  })
);

app.use(compression());
app.use(cookieParser());
app.use(express.json({ limit: '10kb' }));
app.use(express.urlencoded({ extended: true, limit: '10kb' }));
app.use(csrfProtection);

if (!isProduction) {
  app.use(morgan('dev'));
} else {
  app.use(morgan('combined', { stream }));
}

app.use('/api', apiRateLimiter);

app.get('/health', (_req: Request, res: Response) => {
  res.status(200).json({
    success: true,
    message: 'Phantom AI Backend is running',
    timestamp: new Date().toISOString(),
  });
});

app.use('/api/v1', routes);

app.use(notFoundHandler);

app.use(errorHandler);

export default app;
