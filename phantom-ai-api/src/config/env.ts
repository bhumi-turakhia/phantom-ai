import dotenv from 'dotenv';
import path from 'path';

dotenv.config({ path: path.resolve(process.cwd(), '.env') });

interface EnvConfig {
  NODE_ENV: string;
  PORT: number;
  MONGODB_URI: string;
  JWT_SECRET: string;
  JWT_EXPIRES_IN: string;
  CORS_ORIGIN: string;
  RATE_LIMIT_WINDOW_MS: number;
  RATE_LIMIT_MAX_REQUESTS: number;
  LOG_LEVEL: string;
  // WebAuthn / FIDO2 config
  RP_NAME: string;
  RP_ID: string;
  WEBAUTHN_ORIGIN: string;
  COOKIE_DOMAIN?: string;
  // Local Python Custom ML Service
  ML_SERVICE_URL: string;
  ML_SERVICE_TIMEOUT_MS: number;
  // Optional legacy keys (phased out in 3.0)
  GEMINI_API_KEY: string;
  GEMINI_MODEL: string;
  HUGGINGFACE_API_KEY: string;
  HUGGINGFACE_MODEL: string;
}

const requiredEnvVars: string[] = [
  'MONGODB_URI',
  'JWT_SECRET',
];

for (const key of requiredEnvVars) {
  if (!process.env[key]) {
    throw new Error(`Missing required environment variable: ${key}`);
  }
}

export const env: EnvConfig = {
  NODE_ENV: process.env.NODE_ENV || 'development',
  PORT: parseInt(process.env.PORT || '5001', 10),
  MONGODB_URI: process.env.MONGODB_URI as string,
  JWT_SECRET: process.env.JWT_SECRET as string,
  JWT_EXPIRES_IN: process.env.JWT_EXPIRES_IN || '7d',
  CORS_ORIGIN: process.env.CORS_ORIGIN || 'http://localhost:5173',
  RATE_LIMIT_WINDOW_MS: parseInt(process.env.RATE_LIMIT_WINDOW_MS || '900000', 10),
  RATE_LIMIT_MAX_REQUESTS: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS || '100', 10),
  LOG_LEVEL: process.env.LOG_LEVEL || 'info',
  // WebAuthn
  RP_NAME: process.env.RP_NAME || 'Phantom AI Forensic Platform',
  RP_ID: process.env.RP_ID || 'localhost',
  WEBAUTHN_ORIGIN: process.env.WEBAUTHN_ORIGIN || process.env.CORS_ORIGIN || 'http://localhost:5173',
  COOKIE_DOMAIN: process.env.COOKIE_DOMAIN,
  // Local Python Custom ML Service
  ML_SERVICE_URL: process.env.ML_SERVICE_URL || 'http://127.0.0.1:8000',
  ML_SERVICE_TIMEOUT_MS: parseInt(process.env.ML_SERVICE_TIMEOUT_MS || '10000', 10),
  // Phased out AI keys
  GEMINI_API_KEY: process.env.GEMINI_API_KEY || '',
  GEMINI_MODEL: process.env.GEMINI_MODEL || 'gemini-2.5-flash',
  HUGGINGFACE_API_KEY: process.env.HUGGINGFACE_API_KEY || '',
  HUGGINGFACE_MODEL:
    process.env.HUGGINGFACE_MODEL ||
    'black-forest-labs/FLUX.1-schnell',
};

export const isProduction = env.NODE_ENV === 'production';
export const isDevelopment = env.NODE_ENV === 'development';
export const isTest = env.NODE_ENV === 'test';
