import { Response } from 'express';

interface SuccessPayload<T> {
  success: true;
  statusCode: number;
  message: string;
  data: T | null;
  timestamp: string;
}

interface ErrorPayload {
  success: false;
  statusCode: number;
  message: string;
  stack?: string;
  timestamp: string;
}

export class ApiError extends Error {
  public readonly statusCode: number;
  public readonly isOperational: boolean;

  constructor(
    message: string,
    statusCode: number = 500,
    isOperational: boolean = true
  ) {
    super(message);
    this.statusCode = statusCode;
    this.isOperational = isOperational;
    Object.setPrototypeOf(this, new.target.prototype);
    Error.captureStackTrace(this, this.constructor);
  }
}

export class ApiResponse {
  static success<T>(
    res: Response,
    message: string,
    data: T | null = null,
    statusCode: number = 200
  ): Response<SuccessPayload<T>> {
    const payload: SuccessPayload<T> = {
      success: true,
      statusCode,
      message,
      data,
      timestamp: new Date().toISOString(),
    };

    return res.status(statusCode).json(payload);
  }

  static error(
    res: Response,
    message: string,
    statusCode: number = 500,
    stack?: string
  ): Response<ErrorPayload> {
    const payload: ErrorPayload = {
      success: false,
      statusCode,
      message,
      timestamp: new Date().toISOString(),
      ...(stack && { stack }),
    };

    return res.status(statusCode).json(payload);
  }
}