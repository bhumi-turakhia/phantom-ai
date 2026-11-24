import app from './app';
import { env } from './config/env';
import { connectDB, disconnectDB } from './config/db';
import { runAccountMigrations } from './services/migration.service';
import { logger } from './config/logger';

let server: ReturnType<typeof app.listen>;

const startServer = async (): Promise<void> => {
  try {
    await connectDB();
    await runAccountMigrations();

    server = app.listen(env.PORT, () => {
      logger.info(
        `Phantom AI Backend running in ${env.NODE_ENV} mode on port ${env.PORT}`
      );
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    logger.error(`Failed to start server: ${message}`);
    process.exit(1);
  }
};

const gracefulShutdown = async (signal: string): Promise<void> => {
  logger.info(`${signal} received. Shutting down gracefully...`);

  if (server) {
    server.close(async () => {
      logger.info('HTTP server closed');
      await disconnectDB();
      process.exit(0);
    });
  } else {
    await disconnectDB();
    process.exit(0);
  }
};

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

process.on('unhandledRejection', (reason: Error) => {
  logger.error(`Unhandled Rejection: ${reason.message}`);
  process.exit(1);
});

process.on('uncaughtException', (error: Error) => {
  logger.error(`Uncaught Exception: ${error.message}`);
  process.exit(1);
});

startServer();