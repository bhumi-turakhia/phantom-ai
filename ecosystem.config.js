module.exports = {
  apps: [
    {
      name: 'phantom-api',
      cwd: './phantom-ai-api',
      script: 'dist/server.js',
      instances: 'max',
      exec_mode: 'cluster',
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      env: {
        NODE_ENV: 'production',
        PORT: 5001,
      },
    },
    {
      name: 'phantom-ml-service',
      cwd: './phantom-ai-ml',
      script: '.venv/bin/uvicorn',
      args: 'app.main:app --host 127.0.0.1 --port 8000 --workers 2',
      interpreter: 'none',
      autorestart: true,
      watch: false,
      max_memory_restart: '4G',
      env: {
        PHANTOM_ML_HOST: '127.0.0.1',
        PHANTOM_ML_PORT: '8000',
      },
    },
  ],
};

