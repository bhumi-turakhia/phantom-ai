import http from 'http';
import { MLClientService } from '../src/services/mlClient.service';
import { ApiError } from '../src/utils/apiResponse';

describe('Node Backend ML Client Service Integration Suite', () => {
  let mockServer: http.Server;
  let serverPort: number;
  let mlClient: MLClientService;

  beforeAll((done) => {
    // Spin up local mock server mimicking FastAPI responses
    mockServer = http.createServer((req, res) => {
      res.setHeader('Content-Type', 'application/json');

      if (req.method === 'GET' && req.url === '/health') {
        res.writeHead(200);
        res.end(
          JSON.stringify({
            status: 'healthy',
            service: 'Phantom AI Custom ML Service',
            version: '3.0.0-phase2-foundation',
            device: {
              device_type: 'mps',
              device_name: 'Apple Silicon MPS (Metal Performance Shaders)',
              mps_built: true,
              mps_available: true,
              cuda_available: false,
              torch_version: '2.15.0',
              python_version: '3.13.7',
            },
            timestamp: new Date().toISOString(),
          })
        );
      } else if (req.method === 'GET' && req.url === '/model-status') {
        res.writeHead(200);
        res.end(
          JSON.stringify({
            service: 'Phantom AI Custom ML Service',
            device: { device_type: 'mps' },
            models: {
              nlp_bilstm: {
                model_name: 'BiLSTMAttributeLabeler',
                architecture: 'Custom Bidirectional LSTM + Linear CRF Emission Layer',
                is_trained: false,
                checkpoint_exists: false,
                checkpoint_path: 'checkpoints/nlp_bilstm.pt',
                parameter_count: 500000,
                input_shape: '(batch_size, seq_len)',
                output_shape: '(batch_size, seq_len, 15)',
                notes: 'Untrained in Phase 2',
              },
            },
            all_models_trained: false,
          })
        );
      } else if (req.method === 'POST' && req.url === '/analyze-description') {
        let body = '';
        req.on('data', (chunk) => (body += chunk));
        req.on('end', () => {
          res.writeHead(200);
          res.end(
            JSON.stringify({
              status: 'model_not_trained',
              is_model_trained: false,
              attributes: {
                hair: {
                  category: 'hair',
                  state: 'explicit',
                  value: 'short black hair',
                  confidence: 0.85,
                  source_spans: ['short black hair'],
                  uncertainty_cues: [],
                },
                eyes: {
                  category: 'eyes',
                  state: 'unknown',
                  value: null,
                  confidence: 0.0,
                  source_spans: [],
                  uncertainty_cues: [],
                },
              },
              contradictions: [],
              uncertainties: [],
              overall_confidence: 0.85,
              audit_notes: 'Tested without fabricated weights',
            })
          );
        });
      } else if (req.method === 'POST' && req.url === '/validate-attributes') {
        res.writeHead(200);
        res.end(
          JSON.stringify({
            valid: true,
            validation_errors: [],
            contradictions_detected: [],
            verified_states: { hair: 'explicit' },
            summary: 'All attributes validated successfully.',
          })
        );
      } else if (req.method === 'POST' && req.url === '/evaluate') {
        res.writeHead(200);
        res.end(
          JSON.stringify({
            model_name: 'nlp',
            status: 'model_not_trained',
            is_trained: false,
            metrics: null,
            evaluation_details: 'No checkpoint found.',
          })
        );
      } else {
        res.writeHead(404);
        res.end(JSON.stringify({ detail: 'Not Found' }));
      }
    });

    mockServer.listen(0, '127.0.0.1', () => {
      const addr = mockServer.address() as any;
      serverPort = addr.port;
      mlClient = new MLClientService(`http://127.0.0.1:${serverPort}`, 3000);
      done();
    });
  });

  afterAll((done) => {
    mockServer.close(done);
  });

  test('1. Node ML Client verifies health and device detection from Python service', async () => {
    const health = await mlClient.checkHealth();
    expect(health.status).toBe('healthy');
    expect(health.device.device_type).toBe('mps');
    expect(health.device.mps_available).toBe(true);
  });

  test('2. Node ML Client checks model status and detects untrained models truthfully', async () => {
    const status = await mlClient.getModelStatus();
    expect(status.all_models_trained).toBe(false);
    expect(status.models.nlp_bilstm.is_trained).toBe(false);
    expect(status.models.nlp_bilstm.parameter_count).toBeGreaterThan(0);
  });

  test('3. Node ML Client sends witness description and receives structured forensic attributes', async () => {
    const result = await mlClient.analyzeDescription('Suspect had short black hair.');
    expect(result.status).toBe('model_not_trained');
    expect(result.is_model_trained).toBe(false);
    expect(result.attributes.hair.value).toBe('short black hair');
    expect(result.attributes.eyes.state).toBe('unknown');
  });

  test('4. Node ML Client validates input constraints before sending requests', async () => {
    await expect(mlClient.analyzeDescription('')).rejects.toThrow(ApiError);
    await expect(mlClient.analyzeDescription('   ')).rejects.toThrow(/cannot be empty/i);
    await expect(mlClient.evaluateModel('')).rejects.toThrow(/required/i);
  });

  test('5. Node ML Client validates attributes payload', async () => {
    const valRes = await mlClient.validateAttributes({ hair: 'short black' });
    expect(valRes.valid).toBe(true);
    expect(valRes.validation_errors.length).toBe(0);
  });

  test('6. Node ML Client handles evaluate endpoint and reports untrained status', async () => {
    const evalRes = await mlClient.evaluateModel('nlp');
    expect(evalRes.status).toBe('model_not_trained');
    expect(evalRes.is_trained).toBe(false);
    expect(evalRes.metrics).toBeNull();
  });

  test('7. Node ML Client handles connection failure gracefully when Python service is offline', async () => {
    // Target an unused port to simulate service downtime
    const offlineClient = new MLClientService('http://127.0.0.1:59999', 1000);
    try {
      await offlineClient.checkHealth();
      fail('Expected call to offline ML service to throw ApiError');
    } catch (err: any) {
      expect(err).toBeInstanceOf(ApiError);
      expect(err.statusCode).toBe(503);
      expect(err.message).toMatch(/offline or unreachable/i);
    }
  });
});
