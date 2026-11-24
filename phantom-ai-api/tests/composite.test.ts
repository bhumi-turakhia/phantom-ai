import http from 'http';
import { Types } from 'mongoose';
import { MLClientService, mlClient } from '../src/services/mlClient.service';
import * as compositeService from '../src/services/composite.service';
import { Composite } from '../src/models/Composite.model';
import { WitnessDescription } from '../src/models/WitnessDescription.model';
import { Case } from '../src/models/Case.model';
import { SecurityAudit } from '../src/models/SecurityAudit.model';

describe('Phase 5.5 Backend Composite Integration Suite', () => {
  let mockServer: http.Server;
  let serverPort: number;
  let testMlClient: MLClientService;

  beforeAll((done) => {
    mockServer = http.createServer((req, res) => {
      res.setHeader('Content-Type', 'application/json');

      if (req.method === 'POST' && req.url === '/generate-composite') {
        let body = '';
        req.on('data', (chunk) => (body += chunk));
        req.on('end', () => {
          const parsed = JSON.parse(body);

          // Test contradiction rejection
          const hasContradiction = Object.values(parsed.forensic_profile || {}).some(
            (v: any) => v?.state === 'contradictory'
          );
          if (hasContradiction) {
            res.writeHead(409);
            res.end(
              JSON.stringify({
                detail: 'Cannot generate composite: Category contains unresolved contradictory testimony.',
              })
            );
            return;
          }

          // Test incompatibility rejection
          const hasIncompat = Object.values(parsed.forensic_profile || {}).some(
            (v: any) => v?.value === 'bald with bangs'
          );
          if (hasIncompat) {
            res.writeHead(422);
            res.end(
              JSON.stringify({
                detail: "IncompatibleAttributeError: 'bald' cannot co-occur with 'bangs'.",
              })
            );
            return;
          }

          // Return valid generation response with 1x1 valid PNG base64
          // 1x1 8-bit PNG
          const validPngB64 =
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

          res.writeHead(200);
          res.end(
            JSON.stringify({
              status: 'success',
              generator: {
                model_name: 'ConditionalWGANGPGenerator',
                architecture: 'Conditional WGAN-GP with Projection Conditioning',
                phase: 'Phase 5.4 / 5.5',
                checkpoint_path: 'phantom-ai-ml/checkpoints/cgan_generator_phase54.pt',
                checkpoint_sha256: 'a1714389d0d1eddad10fae74a736ca41de5e230eada095e6868e5daeeb2a1964',
                best_epoch: 4,
                parameter_count: 4102336,
                device: 'mps',
              },
              variations: [
                {
                  variation_index: 1,
                  seed: 42,
                  image_base64: validPngB64,
                  data_url: `data:image/png;base64,${validPngB64}`,
                  width: 64,
                  height: 64,
                },
              ],
              conditioned_attributes: ['hair_black (+1.0)', 'facial_goatee (+1.0)'],
              unconditioned_attributes: ['glasses', 'hair_bald'],
              warnings: [],
              latency_ms: 125.4,
              timestamp: new Date().toISOString(),
            })
          );
        });
      } else if (req.method === 'GET' && req.url === '/latent-directions') {
        res.writeHead(200);
        res.end(
          JSON.stringify({
            status: 'success',
            total_dimensions: 20,
            validated_count: 5,
            rejected_count: 15,
            validated_attributes: ['hair_blond', 'hair_gray', 'hair_bangs', 'facial_goatee', 'glasses'],
            detailed_metrics: {},
            unsupported_features: ['smile_intensity', 'continuous_hair_length'],
            timestamp: new Date().toISOString(),
          })
        );
      } else if (req.method === 'POST' && req.url === '/edit-composite') {
        let body = '';
        req.on('data', (chunk) => (body += chunk));
        req.on('end', () => {
          const parsed = JSON.parse(body);
          const traversals = parsed.latent_traversals || [];
          for (const item of traversals) {
            if (item.attribute === 'smile_intensity') {
              res.writeHead(400);
              res.end(
                JSON.stringify({
                  detail: "Attribute 'smile_intensity' is not an empirically validated latent direction.",
                })
              );
              return;
            }
          }

          const validPngB64 =
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

          res.writeHead(200);
          res.end(
            JSON.stringify({
              status: 'success',
              generator: {
                model_name: 'ConditionalWGANGPGenerator',
                architecture: 'Conditional WGAN-GP with Projection Conditioning',
                phase: 'Phase 5.7 Semantic Editing',
                checkpoint_path: 'phantom-ai-ml/checkpoints/cgan_generator_phase54.pt',
                checkpoint_sha256: 'a1714389d0d1eddad10fae74a736ca41de5e230eada095e6868e5daeeb2a1964',
                best_epoch: 4,
                parameter_count: 4102336,
                device: 'mps',
              },
              edited_composite: {
                variation_index: 1,
                seed: parsed.seed || 42,
                image_base64: validPngB64,
                data_url: `data:image/png;base64,${validPngB64}`,
                width: 64,
                height: 64,
              },
              original_composite: {
                variation_index: 0,
                seed: parsed.seed || 42,
                image_base64: validPngB64,
                data_url: `data:image/png;base64,${validPngB64}`,
                width: 64,
                height: 64,
              },
              applied_continuous_adjustments: parsed.continuous_gcit_adjustments || {},
              applied_latent_traversals: parsed.latent_traversals || [],
              conditioned_attributes: ['nose_big (+0.75)', 'hair_black (+1.00)'],
              warnings: [],
              latency_ms: 18.2,
              timestamp: new Date().toISOString(),
            })
          );
        });
      } else {
        res.writeHead(404);
        res.end(JSON.stringify({ detail: 'Not found' }));
      }
    });

    mockServer.listen(0, '127.0.0.1', () => {
      serverPort = (mockServer.address() as any).port;
      testMlClient = new MLClientService(`http://127.0.0.1:${serverPort}`, 5000);
      done();
    });
  });

  afterAll((done) => {
    mockServer.close(done);
  });

  describe('MLClientService.generateComposite', () => {
    it('successfully calls FastAPI generation endpoint and returns variations', async () => {
      const response = await testMlClient.generateComposite({
        forensic_profile: {
          hair: { value: 'black', state: 'explicit' },
          facial_hair: { value: 'goatee', state: 'explicit' },
        },
        seed: 42,
        num_variations: 1,
      });

      expect(response.status).toBe('success');
      expect(response.generator.checkpoint_sha256).toBe(
        'a1714389d0d1eddad10fae74a736ca41de5e230eada095e6868e5daeeb2a1964'
      );
      expect(response.variations.length).toBe(1);
      expect(response.variations[0].seed).toBe(42);
      expect(response.variations[0].data_url).toContain('data:image/png;base64,');
    });

    it('rejects profiles containing unresolved contradictory testimony with 409', async () => {
      await expect(
        testMlClient.generateComposite({
          forensic_profile: {
            hair: { value: 'black', state: 'contradictory' },
          },
        })
      ).rejects.toThrow(/contradictory testimony/i);
    });

    it('rejects physically incompatible profiles with 422', async () => {
      await expect(
        testMlClient.generateComposite({
          forensic_profile: {
            hair: { value: 'bald with bangs', state: 'explicit' },
          },
        })
      ).rejects.toThrow(/IncompatibleAttributeError/i);
    });

    it('handles offline ML service with 503 ApiError', async () => {
      const deadClient = new MLClientService('http://127.0.0.1:59999', 500);
      await expect(
        deadClient.generateComposite({
          forensic_profile: { hair: { value: 'black', state: 'explicit' } },
        })
      ).rejects.toThrow(/offline or unreachable/i);
    });
  });

  describe('Composite Service Approval & RBAC Enforcement', () => {
    const mockCaseId = new Types.ObjectId().toString();
    const mockWitnessId = new Types.ObjectId().toString();
    const mockUserId = new Types.ObjectId().toString();

    beforeEach(() => {
      jest.clearAllMocks();
    });

    it('strictly forbids composite generation if witness description is unapproved', async () => {
      jest.spyOn(Case, 'findOne').mockResolvedValue({
        _id: mockCaseId,
        createdBy: mockUserId,
        assignedTo: [],
        isDeleted: false,
      } as any);

      jest.spyOn(WitnessDescription, 'findOne').mockResolvedValue({
        _id: mockWitnessId,
        case: mockCaseId,
        reviewStatus: 'in_review', // NOT approved
        extractedAttributes: [
          { category: 'hair', value: 'black', evidenceState: 'explicit', sourceText: 'black hair' },
        ],
      } as any);

      jest.spyOn(SecurityAudit, 'create').mockResolvedValue({} as any);

      await expect(
        compositeService.generateCompositeForCase(mockCaseId, mockUserId, 'investigator', {
          witnessDescriptionId: mockWitnessId,
        })
      ).rejects.toThrow(/review status 'in_review'/i);

      expect(SecurityAudit.create).toHaveBeenCalledWith(
        expect.objectContaining({
          event: 'COMPOSITE_GENERATION_FAILURE',
          status: 'failure',
        })
      );
    });

    it('strictly forbids generation if investigator lacks case access', async () => {
      const otherUserId = new Types.ObjectId().toString();
      jest.spyOn(Case, 'findOne').mockResolvedValue({
        _id: mockCaseId,
        createdBy: otherUserId, // Not mockUserId
        assignedTo: [],
        isDeleted: false,
      } as any);

      await expect(
        compositeService.generateCompositeForCase(mockCaseId, mockUserId, 'investigator', {
          witnessDescriptionId: mockWitnessId,
        })
      ).rejects.toThrow(/Forbidden/i);
    });

    it('prevents path traversal attempts when accessing images', async () => {
      jest.spyOn(Case, 'findOne').mockResolvedValue({
        _id: mockCaseId,
        createdBy: mockUserId,
        assignedTo: [],
        isDeleted: false,
      } as any);

      jest.spyOn(Composite, 'findOne').mockResolvedValue({
        _id: new Types.ObjectId().toString(),
        case: mockCaseId,
        variations: [
          {
            variationIndex: 1,
            storageKey: '../../../../etc/passwd',
            mimeType: 'image/png',
          },
        ],
      } as any);

      await expect(
        compositeService.getCompositeImageFile(
          mockCaseId,
          new Types.ObjectId().toString(),
          1,
          mockUserId,
          'investigator'
        )
      ).rejects.toThrow(/not found on server storage/i);
    });
  });

  describe('Phase 5.7 Semantic Editing Suite', () => {
    const mockCaseId = new Types.ObjectId().toString();
    const mockSourceCompositeId = new Types.ObjectId().toString();
    const mockWitnessId = new Types.ObjectId().toString();
    const mockUserId = new Types.ObjectId().toString();

    it('MLClientService.getLatentDirections retrieves validated attributes', async () => {
      const res = await testMlClient.getLatentDirections();
      expect(res.status).toBe('success');
      expect(res.validated_count).toBe(5);
      expect(res.validated_attributes).toContain('glasses');
      expect(res.validated_attributes).toContain('hair_blond');
    });

    it('MLClientService.editComposite executes semantic traversal cleanly', async () => {
      const res = await testMlClient.editComposite({
        forensic_profile: { hair: { value: 'black', state: 'explicit' } },
        seed: 42,
        continuous_gcit_adjustments: { nose_big: 0.75 },
        latent_traversals: [{ attribute: 'glasses', alpha: 1.0 }],
      });
      expect(res.status).toBe('success');
      expect(res.edited_composite).toBeDefined();
      expect([64, 128]).toContain(res.edited_composite.width);
      expect(res.applied_continuous_adjustments.nose_big).toBe(0.75);
      expect(res.applied_latent_traversals[0].attribute).toBe('glasses');
    });

    it('MLClientService.editComposite rejects unvalidated directions (e.g. smile)', async () => {
      await expect(
        testMlClient.editComposite({
          forensic_profile: { hair: { value: 'black', state: 'explicit' } },
          seed: 42,
          latent_traversals: [{ attribute: 'smile_intensity', alpha: 1.0 }],
        })
      ).rejects.toThrow(/not an empirically validated latent direction/i);
    });

    it('compositeService.editCompositeForCase creates next sequential version with parent link', async () => {
      jest.spyOn(Case, 'findOne').mockResolvedValue({
        _id: mockCaseId,
        createdBy: mockUserId,
        assignedTo: [mockUserId],
        isDeleted: false,
      } as any);

      jest.spyOn(Composite, 'findOne').mockImplementation((query: any) => {
        if (query && query._id === mockSourceCompositeId) {
          return Promise.resolve({
            _id: mockSourceCompositeId,
            case: mockCaseId,
            witnessDescription: mockWitnessId,
            version: 1,
            approvedEvidenceSnapshot: { hair: { value: 'black', state: 'explicit' } },
          } as any);
        }
        return {
          sort: () => Promise.resolve({ version: 1 }),
        } as any;
      });

      jest.spyOn(WitnessDescription, 'findOne').mockResolvedValue({
        _id: mockWitnessId,
        reviewStatus: 'approved',
      } as any);

      const mockCreatedComposite = {
        _id: new Types.ObjectId(),
        case: mockCaseId,
        version: 2,
        parentComposite: mockSourceCompositeId,
        editingType: 'hybrid_traversal',
        variations: [
          { variationIndex: 1, isSelected: true, imageUrl: '' },
          { variationIndex: 2, isSelected: false, imageUrl: '' },
        ],
        save: jest.fn().mockResolvedValue(true),
      };
      jest.spyOn(Composite, 'create').mockResolvedValue(mockCreatedComposite as any);
      jest.spyOn(SecurityAudit, 'create').mockResolvedValue({} as any);

      jest.spyOn(mlClient, 'editComposite').mockResolvedValue({
        status: 'success',
        generator: {
          model_name: 'ConditionalWGANGPGenerator',
          architecture: 'Conditional WGAN-GP with Projection Conditioning',
          phase: 'Phase 5.7 Semantic Editing',
          checkpoint_path: 'phantom-ai-ml/checkpoints/cgan_generator_phase54.pt',
          checkpoint_sha256: 'a1714389d0d1eddad10fae74a736ca41de5e230eada095e6868e5daeeb2a1964',
          best_epoch: 4,
          parameter_count: 4102336,
          device: 'mps',
        },
        edited_composite: {
          variation_index: 1,
          seed: 42,
          image_base64:
            'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAADG0lEQVR4AQEQA+/8gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCpHiBDrRvafAAAAAElFTkSuQmCC',
          data_url:
            'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAADG0lEQVR4AQEQA+/8gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCpHiBDrRvafAAAAAElFTkSuQmCC',
          width: 64,
          height: 64,
        },
        applied_continuous_adjustments: { nose_big: 0.5 },
        applied_latent_traversals: [{ attribute: 'glasses', alpha: 1.0, unit_norm: 1.0 }],
        conditioned_attributes: ['nose_big (+0.50)'],
        warnings: [],
        latency_ms: 15.0,
        timestamp: new Date().toISOString(),
      });

      // Test editing
      const res = await compositeService.editCompositeForCase(
        mockCaseId,
        mockSourceCompositeId,
        mockUserId,
        'investigator',
        {
          seed: 42,
          continuousAdjustments: { nose_big: 0.5 },
          latentTraversals: [{ attribute: 'glasses', alpha: 1.0 }],
        }
      );

      expect(res.composite).toBeDefined();
      expect(res.composite.version).toBe(2);
      expect(res.composite.parentComposite).toBe(mockSourceCompositeId);
    });

    it('compositeService.editCompositeForCase rejects when witness is unapproved', async () => {
      jest.spyOn(Case, 'findOne').mockResolvedValue({
        _id: mockCaseId,
        createdBy: mockUserId,
        assignedTo: [mockUserId],
        isDeleted: false,
      } as any);

      jest.spyOn(Composite, 'findOne').mockResolvedValue({
        _id: mockSourceCompositeId,
        case: mockCaseId,
        witnessDescription: mockWitnessId,
        version: 1,
        approvedEvidenceSnapshot: {},
      } as any);

      jest.spyOn(WitnessDescription, 'findOne').mockResolvedValue({
        _id: mockWitnessId,
        reviewStatus: 'in_review', // UNAPPROVED
      } as any);

      await expect(
        compositeService.editCompositeForCase(
          mockCaseId,
          mockSourceCompositeId,
          mockUserId,
          'investigator',
          { seed: 42 }
        )
      ).rejects.toThrow(/Only approved forensic evidence profiles can be edited/i);
    });
  });
});
