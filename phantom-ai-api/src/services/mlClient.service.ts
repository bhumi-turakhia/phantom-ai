import axios, { AxiosInstance, AxiosError } from 'axios';
import { env } from '../config/env';
import { ApiError } from '../utils/apiResponse';
import { logger } from '../config/logger';

export interface MLDeviceInfo {
  device_type: string;
  device_name: string;
  mps_built: boolean;
  mps_available: boolean;
  cuda_available: boolean;
  torch_version: string;
  python_version: string;
}

export interface MLHealthResponse {
  status: string;
  service: string;
  version: string;
  device: MLDeviceInfo;
  timestamp: string;
}

export interface MLModelStatusItem {
  model_name: string;
  architecture: string;
  is_trained: boolean;
  checkpoint_exists: boolean;
  checkpoint_path: string;
  parameter_count: number;
  input_shape: string;
  output_shape: string;
  notes: string;
}

export interface MLModelStatusResponse {
  service: string;
  device: MLDeviceInfo;
  models: Record<string, MLModelStatusItem>;
  all_models_trained: boolean;
}

export interface MLAttributeEvidence {
  category: string;
  state: 'explicit' | 'uncertain' | 'unknown' | 'contradictory' | 'negated';
  value: string | null;
  confidence: number;
  source_spans: string[];
  uncertainty_cues: string[];
  notes?: string;
  start_char?: number | null;
  end_char?: number | null;
  extracted_by?: string | null;
}

export interface MLModelMetadataInfo {
  name: string;
  architecture: string;
  phase: string;
  checkpoint_path: string;
  checkpoint_sha256: string;
  device: string;
  mps_built?: boolean;
}

export interface MLAnalyzeDescriptionResponse {
  status: string;
  is_model_trained: boolean;
  attributes: Record<string, MLAttributeEvidence>;
  contradictions: string[];
  uncertainties: string[];
  overall_confidence: number;
  audit_notes: string;
  model?: MLModelMetadataInfo;
}

export interface MLValidateAttributesResponse {
  valid: boolean;
  validation_errors: string[];
  contradictions_detected: string[];
  verified_states: Record<string, string>;
  summary: string;
}

export interface MLEvaluateResponse {
  model_name: string;
  status: string;
  is_trained: boolean;
  metrics?: Record<string, number> | null;
  evaluation_details: string;
}

export interface MLCompositeVariationItem {
  variation_index: number;
  seed: number;
  image_base64: string;
  data_url: string;
  width: number;
  height: number;
}

export interface MLGenerateCompositeResponse {
  status: string;
  generator: {
    model_name: string;
    architecture: string;
    phase: string;
    checkpoint_path: string;
    checkpoint_sha256: string;
    best_epoch: number;
    parameter_count: number;
    device: string;
  };
  variations: MLCompositeVariationItem[];
  conditioned_attributes: string[];
  unconditioned_attributes: string[];
  warnings: string[];
  latency_ms: number;
  timestamp: string;
}

export interface MLLatentDirectionsResponse {
  status: string;
  total_dimensions: number;
  validated_count: number;
  rejected_count: number;
  validated_attributes: string[];
  detailed_metrics: Record<string, any>;
  unsupported_features: string[];
  timestamp: string;
}

export interface MLEditCompositeResponse {
  status: string;
  generator: MLGenerateCompositeResponse['generator'];
  edited_composite: MLCompositeVariationItem;
  original_composite?: MLCompositeVariationItem | null;
  applied_continuous_adjustments: Record<string, number>;
  applied_latent_traversals: Array<{ attribute: string; alpha: number; unit_norm: number }>;
  conditioned_attributes: string[];
  warnings: string[];
  latency_ms: number;
  timestamp: string;
}

export class MLClientService {
  private client: AxiosInstance;
  public baseURL: string;

  constructor(baseURL: string = env.ML_SERVICE_URL, timeout: number = env.ML_SERVICE_TIMEOUT_MS) {
    this.baseURL = baseURL;
    this.client = axios.create({
      baseURL,
      timeout,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
    });
  }

  private handleError(error: unknown, action: string): never {
    if (axios.isAxiosError(error)) {
      const axiosErr = error as AxiosError<any>;
      const isConnRefused =
        axiosErr.code === 'ECONNREFUSED' ||
        axiosErr.code === 'ENOTFOUND' ||
        axiosErr.code === 'EPERM' ||
        (axiosErr.cause as any)?.code === 'ECONNREFUSED' ||
        (axiosErr.cause as any)?.code === 'EPERM' ||
        axiosErr.message.includes('ECONNREFUSED') ||
        axiosErr.message.includes('connect ECONNREFUSED') ||
        axiosErr.message.includes('connect EPERM') ||
        axiosErr.message.includes('ENOTFOUND');

      if (isConnRefused) {
        logger.error(`[ML-SERVICE] Connection refused during ${action} at ${this.baseURL}`);
        throw new ApiError(
          `Local Python ML service is currently offline or unreachable at ${this.baseURL}. Please start the FastAPI service.`,
          503
        );
      }

      if (axiosErr.code === 'ECONNABORTED' || axiosErr.message.includes('timeout')) {
        logger.error(`[ML-SERVICE] Timeout during ${action}`);
        throw new ApiError(`Python ML service request timed out during ${action}.`, 504);
      }

      const status = axiosErr.response?.status || 500;
      const detail = axiosErr.response?.data?.detail || axiosErr.message;
      logger.error(`[ML-SERVICE] Error during ${action} (${status}): ${JSON.stringify(detail)}`);

      if (status === 422) {
        throw new ApiError(`ML service validation error: ${JSON.stringify(detail)}`, 400);
      }

      throw new ApiError(`ML service error: ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`, status);
    }

    const message = error instanceof Error ? error.message : 'Unknown error';
    logger.error(`[ML-SERVICE] Unexpected error during ${action}: ${message}`);
    throw new ApiError(`Internal error communicating with Python ML service: ${message}`, 500);
  }

  /**
   * Health and hardware acceleration status check.
   */
  async checkHealth(): Promise<MLHealthResponse> {
    try {
      const response = await this.client.get<MLHealthResponse>('/health');
      return response.data;
    } catch (error) {
      this.handleError(error, 'health check');
    }
  }

  /**
   * Inspects model architecture status and checkpoint existence.
   */
  async getModelStatus(): Promise<MLModelStatusResponse> {
    try {
      const response = await this.client.get<MLModelStatusResponse>('/model-status');
      return response.data;
    } catch (error) {
      this.handleError(error, 'model status retrieval');
    }
  }

  /**
   * Analyzes witness testimony to extract facial attributes while preserving uncertainty.
   */
  async analyzeDescription(
    witnessText: string,
    confidenceThreshold: number = 0.5,
    caseId?: string
  ): Promise<MLAnalyzeDescriptionResponse> {
    if (!witnessText || !witnessText.trim()) {
      throw new ApiError('Witness description text cannot be empty.', 400);
    }

    try {
      const response = await this.client.post<MLAnalyzeDescriptionResponse>('/analyze-description', {
        witness_text: witnessText.trim(),
        confidence_threshold: confidenceThreshold,
        case_id: caseId,
      });
      return response.data;
    } catch (error) {
      this.handleError(error, 'description analysis');
    }
  }

  /**
   * Validates proposed facial attributes against forensic constraints and contradiction rules.
   */
  async validateAttributes(attributes: Record<string, any>): Promise<MLValidateAttributesResponse> {
    if (!attributes || typeof attributes !== 'object') {
      throw new ApiError('Attributes must be a valid object dictionary.', 400);
    }

    try {
      const response = await this.client.post<MLValidateAttributesResponse>('/validate-attributes', {
        attributes,
      });
      return response.data;
    } catch (error) {
      this.handleError(error, 'attribute validation');
    }
  }

  /**
   * Evaluates model checkpoints without returning simulated metrics.
   */
  async evaluateModel(modelName: string, datasetName?: string): Promise<MLEvaluateResponse> {
    if (!modelName || !modelName.trim()) {
      throw new ApiError('Model name is required for evaluation.', 400);
    }

    try {
      const response = await this.client.post<MLEvaluateResponse>('/evaluate', {
        model_name: modelName.trim(),
        dataset_name: datasetName,
      });
      return response.data;
    } catch (error) {
      this.handleError(error, 'model evaluation');
    }
  }

  /**
   * Generates condition-consistent forensic composites using the frozen Phase 5.4 Conditional WGAN-GP.
   */
  async generateComposite(payload: {
    forensic_profile: Record<string, any>;
    seed?: number;
    num_variations?: number;
    case_id?: string;
    witness_id?: string;
  }): Promise<MLGenerateCompositeResponse> {
    if (!payload.forensic_profile || typeof payload.forensic_profile !== 'object') {
      throw new ApiError('forensic_profile must be a valid object.', 400);
    }

    try {
      const response = await this.client.post<MLGenerateCompositeResponse>('/generate-composite', payload);
      return response.data;
    } catch (error) {
      this.handleError(error, 'composite generation');
    }
  }

  /**
   * Retrieves Phase 5.7 latent direction evaluation status and supported dimensions.
   */
  async getLatentDirections(): Promise<MLLatentDirectionsResponse> {
    try {
      const response = await this.client.get<MLLatentDirectionsResponse>('/latent-directions');
      return response.data;
    } catch (error) {
      this.handleError(error, 'fetching latent directions');
    }
  }

  /**
   * Executes controlled semantic editing via continuous GCIT conditioning traversal
   * and latent-space hyperplane traversal in Z.
   */
  async editComposite(payload: {
    forensic_profile: Record<string, any>;
    seed: number;
    continuous_gcit_adjustments?: Record<string, number>;
    latent_traversals?: Array<{ attribute: string; alpha: number }>;
    include_original?: boolean;
    case_id?: string;
    witness_id?: string;
  }): Promise<MLEditCompositeResponse> {
    if (!payload.forensic_profile || typeof payload.forensic_profile !== 'object') {
      throw new ApiError('forensic_profile must be a valid object.', 400);
    }
    if (typeof payload.seed !== 'number') {
      throw new ApiError('seed must be a valid integer.', 400);
    }

    try {
      const response = await this.client.post<MLEditCompositeResponse>('/edit-composite', payload);
      return response.data;
    } catch (error) {
      this.handleError(error, 'semantic editing');
    }
  }
}

export const mlClient = new MLClientService();
