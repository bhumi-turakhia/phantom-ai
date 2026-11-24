import { Types } from 'mongoose';
import { getDashboardOverview } from '../src/services/dashboard.service';
import { Case } from '../src/models/Case.model';
import { Composite } from '../src/models/Composite.model';
import { WitnessDescription } from '../src/models/WitnessDescription.model';
import { Report } from '../src/models/Report.model';
import { CaseComment } from '../src/models/CaseComment.model';
import { User } from '../src/models/User.model';
import { SecurityAudit } from '../src/models/SecurityAudit.model';
import { GenerationJob } from '../src/models/GenerationJob.model';
import { mlClient } from '../src/services/mlClient.service';

jest.mock('../src/services/caseCollaboration.service', () => ({
  getCaseTimeline: jest.fn().mockResolvedValue({ events: [] }),
}));

function createChainableQuery<T>(data: T): any {
  const query: any = {
    select: jest.fn().mockReturnThis(),
    sort: jest.fn().mockReturnThis(),
    populate: jest.fn().mockReturnThis(),
    lean: jest.fn().mockResolvedValue(data),
    then: (resolve: any) => Promise.resolve(data).then(resolve),
  };
  return query;
}

describe('Dashboard Data Integrity Service Suite', () => {
  const mockUserId = new Types.ObjectId().toString();

  beforeEach(() => {
    jest.clearAllMocks();

    // Model countDocuments spies
    jest.spyOn(Case, 'countDocuments').mockResolvedValue(10 as any);
    jest.spyOn(Composite, 'countDocuments').mockResolvedValue(5 as any);
    jest.spyOn(WitnessDescription, 'countDocuments').mockResolvedValue(8 as any);
    jest.spyOn(Report, 'countDocuments').mockResolvedValue(2 as any);
    jest.spyOn(CaseComment, 'countDocuments').mockResolvedValue(4 as any);
    jest.spyOn(User, 'countDocuments').mockResolvedValue(3 as any);
    jest.spyOn(SecurityAudit, 'countDocuments').mockResolvedValue(25 as any);

    // Case.find spy
    jest.spyOn(Case, 'find').mockReturnValue(
      createChainableQuery([
        {
          _id: new Types.ObjectId(),
          caseNumber: 'CASE-101',
          title: 'First Case',
          status: 'open',
          priority: 'high',
          createdAt: new Date(),
          updatedAt: new Date(),
        },
        {
          _id: new Types.ObjectId(),
          caseNumber: 'CASE-102',
          title: 'Second Case',
          status: 'closed',
          priority: 'low',
          createdAt: new Date(),
          updatedAt: new Date(),
        },
      ])
    );

    // Composite.find spy
    jest.spyOn(Composite, 'find').mockReturnValue(
      createChainableQuery([
        { variations: [{ variationIndex: 0 }, { variationIndex: 1 }] },
        { variations: [{ variationIndex: 0 }] },
      ])
    );

    // GenerationJob.find spy
    jest.spyOn(GenerationJob, 'find').mockReturnValue(
      createChainableQuery([
        {
          startedAt: new Date('2026-11-01T10:00:00.000Z'),
          completedAt: new Date('2026-11-01T10:00:04.500Z'),
        },
        {
          startedAt: new Date('2026-11-01T11:00:00.000Z'),
          completedAt: new Date('2026-11-01T11:00:05.500Z'),
        },
      ])
    );

    // User.findById spy
    jest.spyOn(User, 'findById').mockReturnValue(
      createChainableQuery({
        _id: new Types.ObjectId(mockUserId),
        name: 'Detective Jane Doe',
        email: 'jane@agency.gov',
        role: 'investigator',
        investigatorId: 'INV-4042',
        badgeNumber: '4042',
      })
    );

    // MLClient spy
    jest.spyOn(mlClient, 'getModelStatus').mockResolvedValue({
      service: 'Phantom AI ML',
      device: {} as any,
      all_models_trained: true,
      models: {
        cgan_generator: {
          is_trained: true,
          checkpoint_exists: true,
          model_name: 'ConditionalWGANGPGenerator',
          parameter_count: 4102336,
          architecture: 'WGAN-GP',
          checkpoint_path: '/path/cgan.pt',
          input_shape: '',
          output_shape: '',
          notes: '',
        },
        nlp_bilstm: {
          is_trained: true,
          checkpoint_exists: true,
          model_name: 'CharAwareBiLSTMAttributeLabeler',
          parameter_count: 2599864,
          architecture: 'BiLSTM-CRF',
          checkpoint_path: '/path/nlp.pt',
          input_shape: '',
          output_shape: '',
          notes: '',
        },
      },
    });
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('calculates Active Cases count strictly from open/in_progress accessible cases', async () => {
    const overview = await getDashboardOverview(mockUserId, 'investigator');
    // Out of 2 cases (1 open, 1 closed), exactly 1 is active
    expect(overview.metrics.activeCases.value).toBe(1);
    expect(overview.metrics.activeCases.label).toBe('Active Cases');
  });

  it('calculates Suspects Generated from actual composite variations without fabricating values', async () => {
    const overview = await getDashboardOverview(mockUserId, 'investigator');
    // 2 variations + 1 variation = 3
    expect(overview.metrics.suspectsGenerated.value).toBe(3);
    expect(overview.metrics.suspectsGenerated.value).not.toBe(1284);
  });

  it('reports AI Analysis Accuracy as "Not yet validated" rather than an invented percentage', async () => {
    const overview = await getDashboardOverview(mockUserId, 'investigator');
    expect(overview.metrics.aiAnalysisAccuracy.value).toBe('Not yet validated');
    expect(overview.metrics.aiAnalysisAccuracy.value).not.toBe(94);
    expect(overview.metrics.aiAnalysisAccuracy.value).not.toBe('94%');
  });

  it('aggregates Database Records across all 7 MongoDB collections accurately', async () => {
    const overview = await getDashboardOverview(mockUserId, 'investigator');
    // 10 + 5 + 8 + 2 + 4 + 3 + 25 = 57
    expect(overview.metrics.databaseRecords.value).toBe(57);
    expect(overview.metrics.databaseRecords.value).not.toBe(58302);
    expect(overview.metrics.databaseRecords.breakdown.cases).toBe(10);
    expect(overview.metrics.databaseRecords.breakdown.securityAudits).toBe(25);
  });

  it('computes empirical mean latency for completed jobs and avoids fabricated 38s', async () => {
    const overview = await getDashboardOverview(mockUserId, 'investigator');
    // 4.5s and 5.5s -> mean is 5.0s
    expect(overview.metrics.averageProcessingTime.value).toBe(5);
    expect(overview.metrics.averageProcessingTime.suffix).toBe('s');
    expect(overview.metrics.averageProcessingTime.sampleSize).toBe(2);
    expect(overview.metrics.averageProcessingTime.value).not.toBe(38);
  });

  it('gracefully handles zero completed jobs by returning "Insufficient data"', async () => {
    jest.spyOn(GenerationJob, 'find').mockReturnValue(createChainableQuery([]));

    const overview = await getDashboardOverview(mockUserId, 'investigator');
    expect(overview.metrics.averageProcessingTime.value).toBe('Insufficient data');
    expect(overview.metrics.averageProcessingTime.suffix).toBeUndefined();
    expect(overview.metrics.averageProcessingTime.sampleSize).toBe(0);
  });

  it('queries live subsystem statuses and reflects offline models when ML service fails', async () => {
    jest.spyOn(mlClient, 'getModelStatus').mockRejectedValue(new Error('Connection refused'));

    const overview = await getDashboardOverview(mockUserId, 'investigator');
    const reconstruction = overview.systemHealth.find((s) => s.key === 'reconstruction_model');
    const extraction = overview.systemHealth.find((s) => s.key === 'extraction_model');

    expect(reconstruction?.status).toBe('Offline');
    expect(reconstruction?.tone).toBe('critical');
    expect(extraction?.status).toBe('Offline');
    expect(extraction?.tone).toBe('critical');
    expect(overview.metrics.systemStatus.numericPercent).toBeLessThan(100);
  });

  it('returns authenticated investigator profile rather than a hardcoded persona', async () => {
    const overview = await getDashboardOverview(mockUserId, 'investigator');
    expect(overview.investigator.name).toBe('Detective Jane Doe');
    expect(overview.investigator.name).not.toBe('Dr. Aris Thorne');
    expect(overview.investigator.badgeNumber).toBe('4042');
  });

  it('returns empty recent activity for an investigator with no cases without fallback mock data', async () => {
    jest.spyOn(Case, 'find').mockReturnValue(createChainableQuery([]));

    const overview = await getDashboardOverview(mockUserId, 'investigator');
    expect(overview.metrics.activeCases.value).toBe(0);
    expect(overview.metrics.suspectsGenerated.value).toBe(0);
    expect(overview.recentActivity).toEqual([]);
    expect(overview.recentActivity.some((a) => a.id.includes('CASE-2026-048'))).toBe(false);
  });
});
