import crypto from 'crypto';
import dotenv from 'dotenv';
import path from 'path';
import { connectDB, disconnectDB } from './config/db';
import { User } from './models/User.model';
import { hashPassword } from './utils/password';
import { generateInvestigatorId } from './utils/recoveryCode';

dotenv.config({ path: path.join(__dirname, '../.env') });

export interface SeedOptions {
  email?: string;
  name?: string;
  passcode?: string;
  investigatorId?: string;
  role?: 'admin' | 'investigator' | 'supervisor' | 'analyst';
  badgeNumber?: string;
  department?: string;
  nodeEnv?: string;
  deactivateAccount?: boolean;
  reactivateAccount?: boolean;
  silent?: boolean;
}

export interface SeedResult {
  action: 'created' | 'verified_unchanged' | 'updated_passcode' | 'deactivated' | 'reactivated';
  email: string;
  investigatorId?: string;
  oneTimePasscodeGenerated: boolean;
  oneTimePasscode?: string;
}

export const INSECURE_DEFAULT_PASSWORDS = new Set([
  'ForensicDev2026!Secure',
  'password',
  'password123',
  'admin',
  'admin123',
  'changeme',
  '12345678',
  'phantom2026',
]);

/**
 * Validates passcode against forensic security standards.
 * Refuses default/weak passcodes and enforces strict complexity in production.
 */
export function validatePasscodePolicy(passcode: string, isProd: boolean): void {
  if (INSECURE_DEFAULT_PASSWORDS.has(passcode)) {
    throw new Error('Refusing insecure configuration: Cannot use known default or compromised development credentials.');
  }

  if (isProd) {
    if (passcode.length < 12) {
      throw new Error('Refusing insecure configuration: Production passcode must be at least 12 characters long.');
    }
    const hasUpper = /[A-Z]/.test(passcode);
    const hasLower = /[a-z]/.test(passcode);
    const hasDigit = /[0-9]/.test(passcode);
    const hasSpecial = /[^A-Za-z0-9]/.test(passcode);
    if (!hasUpper || !hasLower || !hasDigit || !hasSpecial) {
      throw new Error(
        'Refusing insecure configuration: Production passcode must contain uppercase, lowercase, digit, and special characters.'
      );
    }
  } else {
    if (passcode.length < 10) {
      throw new Error('Passcode must be at least 10 characters long.');
    }
  }
}

/**
 * Executes secure initialization or maintenance for administrator/investigator accounts.
 */
export async function seedDatabase(options?: SeedOptions): Promise<SeedResult> {
  const isProduction = (options?.nodeEnv || process.env.NODE_ENV) === 'production';
  const email = (options?.email || process.env.SEED_ADMIN_EMAIL || 'admin@phantomai.gov').toLowerCase().trim();

  const existing = await User.findOne({ email });

  // 1. Safe maintenance on existing accounts
  if (existing) {
    let modified = false;

    // Check account deactivation request
    if (options?.deactivateAccount || process.env.SEED_DEACTIVATE_ACCOUNT === 'true') {
      existing.isActive = false;
      await existing.save({ validateBeforeSave: false });
      if (!options?.silent) {
        process.stdout.write(`Account ${email} has been safely deactivated (isActive: false).\n`);
      }
      return {
        action: 'deactivated',
        email,
        investigatorId: existing.investigatorId,
        oneTimePasscodeGenerated: false,
      };
    }

    // Check account reactivation request
    if (options?.reactivateAccount || process.env.SEED_REACTIVATE_ACCOUNT === 'true') {
      existing.isActive = true;
      await existing.save({ validateBeforeSave: false });
      if (!options?.silent) {
        process.stdout.write(`Account ${email} has been reactivated (isActive: true).\n`);
      }
      return {
        action: 'reactivated',
        email,
        investigatorId: existing.investigatorId,
        oneTimePasscodeGenerated: false,
      };
    }

    // Ensure Investigator ID exists on existing account
    if (!existing.investigatorId) {
      existing.investigatorId = 'INV-7741-0001';
      modified = true;
    }

    // If an explicit passcode rotation was requested
    const requestedPasscode = options?.passcode || process.env.SEED_INITIAL_PASSCODE;
    if (requestedPasscode) {
      validatePasscodePolicy(requestedPasscode, isProduction);
      existing.password = await hashPassword(requestedPasscode);
      modified = true;
      if (!options?.silent) {
        process.stdout.write(`Passcode updated with Argon2id hash for ${email} from environment configuration.\n`);
      }
      if (modified) {
        await existing.save({ validateBeforeSave: false });
      }
      return {
        action: 'updated_passcode',
        email,
        investigatorId: existing.investigatorId,
        oneTimePasscodeGenerated: false,
      };
    }

    if (modified) {
      await existing.save({ validateBeforeSave: false });
    }

    if (!options?.silent) {
      process.stdout.write(`Existing investigator account preserved without modification: ${existing.investigatorId} (${existing.email}).\n`);
    }

    return {
      action: 'verified_unchanged',
      email,
      investigatorId: existing.investigatorId,
      oneTimePasscodeGenerated: false,
    };
  }

  // 2. Initial Account Creation: No default known credentials allowed
  let passcode = options?.passcode || process.env.SEED_INITIAL_PASSCODE;
  let isRandom = false;

  if (passcode) {
    validatePasscodePolicy(passcode, isProduction);
  } else {
    // Generate a high-entropy, cryptographically random one-time credential
    passcode = crypto.randomBytes(24).toString('base64url');
    isRandom = true;
  }

  let assignedInvestigatorId = options?.investigatorId || process.env.SEED_ADMIN_INVESTIGATOR_ID;
  if (!assignedInvestigatorId) {
    const existingWithDefault = await User.findOne({ investigatorId: 'INV-7741-0001' });
    if (!existingWithDefault) {
      assignedInvestigatorId = 'INV-7741-0001';
    } else {
      assignedInvestigatorId = generateInvestigatorId(options?.badgeNumber || '7741');
      let collision = await User.findOne({ investigatorId: assignedInvestigatorId });
      while (collision) {
        assignedInvestigatorId = generateInvestigatorId(options?.badgeNumber || '7741');
        collision = await User.findOne({ investigatorId: assignedInvestigatorId });
      }
    }
  }

  const hashedPassword = await hashPassword(passcode);
  const user = await User.create({
    investigatorId: assignedInvestigatorId,
    name: options?.name || process.env.SEED_ADMIN_NAME || 'Lead Forensic Administrator',
    email,
    password: hashedPassword,
    role: options?.role || (process.env.SEED_ADMIN_ROLE as any) || 'admin',
    badgeNumber: options?.badgeNumber || process.env.SEED_ADMIN_BADGE || '7741',
    department: options?.department || process.env.SEED_ADMIN_DEPARTMENT || 'Forensics Intelligence Unit',
    isActive: true,
  });

  if (isRandom && !options?.silent) {
    // Output ONLY ONCE to interactive terminal stdout, never log to persistent application logs
    process.stdout.write('\n================================================================\n');
    process.stdout.write('[SECURITY NOTICE] Initial Administrator Account Created:\n');
    process.stdout.write(`Email: ${user.email}\n`);
    process.stdout.write(`Investigator ID: ${user.investigatorId}\n`);
    process.stdout.write(`One-Time Passcode: ${passcode}\n`);
    process.stdout.write('IMPORTANT: Store this one-time passcode securely. Immediately log in\n');
    process.stdout.write('and enroll your FIDO2/WebAuthn device passkey. This credential will\n');
    process.stdout.write('never be displayed again.\n');
    process.stdout.write('================================================================\n\n');
  } else if (!options?.silent) {
    process.stdout.write(`Administrator account created with configured credentials for ${user.email} (${user.investigatorId}).\n`);
  }

  return {
    action: 'created',
    email: user.email,
    investigatorId: user.investigatorId,
    oneTimePasscodeGenerated: isRandom,
    oneTimePasscode: isRandom ? passcode : undefined,
  };
}

// Standalone execution wrapper
if (require.main === module) {
  connectDB()
    .then(() => seedDatabase())
    .then(async () => {
      await disconnectDB();
      process.exit(0);
    })
    .catch(async (err) => {
      process.stderr.write(`[SEED ERROR] ${err.message}\n`);
      await disconnectDB().catch(() => {});
      process.exit(1);
    });
}
