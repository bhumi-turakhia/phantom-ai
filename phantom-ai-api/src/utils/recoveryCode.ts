import crypto from 'crypto';

/**
 * Generates a unique, non-guessable investigator ID.
 * Format: INV-XXXX-XXXX (e.g. INV-7741-9E2A or INV-B4D2-89CF)
 */
export function generateInvestigatorId(badgeNumber?: string): string {
  const prefix = badgeNumber ? badgeNumber.replace(/[^A-Za-z0-9]/g, '').slice(0, 4).toUpperCase() : null;
  const part1 = prefix && prefix.length === 4 ? prefix : crypto.randomBytes(2).toString('hex').toUpperCase();
  const part2 = crypto.randomBytes(2).toString('hex').toUpperCase();
  const part3 = crypto.randomBytes(2).toString('hex').toUpperCase();
  return `INV-${part1}-${part2}${part3}`;
}

/**
 * Generates 6 cryptographically secure single-use recovery codes.
 * Format: 8 characters split by hyphen, e.g. "A9F2-4D8B"
 */
export function generateRecoveryCodes(count = 6): { plainCodes: string[]; hashedCodes: string[] } {
  const plainCodes: string[] = [];
  const hashedCodes: string[] = [];

  for (let i = 0; i < count; i++) {
    const raw = crypto.randomBytes(4).toString('hex').toUpperCase();
    const formatted = `${raw.slice(0, 4)}-${raw.slice(4, 8)}`;
    plainCodes.push(formatted);
    hashedCodes.push(hashRecoveryCode(formatted));
  }

  return { plainCodes, hashedCodes };
}

/**
 * Hashes a recovery code using SHA-256 with normalized casing and spacing.
 */
export function hashRecoveryCode(code: string): string {
  const normalized = code.replace(/[^A-Za-z0-9]/g, '').toUpperCase();
  return crypto.createHash('sha256').update(normalized).digest('hex');
}

/**
 * Verifies if candidate code matches stored hash.
 */
export function verifyRecoveryCode(candidate: string, storedHash: string): boolean {
  const hashedCandidate = hashRecoveryCode(candidate);
  return crypto.timingSafeEqual(Buffer.from(hashedCandidate), Buffer.from(storedHash));
}

