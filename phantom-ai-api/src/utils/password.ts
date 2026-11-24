import argon2 from 'argon2';
import bcrypt from 'bcryptjs';

/**
 * Hash a plain passcode using Argon2id with secure forensic platform parameters.
 */
export async function hashPassword(plainText: string): Promise<string> {
  return argon2.hash(plainText, {
    type: argon2.argon2id,
    memoryCost: 19456, // 19 MiB
    timeCost: 2,
    parallelism: 1,
  });
}

/**
 * Verify a candidate passcode against a stored hash.
 * Supports legacy bcrypt hashes ($2a$, $2b$, $2y$) and indicates if re-hashing
 * to Argon2id is needed.
 */
export async function verifyPassword(
  candidate: string,
  storedHash: string
): Promise<{ valid: boolean; needsRehash: boolean }> {
  if (!storedHash) {
    return { valid: false, needsRehash: false };
  }

  // Check for legacy bcrypt hash
  if (
    storedHash.startsWith('$2a$') ||
    storedHash.startsWith('$2b$') ||
    storedHash.startsWith('$2y$')
  ) {
    try {
      const valid = await bcrypt.compare(candidate, storedHash);
      return { valid, needsRehash: valid };
    } catch {
      return { valid: false, needsRehash: false };
    }
  }

  // Verify Argon2id
  try {
    const valid = await argon2.verify(storedHash, candidate);
    return { valid, needsRehash: false };
  } catch {
    return { valid: false, needsRehash: false };
  }
}

