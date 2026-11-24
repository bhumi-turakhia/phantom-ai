import { User } from '../models/User.model';
import { generateInvestigatorId } from '../utils/recoveryCode';
import { logger } from '../config/logger';

/**
 * Ensures all existing and migrated accounts have a unique investigator ID.
 */
export async function runAccountMigrations(): Promise<void> {
  try {
    const unassignedUsers = await User.find({
      $or: [{ investigatorId: { $exists: false } }, { investigatorId: null }, { investigatorId: '' }],
    });

    if (unassignedUsers.length > 0) {
      logger.info(`Migrating ${unassignedUsers.length} user accounts to assign Investigator IDs...`);

      for (const user of unassignedUsers) {
        let assignedId = generateInvestigatorId(user.badgeNumber);
        // Ensure uniqueness
        let existingWithId = await User.findOne({ investigatorId: assignedId });
        while (existingWithId) {
          assignedId = generateInvestigatorId();
          existingWithId = await User.findOne({ investigatorId: assignedId });
        }

        user.investigatorId = assignedId;
        await user.save({ validateBeforeSave: false });
        logger.info(`Assigned Investigator ID ${assignedId} to user ${user.email}`);
      }
    }
  } catch (err: any) {
    logger.error(`Account migration failed: ${err.message}`);
  }
}

