import { Schema, model, Document, Types } from 'mongoose';
import { hashPassword, verifyPassword } from '../utils/password';

export type UserRole = 'admin' | 'investigator' | 'supervisor' | 'analyst';

export interface IPasskeyCredential {
  credentialID: string;
  credentialPublicKey: string; // base64url encoded
  counter: number;
  transports?: string[];
  deviceType?: string;
  backedUp?: boolean;
  name?: string;
  createdAt: Date;
}

export interface IRecoveryCode {
  codeHash: string;
  used: boolean;
  usedAt?: Date;
  createdAt: Date;
}

export interface IUser extends Document {
  _id: Types.ObjectId;
  investigatorId: string;
  name: string;
  email: string;
  password: string;
  role: UserRole;
  badgeNumber?: string;
  department?: string;
  isActive: boolean;
  lastLogin?: Date;
  passkeys: IPasskeyCredential[];
  currentChallenge?: {
    challenge: string;
    expiresAt: Date;
    flow: 'registration' | 'authentication';
  };
  recoveryCodes: IRecoveryCode[];
  failedLoginAttempts: number;
  lockUntil?: Date;
  createdAt: Date;
  updatedAt: Date;
  comparePassword(candidatePassword: string): Promise<{ valid: boolean; needsRehash: boolean }>;
  isLocked(): boolean;
  incLoginAttempts(): Promise<void>;
  resetLoginAttempts(): Promise<void>;
}

const passkeySchema = new Schema<IPasskeyCredential>(
  {
    credentialID: { type: String, required: true },
    credentialPublicKey: { type: String, required: true },
    counter: { type: Number, required: true, default: 0 },
    transports: [{ type: String }],
    deviceType: { type: String },
    backedUp: { type: Boolean, default: false },
    name: { type: String, default: 'Platform Authenticator' },
    createdAt: { type: Date, default: Date.now },
  },
  { _id: false }
);

const recoveryCodeSchema = new Schema<IRecoveryCode>(
  {
    codeHash: { type: String, required: true },
    used: { type: Boolean, default: false },
    usedAt: { type: Date },
    createdAt: { type: Date, default: Date.now },
  },
  { _id: false }
);

const userSchema = new Schema<IUser>(
  {
    investigatorId: {
      type: String,
      unique: true,
      sparse: true,
      trim: true,
      uppercase: true,
    },
    name: {
      type: String,
      required: [true, 'Name is required'],
      trim: true,
      minlength: [2, 'Name must be at least 2 characters'],
      maxlength: [100, 'Name must not exceed 100 characters'],
    },
    email: {
      type: String,
      required: [true, 'Email is required'],
      unique: true,
      lowercase: true,
      trim: true,
      match: [/^\S+@\S+\.\S+$/, 'Please provide a valid email address'],
    },
    password: {
      type: String,
      required: [true, 'Password is required'],
      minlength: [8, 'Password must be at least 8 characters'],
      select: false,
    },
    role: {
      type: String,
      enum: {
        values: ['admin', 'investigator', 'supervisor', 'analyst'],
        message: '{VALUE} is not a valid role',
      },
      default: 'investigator',
    },
    badgeNumber: {
      type: String,
      trim: true,
    },
    department: {
      type: String,
      trim: true,
    },
    isActive: {
      type: Boolean,
      default: true,
    },
    lastLogin: {
      type: Date,
    },
    passkeys: {
      type: [passkeySchema],
      default: [],
    },
    currentChallenge: {
      challenge: { type: String },
      expiresAt: { type: Date },
      flow: { type: String, enum: ['registration', 'authentication'] },
    },
    recoveryCodes: {
      type: [recoveryCodeSchema],
      default: [],
    },
    failedLoginAttempts: {
      type: Number,
      default: 0,
    },
    lockUntil: {
      type: Date,
    },
  },
  {
    timestamps: true,
    toJSON: {
      transform: (_doc, ret) => {
        delete (ret as Record<string, unknown>).password;
        delete (ret as Record<string, unknown>).currentChallenge;
        delete (ret as Record<string, unknown>).recoveryCodes;
        return ret;
      },
    },
  }
);

userSchema.index({ role: 1 });
userSchema.index({ isActive: 1 });

userSchema.pre('save', async function (next) {
  if (!this.isModified('password')) return next();

  // If already hashed (Argon2 or bcrypt), do not re-hash
  if (
    this.password.startsWith('$argon2') ||
    this.password.startsWith('$2a$') ||
    this.password.startsWith('$2b$')
  ) {
    return next();
  }

  this.password = await hashPassword(this.password);
  next();
});

userSchema.methods.comparePassword = async function (
  candidatePassword: string
): Promise<{ valid: boolean; needsRehash: boolean }> {
  return verifyPassword(candidatePassword, this.password);
};

userSchema.methods.isLocked = function (): boolean {
  return !!(this.lockUntil && this.lockUntil.getTime() > Date.now());
};

userSchema.methods.incLoginAttempts = async function (): Promise<void> {
  // If a previous lock has expired, restart count at 1
  if (this.lockUntil && this.lockUntil.getTime() < Date.now()) {
    await this.updateOne({
      $set: { failedLoginAttempts: 1 },
      $unset: { lockUntil: 1 },
    });
    return;
  }

  const updates: any = { $inc: { failedLoginAttempts: 1 } };
  // Lock for 15 minutes if attempts reach 5
  if (this.failedLoginAttempts + 1 >= 5 && !this.isLocked()) {
    updates.$set = { lockUntil: new Date(Date.now() + 15 * 60 * 1000) };
  }

  await this.updateOne(updates);
};

userSchema.methods.resetLoginAttempts = async function (): Promise<void> {
  await this.updateOne({
    $set: { failedLoginAttempts: 0 },
    $unset: { lockUntil: 1 },
  });
};

export const User = model<IUser>('User', userSchema);
