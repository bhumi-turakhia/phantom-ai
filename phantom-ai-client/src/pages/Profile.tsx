import { useState, useEffect } from "react";
import {
  ShieldCheck,
  UserCheck,
  KeyRound,
  Building2,
  BadgeCheck,
  Lock,
  AlertCircle,
  CheckCircle2,
  Loader2,
} from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "../components/ui/Card";
import { Input } from "../components/ui/Input";
import { Button } from "../components/ui/Button";
import { StatusBadge } from "../components/ui/Badge";
import { PageTransition } from "../components/layout/PageTransition";
import {
  getCurrentUser,
  updateProfile,
  type InvestigatorUser,
} from "../services/authService";

export default function Profile() {
  const [user, setUser] = useState<InvestigatorUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Form states
  const [name, setName] = useState("");
  const [badgeNumber, setBadgeNumber] = useState("");
  const [department, setDepartment] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState<string | null>(null);

  // Password update states
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [isUpdatingPassword, setIsUpdatingPassword] = useState(false);
  const [passwordSuccess, setPasswordSuccess] = useState<string | null>(null);
  const [passwordError, setPasswordError] = useState<string | null>(null);

  useEffect(() => {
    async function loadProfile() {
      setIsLoading(true);
      setError(null);
      try {
        const profile = await getCurrentUser();
        setUser(profile);
        setName(profile.name || "");
        setBadgeNumber(profile.badgeNumber || "");
        setDepartment(profile.department || "");
        if (typeof window !== "undefined") {
          localStorage.setItem("user", JSON.stringify(profile));
        }
      } catch (err: any) {
        setError(
          err?.response?.data?.message ||
            err?.message ||
            "Profile information unavailable. Please re-authenticate."
        );
      } finally {
        setIsLoading(false);
      }
    }
    loadProfile();
  }, []);

  const handleSaveProfile = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      setError("Full name cannot be empty");
      return;
    }

    setIsSaving(true);
    setSaveSuccess(null);
    setError(null);

    try {
      const updated = await updateProfile({
        name: name.trim(),
        badgeNumber: badgeNumber.trim(),
        department: department.trim(),
      });
      setUser(updated);
      setSaveSuccess("Investigator profile updated successfully.");
      setTimeout(() => setSaveSuccess(null), 4000);
    } catch (err: any) {
      setError(
        err?.response?.data?.message ||
          err?.message ||
          "Failed to update profile attributes."
      );
    } finally {
      setIsSaving(false);
    }
  };

  const handleUpdatePassword = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentPassword) {
      setPasswordError("Current passcode is required");
      return;
    }
    if (newPassword.length < 8) {
      setPasswordError("New passcode must be at least 8 characters");
      return;
    }

    setIsUpdatingPassword(true);
    setPasswordSuccess(null);
    setPasswordError(null);

    try {
      await updateProfile({
        currentPassword,
        newPassword,
      });
      setPasswordSuccess("Passcode updated successfully.");
      setCurrentPassword("");
      setNewPassword("");
      setTimeout(() => setPasswordSuccess(null), 4000);
    } catch (err: any) {
      setPasswordError(
        err?.response?.data?.message ||
          err?.message ||
          "Failed to update passcode."
      );
    } finally {
      setIsUpdatingPassword(false);
    }
  };

  if (isLoading) {
    return (
      <PageTransition>
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <Loader2 className="w-8 h-8 animate-spin text-[#27B3A3] mb-3" />
          <p className="text-sm font-mono text-[#7893B3]">
            Authenticating investigator profile telemetry...
          </p>
        </div>
      </PageTransition>
    );
  }

  if (error && !user) {
    return (
      <PageTransition>
        <div className="max-w-xl mx-auto py-12">
          <div className="p-6 rounded-lg border border-[#D64550]/40 bg-[#D64550]/10 text-center space-y-3">
            <AlertCircle className="w-8 h-8 text-[#D64550] mx-auto" />
            <h2 className="text-base font-semibold text-white">
              Profile Unavailable
            </h2>
            <p className="text-sm text-[#B8C7DA]">{error}</p>
          </div>
        </div>
      </PageTransition>
    );
  }

  const roleLabel = (user?.role || "INVESTIGATOR").toUpperCase();

  return (
    <PageTransition>
      <div className="max-w-3xl space-y-6 pb-12">
        {/* Header with verified profile information */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[#1E2F47] pb-5">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-lg bg-[#18263A] border border-[#27B3A3]/40 flex items-center justify-center text-[#27B3A3] shrink-0">
              <UserCheck className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-ink-primary tracking-tight">
                {user?.name || "Investigator Profile"}
              </h1>
              <p className="label-technical text-ink-disabled mt-0.5">
                {user?.email} · {user?.investigatorId || "INV-ACTIVE"}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2.5 shrink-0">
            <StatusBadge tone="verified">
              CLEARANCE: {roleLabel}
            </StatusBadge>
            <StatusBadge tone={user?.hasPasskey ? "verified" : "pending"}>
              {user?.hasPasskey ? "FIDO2 PASSKEY ENROLLED" : "PASSKEY PENDING"}
            </StatusBadge>
          </div>
        </div>

        {/* Global Feedback Banners */}
        {saveSuccess && (
          <div className="flex items-center gap-2.5 p-3 rounded border border-[#27B3A3]/40 bg-[#27B3A3]/10 text-[#27B3A3] text-sm">
            <CheckCircle2 className="w-4 h-4 shrink-0" />
            <span>{saveSuccess}</span>
          </div>
        )}
        {error && (
          <div className="flex items-center gap-2.5 p-3 rounded border border-[#D64550]/40 bg-[#D64550]/10 text-[#D64550] text-sm">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Editable Profile Identity Card */}
        <Card>
          <CardHeader>
            <CardTitle>Investigator Details</CardTitle>
            <span className="label-technical text-ink-disabled">
              Permitted identity attributes
            </span>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSaveProfile} className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <Input
                  id="profile-name"
                  label="Full Name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Official investigator name"
                  required
                />
                <Input
                  id="profile-badge"
                  label="Badge / Shield Number"
                  value={badgeNumber}
                  onChange={(e) => setBadgeNumber(e.target.value)}
                  placeholder="e.g. 7741"
                />
              </div>

              <Input
                id="profile-department"
                label="Department / Unit"
                value={department}
                onChange={(e) => setDepartment(e.target.value)}
                placeholder="e.g. Forensic Facial Intelligence Division"
              />

              <div className="pt-2 flex justify-end">
                <Button type="submit" size="sm" disabled={isSaving}>
                  {isSaving ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 animate-spin mr-1.5" />
                      Saving changes...
                    </>
                  ) : (
                    "Save profile changes"
                  )}
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>

        {/* Protected Security & Clearance Card */}
        <Card>
          <CardHeader>
            <CardTitle>Protected Agency Credentials</CardTitle>
            <span className="label-technical text-ink-disabled">
              Read-only immutable security attributes
            </span>
          </CardHeader>
          <CardContent className="space-y-3.5">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div className="p-3 rounded border border-line bg-bg-inset">
                <div className="flex items-center gap-2 text-ink-secondary mb-1">
                  <BadgeCheck className="w-3.5 h-3.5 text-[#27B3A3]" />
                  <span className="label-technical text-[10px]">Investigator ID</span>
                </div>
                <div className="font-mono text-sm font-semibold text-ink-primary">
                  {user?.investigatorId || "INV-UNKNOWN"}
                </div>
              </div>

              <div className="p-3 rounded border border-line bg-bg-inset">
                <div className="flex items-center gap-2 text-ink-secondary mb-1">
                  <ShieldCheck className="w-3.5 h-3.5 text-[#27B3A3]" />
                  <span className="label-technical text-[10px]">Agency Role</span>
                </div>
                <div className="font-mono text-sm font-semibold text-ink-primary">
                  {roleLabel}
                </div>
              </div>

              <div className="p-3 rounded border border-line bg-bg-inset">
                <div className="flex items-center gap-2 text-ink-secondary mb-1">
                  <Building2 className="w-3.5 h-3.5 text-[#27B3A3]" />
                  <span className="label-technical text-[10px]">Account Status</span>
                </div>
                <div className="font-mono text-sm font-semibold text-ink-primary flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-[#27B3A3]" />
                  <span>VERIFIED ACTIVE</span>
                </div>
              </div>
            </div>

            <div className="p-3 rounded border border-line bg-bg-inset flex items-center justify-between">
              <div>
                <span className="label-technical text-[10px] text-ink-secondary">
                  Official Agency Email
                </span>
                <div className="font-mono text-sm text-ink-primary mt-0.5">
                  {user?.email}
                </div>
              </div>
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono bg-bg-surface border border-line text-ink-disabled">
                <Lock className="w-3 h-3" />
                LOCKED
              </span>
            </div>

            <p className="text-[11px] text-ink-disabled flex items-center gap-1.5 pt-1">
              <Lock className="w-3.5 h-3.5 text-ink-disabled shrink-0" />
              <span>
                Protected operational clearance and investigator IDs cannot be altered through profile forms. Contact an Agency Administrator for credential modifications.
              </span>
            </p>
          </CardContent>
        </Card>

        {/* Security & Passcode Update */}
        <Card>
          <CardHeader>
            <CardTitle>Passcode & Cryptographic Security</CardTitle>
            <span className="label-technical text-ink-disabled">
              Account passcode & authentication credentials
            </span>
          </CardHeader>
          <CardContent>
            {passwordSuccess && (
              <div className="mb-4 flex items-center gap-2 p-2.5 rounded border border-[#27B3A3]/40 bg-[#27B3A3]/10 text-[#27B3A3] text-xs">
                <CheckCircle2 className="w-4 h-4 shrink-0" />
                <span>{passwordSuccess}</span>
              </div>
            )}
            {passwordError && (
              <div className="mb-4 flex items-center gap-2 p-2.5 rounded border border-[#D64550]/40 bg-[#D64550]/10 text-[#D64550] text-xs">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{passwordError}</span>
              </div>
            )}

            <form onSubmit={handleUpdatePassword} className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <Input
                  id="profile-current-password"
                  label="Current Passcode"
                  type="password"
                  value={currentPassword}
                  onChange={(e) => setCurrentPassword(e.target.value)}
                  placeholder="Verify existing passcode"
                  required
                />
                <Input
                  id="profile-new-password"
                  label="New Passcode"
                  type="password"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  placeholder="Minimum 8 characters"
                  required
                />
              </div>

              <div className="pt-2 flex items-center justify-between">
                <span className="text-xs font-mono text-ink-disabled flex items-center gap-1.5">
                  <KeyRound className="w-3.5 h-3.5 text-[#27B3A3]" />
                  <span>Requires lowercase, uppercase & digit</span>
                </span>
                <Button
                  type="submit"
                  variant="secondary"
                  size="sm"
                  disabled={isUpdatingPassword}
                >
                  {isUpdatingPassword ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 animate-spin mr-1.5" />
                      Updating...
                    </>
                  ) : (
                    "Update passcode"
                  )}
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>
      </div>
    </PageTransition>
  );
}
