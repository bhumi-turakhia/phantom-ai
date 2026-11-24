import { useState, type FormEvent } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Fingerprint,
  CheckCircle2,
  AlertTriangle,
  Copy,
  ArrowRight,
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import { FingerprintMotif } from "../components/ui/FingerprintMotif";
import { ForensicGridBackground } from "../components/ui/ForensicGridBackground";
import { PhantomLogo } from "../components/ui/PhantomLogo";
import { Input } from "../components/ui/Input";
import { Button } from "../components/ui/Button";
import { ClearanceBadge } from "../components/ui/Badge";
import { register, registerPasskey, type InvestigatorUser } from "../services/authService";

export default function Register() {
  const navigate = useNavigate();

  const [form, setForm] = useState({
    name: "",
    email: "",
    badgeNumber: "",
    department: "",
    password: "",
    confirm: "",
  });

  const [step, setStep] = useState<"FORM" | "ENROLL_PASSKEY" | "RECOVERY_CODES">("FORM");
  const [registeredUser, setRegisteredUser] = useState<InvestigatorUser | null>(null);
  const [preAuthToken, setPreAuthToken] = useState<string>("");
  const [recoveryCodes, setRecoveryCodes] = useState<string[]>([]);
  const [copiedCodes, setCopiedCodes] = useState(false);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function update(key: keyof typeof form) {
    return (e: React.ChangeEvent<HTMLInputElement>) =>
      setForm((prev) => ({
        ...prev,
        [key]: e.target.value,
      }));
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);

    if (!form.name || !form.email || !form.password || !form.confirm) {
      setError("Please complete all required fields.");
      return;
    }

    if (form.password !== form.confirm) {
      setError("Passcodes do not match.");
      return;
    }

    if (!/(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}/.test(form.password)) {
      setError(
        "Passcode must contain at least 8 characters, one uppercase letter, one lowercase letter, and one number."
      );
      return;
    }

    try {
      setLoading(true);
      const response = await register(
        form.name.trim(),
        form.email.trim().toLowerCase(),
        form.password,
        form.badgeNumber.trim() || undefined,
        form.department.trim() || undefined
      );

      setRegisteredUser(response.user);
      setPreAuthToken(response.preAuthToken);
      setStep("ENROLL_PASSKEY");
    } catch (err: any) {
      setError(
        err?.response?.data?.message ||
        "Registration failed. Please check your details and try again."
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleEnrollPasskey() {
    if (!preAuthToken) {
      setError("Registration token expired. Please re-register or log in.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const res = await registerPasskey(preAuthToken, "Primary Platform Key");
      if (res.recoveryCodes && res.recoveryCodes.length > 0) {
        setRecoveryCodes(res.recoveryCodes);
        setStep("RECOVERY_CODES");
      } else {
        navigate("/dashboard");
      }
    } catch (err: any) {
      console.error("Passkey registration failed:", err);
      if (err.name === "NotAllowedError") {
        setError("Device verification was cancelled. Click below to retry.");
      } else {
        setError(
          err?.response?.data?.message ||
          err?.message ||
          "Failed to register platform authenticator."
        );
      }
    } finally {
      setLoading(false);
    }
  }

  function copyCodes() {
    navigator.clipboard.writeText(recoveryCodes.join("\n"));
    setCopiedCodes(true);
    setTimeout(() => setCopiedCodes(false), 2000);
  }

  return (
    <div className="relative flex min-h-screen items-center justify-center bg-[#F3F6FA] overflow-hidden py-10 px-4">
      <ForensicGridBackground variant="auth" animated={true} />
      <FingerprintMotif
        className="pointer-events-none absolute -top-24 -left-24 h-[480px] w-[480px] z-0"
        opacity={0.05}
      />
      <FingerprintMotif
        className="pointer-events-none absolute -bottom-32 -right-32 h-[420px] w-[420px] z-0"
        opacity={0.04}
      />

      <motion.div
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="relative z-10 w-full max-w-md rounded-md border border-[#DCE4EE] bg-white p-8 shadow-panel"
      >
        <div className="mb-5 pb-4 border-b border-[#DCE4EE]">
          <PhantomLogo variant="full" size="sm" theme="light" showTagline={true} />
        </div>

        <div className="mb-4 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-ink-primary">
              Investigator Onboarding
            </h1>
            <p className="label-technical text-ink-disabled mt-0.5">
              Identity Verification & FIDO2 Enrollment
            </p>
          </div>
          <ClearanceBadge level="ELEVATED" />
        </div>

        {error && (
          <div className="mb-4 flex items-start gap-2 rounded border border-status-critical/30 bg-status-critical/5 p-3 text-xs text-status-critical">
            <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
            <span>{error}</span>
          </div>
        )}

        <AnimatePresence mode="wait">
          {step === "FORM" && (
            <motion.form
              key="step-form"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onSubmit={handleSubmit}
              className="space-y-4"
            >
              <Input
                id="name"
                label="Full name"
                placeholder="Agent Jane Doe"
                value={form.name}
                onChange={update("name")}
                required
              />

              <Input
                id="email"
                label="Agency email"
                type="email"
                placeholder="j.doe@agency.gov"
                value={form.email}
                onChange={update("email")}
                required
              />

              <div className="grid grid-cols-2 gap-3">
                <Input
                  id="badgeNumber"
                  label="Badge number (opt.)"
                  placeholder="8821"
                  value={form.badgeNumber}
                  onChange={update("badgeNumber")}
                />
                <Input
                  id="department"
                  label="Division / Unit"
                  placeholder="Forensics"
                  value={form.department}
                  onChange={update("department")}
                />
              </div>

              <Input
                id="password"
                label="Create passcode"
                type="password"
                placeholder="••••••••••••"
                value={form.password}
                onChange={update("password")}
                required
              />

              <Input
                id="confirm"
                label="Confirm passcode"
                type="password"
                placeholder="••••••••••••"
                value={form.confirm}
                onChange={update("confirm")}
                required
              />

              <p className="text-[11px] text-ink-disabled">
                Passcode must contain 8+ characters, including upper, lower, and numeric characters.
              </p>

              <Button
                type="submit"
                loading={loading}
                className="w-full mt-2"
              >
                Continue to Device Enrollment →
              </Button>
            </motion.form>
          )}

          {step === "ENROLL_PASSKEY" && (
            <motion.div
              key="step-enroll"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              className="space-y-5"
            >
              <div className="rounded border border-gold/30 bg-gold/5 p-4 text-center">
                <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-gold/20">
                  <Fingerprint className="h-6 w-6 text-gold animate-pulse" />
                </div>
                <h3 className="text-sm font-semibold text-ink-primary">
                  Account Created Successfully
                </h3>
                <div className="mt-2 inline-block rounded bg-bg-inset border border-line px-3 py-1 font-mono text-xs text-gold">
                  Investigator ID: {registeredUser?.investigatorId}
                </div>
                <p className="text-xs text-ink-secondary mt-3">
                  To complete security authorization, register this device&apos;s platform authenticator (Touch ID / Face ID / Windows Hello).
                </p>
              </div>

              <Button
                type="button"
                onClick={handleEnrollPasskey}
                loading={loading}
                className="w-full"
              >
                <Fingerprint className="h-4 w-4 mr-2" />
                Enroll Device Passkey
              </Button>

              <div className="text-center">
                <button
                  type="button"
                  onClick={() => navigate("/login")}
                  className="text-xs text-ink-tertiary hover:text-gold"
                >
                  Enroll later from login page →
                </button>
              </div>
            </motion.div>
          )}

          {step === "RECOVERY_CODES" && (
            <motion.div
              key="step-recovery"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              className="space-y-4"
            >
              <div className="rounded border border-status-verified/40 bg-status-verified/10 p-4">
                <div className="flex items-center gap-2 text-status-verified font-medium text-sm mb-1">
                  <CheckCircle2 className="h-4 w-4" />
                  Passkey Successfully Enrolled
                </div>
                <p className="text-xs text-ink-secondary">
                  Save your emergency single-use recovery codes in a secure agency vault. If your physical device is lost or damaged, these codes restore access.
                </p>
              </div>

              <div className="grid grid-cols-2 gap-2 rounded border border-line bg-bg-inset p-3 font-mono text-xs text-ink-primary text-center">
                {recoveryCodes.map((code, idx) => (
                  <div key={idx} className="rounded bg-bg-elevated py-1.5 border border-line/60">
                    {code}
                  </div>
                ))}
              </div>

              <div className="flex gap-2">
                <Button
                  type="button"
                  variant="secondary"
                  onClick={copyCodes}
                  className="w-1/2 text-xs"
                >
                  <Copy className="h-3.5 w-3.5 mr-1" />
                  {copiedCodes ? "Copied!" : "Copy Codes"}
                </Button>
                <Button
                  type="button"
                  variant="primary"
                  onClick={() => navigate("/dashboard")}
                  className="w-1/2 text-xs"
                >
                  Enter Dashboard
                  <ArrowRight className="h-3.5 w-3.5 ml-1" />
                </Button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="mt-6 flex items-center justify-between border-t border-line pt-4 text-xs">
          <span className="label-technical text-ink-disabled">
            Security Clearance Level 3
          </span>
          {step === "FORM" && (
            <a
              href="/"
              className="text-gold hover:text-gold-bright transition-colors"
            >
              Already registered? Login →
            </a>
          )}
        </div>
      </motion.div>
    </div>
  );
}
