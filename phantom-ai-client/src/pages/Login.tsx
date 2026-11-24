import React, { useState, type FormEvent } from "react";
import { useNavigate, Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  Lock,
  Fingerprint,
  KeyRound,
  AlertTriangle,
  ArrowRight,
  RotateCcw,
  Copy,
  CheckCircle2,
  ShieldCheck,
  Cpu,
  ScanFace,
  Layers,
} from "lucide-react";
import { ForensicGridBackground } from "../components/ui/ForensicGridBackground";
import { PhantomLogo } from "../components/ui/PhantomLogo";
import { Input } from "../components/ui/Input";
import { Button } from "../components/ui/Button";
import {
  loginPasscode,
  authenticateWithPasskey,
  registerPasskey,
  recoverAccount,
  type PreAuthResult,
} from "../services/authService";

type AuthStage =
  | "CREDENTIALS"
  | "PASSKEY_MFA"
  | "ENROLL_PASSKEY"
  | "RECOVERY"
  | "RECOVERY_CODES_DISPLAY";

/**
 * Scientific Biometric Facial Mesh Graphic:
 * Depicts facial landmark geometry, Delaunay facial triangulation,
 * and forensic conditioning coordinate axes in Forensic Teal (#27B3A3)
 * and Steel Blue (#7893B3).
 */
const BiometricFacialMeshGraphic: React.FC = () => {
  return (
    <div className="relative w-full max-w-[460px] aspect-[4/3] rounded-lg border border-[#243752] bg-[#0C1524]/80 p-5 shadow-2xl overflow-hidden backdrop-blur-xs select-none">
      {/* Background Reticle & Viewfinder Grid */}
      <svg
        viewBox="0 0 400 300"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-full"
        aria-hidden="true"
      >
        <defs>
          <radialGradient id="meshGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#27B3A3" stopOpacity="0.18" />
            <stop offset="100%" stopColor="#27B3A3" stopOpacity="0" />
          </radialGradient>
        </defs>

        {/* Ambient Glow behind facial center */}
        <circle cx="200" cy="145" r="90" fill="url(#meshGlow)" />

        {/* Circular Coordinate Radar Reticles */}
        <circle cx="200" cy="145" r="110" stroke="#7893B3" strokeWidth="0.8" strokeDasharray="3 3" opacity="0.35" />
        <circle cx="200" cy="145" r="75" stroke="#7893B3" strokeWidth="0.8" strokeDasharray="2 4" opacity="0.4" />
        <line x1="200" y1="20" x2="200" y2="270" stroke="#7893B3" strokeWidth="0.7" strokeDasharray="3 3" opacity="0.3" />
        <line x1="60" y1="145" x2="340" y2="145" stroke="#7893B3" strokeWidth="0.7" strokeDasharray="3 3" opacity="0.3" />

        {/* Viewfinder Focus Brackets */}
        <path d="M 40 50 L 40 40 L 50 40" stroke="#27B3A3" strokeWidth="1.5" strokeLinecap="round" />
        <path d="M 360 40 L 350 40 L 350 50" stroke="#27B3A3" strokeWidth="1.5" strokeLinecap="round" transform="translate(710, 0) scale(-1, 1)" />
        <path d="M 40 240 L 40 250 L 50 250" stroke="#27B3A3" strokeWidth="1.5" strokeLinecap="round" />
        <path d="M 360 250 L 350 250 L 350 240" stroke="#27B3A3" strokeWidth="1.5" strokeLinecap="round" transform="translate(710, 0) scale(-1, 1)" />

        {/* Biometric Landmark Facial Topology Mesh */}
        <g stroke="#7893B3" strokeWidth="1" strokeOpacity="0.55" strokeLinejoin="round" strokeLinecap="round">
          {/* Cranial Forehead Triangulation */}
          <line x1="200" y1="55" x2="200" y2="95" />
          <line x1="160" y1="70" x2="200" y2="95" />
          <line x1="240" y1="70" x2="200" y2="95" />
          <line x1="135" y1="100" x2="168" y2="110" />
          <line x1="168" y1="110" x2="200" y2="95" />
          <line x1="265" y1="100" x2="232" y2="110" />
          <line x1="232" y1="110" x2="200" y2="95" />
          <line x1="160" y1="70" x2="135" y2="100" />
          <line x1="240" y1="70" x2="265" y2="100" />

          {/* Eye Orbit & Zygomatic Intersections */}
          <line x1="168" y1="110" x2="200" y2="128" />
          <line x1="232" y1="110" x2="200" y2="128" />
          <line x1="135" y1="100" x2="145" y2="148" />
          <line x1="145" y1="148" x2="168" y2="110" />
          <line x1="265" y1="100" x2="255" y2="148" />
          <line x1="255" y1="148" x2="232" y2="110" />

          {/* Nasal Bridge to Subnasale */}
          <line x1="168" y1="110" x2="182" y2="162" />
          <line x1="232" y1="110" x2="218" y2="162" />
          <line x1="200" y1="128" x2="200" y2="165" />
          <line x1="182" y1="162" x2="200" y2="165" />
          <line x1="218" y1="162" x2="200" y2="165" />
          <line x1="145" y1="148" x2="182" y2="162" />
          <line x1="255" y1="148" x2="218" y2="162" />

          {/* Oral Cavity & Mandibular Angles */}
          <line x1="200" y1="165" x2="200" y2="190" />
          <line x1="182" y1="162" x2="178" y2="195" />
          <line x1="218" y1="162" x2="222" y2="195" />
          <line x1="178" y1="195" x2="200" y2="190" />
          <line x1="222" y1="195" x2="200" y2="190" />
          <line x1="178" y1="195" x2="200" y2="206" />
          <line x1="222" y1="195" x2="200" y2="206" />
          <line x1="145" y1="148" x2="160" y2="205" />
          <line x1="160" y1="205" x2="178" y2="195" />
          <line x1="255" y1="148" x2="240" y2="205" />
          <line x1="240" y1="205" x2="222" y2="195" />

          {/* Chin Apex (Menton) Convergence */}
          <line x1="178" y1="195" x2="200" y2="238" />
          <line x1="222" y1="195" x2="200" y2="238" />
          <line x1="200" y1="206" x2="200" y2="238" />
          <line x1="160" y1="205" x2="200" y2="238" />
          <line x1="240" y1="205" x2="200" y2="238" />
        </g>

        {/* Biometric Axis Highlight Lines (Teal) */}
        <g stroke="#27B3A3" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
          <line x1="168" y1="110" x2="232" y2="110" />
          <line x1="200" y1="95" x2="200" y2="128" />
          <path d="M 182 162 L 200 165 L 218 162" fill="none" />
          <path d="M 178 195 L 200 190 L 222 195" fill="none" />
        </g>

        {/* Human Facial Silhouette Outline */}
        <path
          d="M 200 55 L 160 70 L 135 100 L 145 148 L 160 205 L 200 238 L 240 205 L 255 148 L 265 100 L 240 70 Z"
          stroke="#27B3A3"
          strokeWidth="1.8"
          strokeLinejoin="round"
          fill="none"
        />

        {/* Forensic Evidence Landmark Nodes */}
        <g fill="#27B3A3">
          <circle cx="200" cy="95" r="2.8" />
          <circle cx="168" cy="110" r="3.2" />
          <circle cx="232" cy="110" r="3.2" />
          <circle cx="145" cy="148" r="2.6" />
          <circle cx="255" cy="148" r="2.6" />
          <circle cx="200" cy="165" r="3.2" />
          <circle cx="178" cy="195" r="2.5" />
          <circle cx="222" cy="195" r="2.5" />
          <circle cx="200" cy="238" r="3.0" />
        </g>

        {/* Subtle Concentric Pulse Rings on Key Landmarks */}
        <circle cx="168" cy="110" r="7" stroke="#27B3A3" strokeWidth="0.8" opacity="0.5" />
        <circle cx="232" cy="110" r="7" stroke="#27B3A3" strokeWidth="0.8" opacity="0.5" />
        <circle cx="200" cy="165" r="8" stroke="#27B3A3" strokeWidth="0.8" opacity="0.5" />
      </svg>

      {/* Forensic Telemetry Overlays */}
      <div className="absolute top-3 left-3 flex items-center gap-2 font-mono text-[9px] uppercase tracking-wider text-[#7893B3] bg-[#101B2D]/80 px-2 py-1 rounded border border-[#243752]">
        <span className="w-1.5 h-1.5 rounded-full bg-[#27B3A3] animate-pulse" />
        <span>LANDMARK VECTORS: 20-D GCIT</span>
      </div>

      <div className="absolute bottom-3 right-3 font-mono text-[9px] uppercase tracking-wider text-[#7893B3] bg-[#101B2D]/80 px-2 py-1 rounded border border-[#243752]">
        <span>FIDELITY MESH · 64×64 RESOLUTION</span>
      </div>
    </div>
  );
};

export default function Login() {
  const navigate = useNavigate();

  const [stage, setStage] = useState<AuthStage>("CREDENTIALS");
  const [identifier, setIdentifier] = useState("");
  const [passcode, setPasscode] = useState("");
  const [recoveryCode, setRecoveryCode] = useState("");

  const [preAuthData, setPreAuthData] = useState<PreAuthResult | null>(null);
  const [issuedRecoveryCodes, setIssuedRecoveryCodes] = useState<string[]>([]);
  const [copiedCodes, setCopiedCodes] = useState(false);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [infoMessage, setInfoMessage] = useState<string | null>(null);

  // Stage 1: Passcode Verification
  async function handleCredentialSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setInfoMessage(null);

    if (!identifier.trim() || !passcode.trim()) {
      setError("Please provide both your Investigator ID / Email and your Passcode.");
      return;
    }

    setLoading(true);

    try {
      const result = await loginPasscode(identifier.trim(), passcode);
      setPreAuthData(result);

      if (result.requirePasskey && result.options) {
        setStage("PASSKEY_MFA");
        // Automatically attempt passkey verification for smoother platform UX
        tryPasskeyAuth(result.preAuthToken, result.options);
      } else if (result.requirePasskeyEnrollment) {
        setStage("ENROLL_PASSKEY");
        setInfoMessage("Passkey enrollment is required for this investigator account.");
      }
    } catch (err: any) {
      const message =
        err?.response?.data?.message ||
        err?.message ||
        "Authentication failed. Verify credentials and try again.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  // Stage 2A: WebAuthn Passkey Assertion
  async function tryPasskeyAuth(token?: string, options?: any) {
    const activeToken = token || preAuthData?.preAuthToken;
    const activeOptions = options || preAuthData?.options;

    if (!activeToken || !activeOptions) {
      setError("Missing active passkey session challenge. Please re-enter your passcode.");
      setStage("CREDENTIALS");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await authenticateWithPasskey(activeToken, activeOptions);
      navigate("/dashboard");
    } catch (err: any) {
      console.warn("Passkey verification prompt result:", err);
      if (err.name === "NotAllowedError") {
        setError("Device verification was cancelled or timed out. Click below to try again.");
      } else {
        setError(
          err?.response?.data?.message ||
            err?.message ||
            "Passkey assertion could not be verified."
        );
      }
    } finally {
      setLoading(false);
    }
  }

  // Stage 2B: Initial Device Passkey Enrollment
  async function handleEnrollPasskey() {
    if (!preAuthData?.preAuthToken) {
      setError("Session expired. Please restart login.");
      setStage("CREDENTIALS");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const res = await registerPasskey(
        preAuthData.preAuthToken,
        "Investigator Platform Key"
      );
      if (res.recoveryCodes && res.recoveryCodes.length > 0) {
        setIssuedRecoveryCodes(res.recoveryCodes);
        setStage("RECOVERY_CODES_DISPLAY");
      } else {
        navigate("/dashboard");
      }
    } catch (err: any) {
      console.error("Passkey registration failed:", err);
      if (err.name === "NotAllowedError") {
        setError("Device enrollment was cancelled. Click below to try again.");
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

  // Stage 3: Emergency Recovery
  async function handleRecoverySubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);

    if (!identifier.trim() || !passcode.trim() || !recoveryCode.trim()) {
      setError(
        "Investigator ID, Passcode, and single-use Recovery Code are all required."
      );
      return;
    }

    setLoading(true);

    try {
      const res = await recoverAccount(
        identifier.trim(),
        passcode,
        recoveryCode.trim()
      );
      setPreAuthData({
        preAuthToken: res.preAuthToken,
        requirePasskeyEnrollment: true,
        user: {
          investigatorId: identifier.toUpperCase(),
          name: "Investigator",
          email: "",
        },
      });
      setStage("ENROLL_PASSKEY");
      setInfoMessage(
        "Recovery code verified. You must now enroll a new device passkey."
      );
    } catch (err: any) {
      setError(
        err?.response?.data?.message ||
          "Emergency recovery verification failed. Check credentials and code."
      );
    } finally {
      setLoading(false);
    }
  }

  function copyRecoveryCodes() {
    navigator.clipboard.writeText(issuedRecoveryCodes.join("\n"));
    setCopiedCodes(true);
    setTimeout(() => setCopiedCodes(false), 2000);
  }

  return (
    <div className="min-h-screen w-full flex flex-col lg:flex-row bg-[#101B2D] text-[#18263A] relative overflow-hidden">
      {/* =====================================================================
       * LEFT SECTION: BRAND IDENTITY & FORENSIC INTELLIGENCE (52% on Desktop)
       * ===================================================================== */}
      <div className="relative w-full lg:w-[52%] xl:w-[55%] flex flex-col justify-between p-6 sm:p-10 lg:p-14 bg-[#101B2D] text-white border-b lg:border-b-0 lg:border-r border-[#1E2F47] overflow-hidden z-10">
        {/* Animated Technical Blueprint Grid Background */}
        <ForensicGridBackground variant="auth" animated={true} />

        {/* Ambient Radial Vignette */}
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_30%_25%,rgba(39,179,163,0.12),transparent_70%)]" />

        {/* Top: Primary Brand Header */}
        <div className="relative z-10 flex items-center justify-between">
          <PhantomLogo variant="full" size="md" theme="dark" showTagline={false} />
          <span className="hidden sm:inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#18263A] border border-[#243752] font-mono text-[10px] text-[#27B3A3] uppercase tracking-wider">
            <span className="w-1.5 h-1.5 rounded-full bg-[#27B3A3]" />
            CRIMINAL FORENSICS
          </span>
        </div>

        {/* Center: Brand Statement & Scientific Biometric Illustration */}
        <div className="relative z-10 my-8 sm:my-10 lg:my-12 space-y-6 max-w-xl">
          <div className="space-y-3">
            <div className="inline-flex items-center gap-2 font-mono text-xs text-[#27B3A3] uppercase tracking-[0.16em]">
              <span className="w-2 h-0.5 bg-[#27B3A3]" />
              Evidence-Grounded Forensic Intelligence
            </div>
            <h1 className="text-2xl sm:text-3xl xl:text-4xl font-bold tracking-tight text-white leading-tight">
              Turning witness descriptions into structured investigative insight.
            </h1>
            <p className="text-sm text-[#B8C7DA] leading-relaxed">
              Phantom AI couples character-aware BiLSTM-CRF sequence extraction
              with conditional generative synthesis (WGAN-GP) to reconstruct facial
              composites under strict human investigator oversight.
            </p>
          </div>

          {/* Scientific Biometric Face Mesh Illustration (Hidden on very small viewports) */}
          <div className="hidden sm:block pt-2">
            <BiometricFacialMeshGraphic />
          </div>
        </div>

        {/* Bottom: Architecture Provenance Badges */}
        <div className="relative z-10 flex flex-wrap items-center gap-3 pt-4 border-t border-[#1E2F47] text-[11px] font-mono text-[#7893B3]">
          <div className="flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-[#27B3A3]" />
            <span>BiLSTM-CRF NLP</span>
          </div>
          <span>•</span>
          <div className="flex items-center gap-1.5">
            <ScanFace className="w-3.5 h-3.5 text-[#27B3A3]" />
            <span>Conditional WGAN-GP</span>
          </div>
          <span>•</span>
          <div className="flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-[#27B3A3]" />
            <span>Traceable GCIT</span>
          </div>
        </div>
      </div>

      {/* =====================================================================
       * RIGHT SECTION: REFINED AUTHENTICATION PANEL (48% on Desktop)
       * ===================================================================== */}
      <div className="relative w-full lg:w-[48%] xl:w-[45%] flex items-center justify-center p-6 sm:p-10 lg:p-14 bg-[#F3F6FA] overflow-y-auto">
        {/* Subtle Workspace Background Grid behind the white card */}
        <ForensicGridBackground variant="workspace-subtle" animated={true} />

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, ease: "easeOut" }}
          className="relative z-10 w-full max-w-md rounded-lg border border-[#DCE4EE] bg-white p-7 sm:p-9 shadow-[0_10px_35px_-5px_rgba(16,27,45,0.1)]"
        >
          {/* Card Header */}
          <div className="mb-6 space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="font-mono text-[10px] font-semibold text-[#5A718E] uppercase tracking-widest">
                INVESTIGATOR GATEWAY
              </span>
              <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-[#27B3A3]/12 border border-[#27B3A3]/30 font-mono text-[10px] text-[#147A6E] font-medium">
                <ShieldCheck className="w-3 h-3 text-[#27B3A3]" />
                <span>FIDO2 / WEBAUTHN</span>
              </div>
            </div>
            <h2 className="text-xl font-bold tracking-tight text-[#18263A]">
              Investigator Sign In
            </h2>
            <p className="text-xs text-[#5A718E]">
              Authenticate with your credentials or platform hardware passkey.
            </p>
          </div>

          {/* Feedback Alerts */}
          {error && (
            <div
              role="alert"
              className="mb-5 flex items-start gap-2.5 rounded-md border border-[#D64550]/40 bg-[#FEF2F2] p-3 text-xs text-[#B4232E]"
            >
              <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5 text-[#D64550]" />
              <div className="leading-relaxed">{error}</div>
            </div>
          )}

          {infoMessage && (
            <div
              role="status"
              className="mb-5 rounded-md border border-[#27B3A3]/40 bg-[#F4FBFA] p-3 text-xs text-[#147A6E]"
            >
              {infoMessage}
            </div>
          )}

          <AnimatePresence mode="wait">
            {/* STAGE 1: CREDENTIALS (ID + PASSCODE) */}
            {stage === "CREDENTIALS" && (
              <motion.form
                key="stage-credentials"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
                onSubmit={handleCredentialSubmit}
                className="space-y-4"
              >
                <Input
                  id="identifier"
                  label="Investigator ID or Agency Email"
                  placeholder="e.g. INV-7741-0001 or investigator@phantomai.gov"
                  value={identifier}
                  onChange={(e) => setIdentifier(e.target.value)}
                  required
                  autoFocus
                />

                <Input
                  id="passcode"
                  label="Investigator Passcode"
                  type="password"
                  placeholder="••••••••••••"
                  value={passcode}
                  onChange={(e) => setPasscode(e.target.value)}
                  required
                />

                <Button
                  type="submit"
                  variant="primary"
                  loading={loading}
                  className="w-full mt-2"
                >
                  <Lock className="h-4 w-4 mr-2" />
                  Verify Passcode
                </Button>

                <div className="flex items-center justify-between pt-2">
                  <button
                    type="button"
                    onClick={() => {
                      setError(null);
                      setStage("RECOVERY");
                    }}
                    className="text-xs text-[#5A718E] hover:text-[#27B3A3] transition-colors cursor-pointer"
                  >
                    Lost passkey? Use recovery code
                  </button>
                </div>
              </motion.form>
            )}

            {/* STAGE 2: PASSKEY MFA ASSERTION */}
            {stage === "PASSKEY_MFA" && (
              <motion.div
                key="stage-passkey-mfa"
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.98 }}
                className="space-y-5"
              >
                <div className="rounded-md border border-[#DCE4EE] bg-[#F8FAFC] p-4 text-center">
                  <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-[#27B3A3]/12 border border-[#27B3A3]/30">
                    <Fingerprint className="h-6 w-6 text-[#27B3A3] animate-pulse" />
                  </div>
                  <h3 className="text-sm font-semibold text-[#18263A]">
                    Hardware Passkey Assertion
                  </h3>
                  <p className="text-xs text-[#5A718E] mt-1">
                    Verifying identity for{" "}
                    <span className="font-mono text-[#18263A] font-semibold">
                      {preAuthData?.user.investigatorId || identifier}
                    </span>
                  </p>
                  <p className="font-mono text-[11px] text-[#7893B3] mt-3">
                    Use Touch ID, Face ID, Windows Hello, or platform security key.
                  </p>
                </div>

                <Button
                  type="button"
                  variant="primary"
                  onClick={() => tryPasskeyAuth()}
                  loading={loading}
                  className="w-full"
                >
                  <Fingerprint className="h-4 w-4 mr-2" />
                  Verify with Device Passkey
                </Button>

                <div className="flex items-center justify-between border-t border-[#DCE4EE] pt-3">
                  <button
                    type="button"
                    onClick={() => {
                      setError(null);
                      setStage("CREDENTIALS");
                    }}
                    className="text-xs text-[#5A718E] hover:text-[#18263A] transition-colors cursor-pointer"
                  >
                    ← Back to Passcode
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setError(null);
                      setStage("RECOVERY");
                    }}
                    className="text-xs text-[#27B3A3] hover:underline cursor-pointer"
                  >
                    Emergency recovery →
                  </button>
                </div>
              </motion.div>
            )}

            {/* STAGE 3: MANDATORY PASSKEY ENROLLMENT */}
            {stage === "ENROLL_PASSKEY" && (
              <motion.div
                key="stage-enroll-passkey"
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.98 }}
                className="space-y-5"
              >
                <div className="rounded-md border border-[#27B3A3]/30 bg-[#F4FBFA] p-4 text-center">
                  <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-[#27B3A3]/20">
                    <KeyRound className="h-6 w-6 text-[#27B3A3]" />
                  </div>
                  <h3 className="text-sm font-semibold text-[#18263A]">
                    Enroll Hardware Passkey
                  </h3>
                  <p className="text-xs text-[#5A718E] mt-1 leading-relaxed">
                    Per forensic protocol, investigator accounts must enroll a platform
                    authenticator before accessing evidence records.
                  </p>
                  <p className="text-[11px] text-[#7893B3] mt-3">
                    Cryptographic keys remain safely stored on your device. No biometric
                    images or templates are ever captured or transmitted.
                  </p>
                </div>

                <Button
                  type="button"
                  variant="primary"
                  onClick={handleEnrollPasskey}
                  loading={loading}
                  className="w-full"
                >
                  <Fingerprint className="h-4 w-4 mr-2" />
                  Register Device Passkey
                </Button>

                <div className="text-center">
                  <button
                    type="button"
                    onClick={() => setStage("CREDENTIALS")}
                    className="text-xs text-[#5A718E] hover:text-[#18263A] transition-colors cursor-pointer"
                  >
                    Cancel and return
                  </button>
                </div>
              </motion.div>
            )}

            {/* STAGE 4: EMERGENCY RECOVERY CODE */}
            {stage === "RECOVERY" && (
              <motion.form
                key="stage-recovery"
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                onSubmit={handleRecoverySubmit}
                className="space-y-4"
              >
                <div className="rounded-md border border-[#DCE4EE] bg-[#F8FAFC] p-3 text-xs text-[#5A718E] leading-relaxed">
                  <span className="font-semibold text-[#18263A]">
                    Forensic Recovery Protocol:{" "}
                  </span>
                  Enter your single-use emergency recovery code. This will allow you to
                  re-enroll a new device passkey.
                </div>

                <Input
                  id="recovery-identifier"
                  label="Investigator ID or Email"
                  value={identifier}
                  onChange={(e) => setIdentifier(e.target.value)}
                  required
                />

                <Input
                  id="recovery-passcode"
                  label="Investigator Passcode"
                  type="password"
                  value={passcode}
                  onChange={(e) => setPasscode(e.target.value)}
                  required
                />

                <Input
                  id="recovery-code"
                  label="Single-Use Recovery Code"
                  placeholder="e.g. A3F9-2B8C"
                  value={recoveryCode}
                  onChange={(e) => setRecoveryCode(e.target.value.toUpperCase())}
                  required
                />

                <Button
                  type="submit"
                  loading={loading}
                  variant="primary"
                  className="w-full"
                >
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Verify Recovery Code
                </Button>

                <div className="text-center pt-2">
                  <button
                    type="button"
                    onClick={() => {
                      setError(null);
                      setStage("CREDENTIALS");
                    }}
                    className="text-xs text-[#5A718E] hover:text-[#18263A] transition-colors cursor-pointer"
                  >
                    ← Back to standard login
                  </button>
                </div>
              </motion.form>
            )}

            {/* STAGE 5: RECOVERY CODES DISPLAY */}
            {stage === "RECOVERY_CODES_DISPLAY" && (
              <motion.div
                key="stage-recovery-display"
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.98 }}
                className="space-y-4"
              >
                <div className="rounded-md border border-[#1D9E75]/40 bg-[#ECFDF5] p-4">
                  <div className="flex items-center gap-2 text-[#136F52] font-semibold text-sm mb-1">
                    <CheckCircle2 className="h-4 w-4 text-[#1D9E75]" />
                    Passkey Registered Successfully
                  </div>
                  <p className="text-xs text-[#3B4F6B] leading-relaxed">
                    Save your emergency recovery codes in a secure location. They will{" "}
                    <span className="text-[#B4232E] font-medium">never be displayed again</span>.
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-2 rounded-md border border-[#DCE4EE] bg-[#F8FAFC] p-3 font-mono text-xs text-[#18263A] text-center">
                  {issuedRecoveryCodes.map((code, idx) => (
                    <div
                      key={idx}
                      className="rounded bg-white py-1.5 border border-[#DCE4EE] shadow-2xs font-semibold"
                    >
                      {code}
                    </div>
                  ))}
                </div>

                <div className="flex gap-2">
                  <Button
                    type="button"
                    variant="secondary"
                    onClick={copyRecoveryCodes}
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
                    Dashboard
                    <ArrowRight className="h-3.5 w-3.5 ml-1" />
                  </Button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Card Footer */}
          <div className="mt-7 flex items-center justify-between border-t border-[#DCE4EE] pt-4 text-xs text-[#5A718E]">
            <span className="font-mono text-[10px] uppercase tracking-wider">
              OFFICIAL USE ONLY
            </span>
            {stage === "CREDENTIALS" && (
              <Link
                to="/register"
                className="text-[#27B3A3] hover:underline font-medium transition-colors"
              >
                Request credentials →
              </Link>
            )}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
