import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Menu, ShieldCheck, Clock, UserCheck, ChevronRight } from "lucide-react";
import { getCurrentUser } from "../../services/authService";

interface TopBarProps {
  onMenuToggle?: () => void;
  onToggleMobileNav?: () => void;
}

const ROUTE_META: Record<string, { title: string; subtitle: string; code: string }> = {
  "/dashboard": {
    title: "Command Center",
    subtitle: "Forensic Operations & Telemetry Overview",
    code: "MOD-OPS-01",
  },
  "/cases": {
    title: "Case Registry & Collaboration",
    subtitle: "Active Investigations, Evidence Chain & Timeline",
    code: "MOD-CAS-02",
  },
  "/generate": {
    title: "Composite Synthesis Studio",
    subtitle: "NLP Extraction, Traceability & WGAN-GP Conditioning",
    code: "MOD-GEN-03",
  },
  "/database": {
    title: "Reference Identity Archive",
    subtitle: "CelebA Identity-Disjoint Reference Index",
    code: "MOD-ARC-04",
  },
  "/reports": {
    title: "Forensic Dossier & Audit Exports",
    subtitle: "Evidence-Grounded PDF Dossiers & Provenance Verification",
    code: "MOD-DOS-05",
  },
  "/profile": {
    title: "Investigator Credentials & Security",
    subtitle: "WebAuthn Passkeys, Recovery Keys & Clearance Status",
    code: "MOD-SEC-06",
  },
};

export const TopBar: React.FC<TopBarProps> = ({ onMenuToggle, onToggleMobileNav }) => {
  const handleMenuToggle = onMenuToggle || onToggleMobileNav;
  const location = useLocation();
  const navigate = useNavigate();
  const [time, setTime] = useState(new Date());
  const [user, setUser] = useState<{
    name?: string;
    role?: string;
    investigatorId?: string;
    badgeNumber?: string;
  } | null>(null);

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const refreshUser = async () => {
      try {
        const raw = localStorage.getItem("user");
        if (raw) {
          setUser(JSON.parse(raw));
        } else if (localStorage.getItem("token")) {
          const res = await getCurrentUser();
          if (res) {
            setUser(res);
            localStorage.setItem("user", JSON.stringify(res));
          }
        } else {
          setUser(null);
        }
      } catch {
        setUser(null);
      }
    };

    refreshUser();

    const handleUserUpdate = (e: any) => {
      if (e.detail) {
        setUser(e.detail);
      } else {
        refreshUser();
      }
    };

    window.addEventListener("phantom:user_updated", handleUserUpdate);
    window.addEventListener("storage", refreshUser);

    return () => {
      window.removeEventListener("phantom:user_updated", handleUserUpdate);
      window.removeEventListener("storage", refreshUser);
    };
  }, [location.pathname]);

  const currentMeta = ROUTE_META[location.pathname] || {
    title: "Forensic Workspace",
    subtitle: "Phantom AI Forensic Intelligence Platform",
    code: "MOD-SYS-00",
  };

  const roleLabel = (user?.role || "INVESTIGATOR").toUpperCase();
  const investigatorBadge = user?.investigatorId || user?.badgeNumber || "INV-ACTIVE";

  return (
    <header className="h-16 bg-[#18263A] text-white border-b border-[#2A3B54] px-4 lg:px-8 flex items-center justify-between sticky top-0 z-30 shadow-xs">
      {/* Left: Mobile Menu Trigger + Breadcrumb / Current Module */}
      <div className="flex items-center gap-3 min-w-0">
        <button
          type="button"
          onClick={handleMenuToggle}
          aria-label="Open navigation menu"
          className="lg:hidden p-2 rounded-md text-[#7893B3] hover:text-white hover:bg-[#101B2D]/60 transition-colors cursor-pointer"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-2.5 min-w-0">
          <span className="hidden sm:inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono font-semibold tracking-wider uppercase bg-[#101B2D] text-[#27B3A3] border border-[#27B3A3]/30">
            {currentMeta.code}
          </span>
          <ChevronRight className="hidden sm:block w-3.5 h-3.5 text-[#7893B3] shrink-0" />
          <div className="truncate">
            <div className="text-sm font-semibold text-white tracking-tight truncate">
              {currentMeta.title}
            </div>
            <div className="hidden md:block text-[11px] text-[#7893B3] truncate">
              {currentMeta.subtitle}
            </div>
          </div>
        </div>
      </div>

      {/* Right: Telemetry, Clearance, Clock & Investigator Pill */}
      <div className="flex items-center gap-3 sm:gap-5">
        {/* Traceability Status Pill */}
        <div className="hidden xl:flex items-center gap-2 px-3 py-1 rounded-md bg-[#101B2D]/80 border border-[#2A3B54]">
          <ShieldCheck className="w-3.5 h-3.5 text-[#27B3A3]" />
          <span className="font-mono text-[10px] text-[#B8C7DA] uppercase tracking-wider">
            GCIT TRACEABILITY: <strong className="text-[#27B3A3]">ENFORCED</strong>
          </span>
        </div>

        {/* System UTC / Local Time */}
        <div className="hidden sm:flex items-center gap-2 px-2.5 py-1 rounded bg-[#101B2D]/60 border border-[#2A3B54]/80 font-mono text-[11px] text-[#B8C7DA]">
          <Clock className="w-3.5 h-3.5 text-[#7893B3]" />
          <span>{time.toLocaleTimeString([], { hour12: false })}</span>
        </div>

        {/* Investigator Identity Button */}
        <button
          type="button"
          onClick={() => navigate("/profile")}
          className="flex items-center gap-2.5 pl-2.5 pr-3 py-1.5 rounded-md bg-[#101B2D] hover:bg-[#101B2D]/80 border border-[#2A3B54] hover:border-[#27B3A3]/50 transition-all cursor-pointer"
          title="View Investigator Credentials & Security"
        >
          <div className="w-6 h-6 rounded bg-[#27B3A3]/15 border border-[#27B3A3]/40 flex items-center justify-center text-[#27B3A3]">
            <UserCheck className="w-3.5 h-3.5" />
          </div>
          <div className="text-left hidden md:block">
            <div className="text-xs font-medium text-white leading-none">
              {user?.name || "Forensic Investigator"}
            </div>
            <div className="font-mono text-[10px] text-[#27B3A3] tracking-wider uppercase mt-0.5">
              {roleLabel} · {investigatorBadge}
            </div>
          </div>
        </button>
      </div>
    </header>
  );
};
