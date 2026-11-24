import { NavLink, useNavigate } from "react-router-dom";
import {
  LayoutDashboard,
  FolderSearch,
  ScanFace,
  Database,
  FileBarChart,
  UserCheck,
  LogOut,
} from "lucide-react";
import { FingerprintMotif } from "../ui/FingerprintMotif";
import { ForensicGridBackground } from "../ui/ForensicGridBackground";
import { PhantomLogo } from "../ui/PhantomLogo";
import { cn } from "../../lib/utils";
import { logout } from "../../services/authService";

const NAV_ITEMS = [
  { to: "/dashboard", label: "Dashboard", code: "OPS-01", icon: LayoutDashboard },
  { to: "/cases", label: "Cases", code: "CAS-02", icon: FolderSearch },
  { to: "/generate", label: "Generate", code: "SYN-03", icon: ScanFace },
  { to: "/database", label: "Database", code: "REF-04", icon: Database },
  { to: "/reports", label: "Reports", code: "DOS-05", icon: FileBarChart },
];

export function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const navigate = useNavigate();

  async function handleLogout() {
    try {
      await logout();
    } catch {
      // Proceed to login screen even if session revocation fails offline
    }
    navigate("/");
  }

  return (
    <aside
      className="relative flex h-screen w-64 shrink-0 flex-col overflow-hidden border-r border-[#1E2F47] bg-[#101B2D] text-[#F3F6FA] select-none"
      aria-label="Primary Forensic Navigation"
    >
      {/* Stage 1.5: Static Steel-Blue (#7893B3) Blueprint Grid on Midnight Navy (#101B2D) */}
      <ForensicGridBackground variant="sidebar-static" animated={false} />

      {/* Brand Identity Header */}
      <div className="relative z-10 flex items-center justify-between border-b border-[#1E2F47] px-5 py-4 overflow-hidden">
        <FingerprintMotif
          className="absolute -top-6 -left-6 h-24 w-24"
          strokeColor="#7893B3"
          opacity={0.08}
        />
        <PhantomLogo variant="full" size="sm" theme="dark" showTagline={true} />
      </div>

      {/* Section Label */}
      <div className="relative z-10 px-5 pt-4 pb-1.5">
        <span className="label-technical text-[10px] text-[#7893B3]">
          Investigation Modules
        </span>
      </div>

      {/* Primary Navigation */}
      <nav className="relative z-10 flex-1 space-y-1 px-3 py-1.5 overflow-y-auto">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            onClick={onNavigate}
            className={({ isActive }) =>
              cn(
                "group relative flex items-center justify-between rounded-md px-3 py-2.5 text-sm font-medium transition-colors duration-150",
                isActive
                  ? "bg-[#18263A] text-white border border-[#27B3A3]/30 shadow-xs"
                  : "text-[#B8C7DA] hover:bg-[#18263A]/60 hover:text-white border border-transparent"
              )
            }
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <span className="absolute left-0 top-1.5 bottom-1.5 w-1 rounded-r bg-[#27B3A3]" />
                )}
                <div className="flex items-center gap-3">
                  <item.icon
                    className={cn(
                      "h-4 w-4 transition-colors",
                      isActive
                        ? "text-[#27B3A3]"
                        : "text-[#7893B3] group-hover:text-[#B8C7DA]"
                    )}
                    strokeWidth={1.75}
                  />
                  <span>{item.label}</span>
                </div>
                <span
                  className={cn(
                    "font-mono text-[10px] transition-colors",
                    isActive ? "text-[#27B3A3]" : "text-[#7893B3]/70"
                  )}
                >
                  {item.code}
                </span>
              </>
            )}
          </NavLink>
        ))}

        <div className="pt-4 pb-1.5 px-2">
          <span className="label-technical text-[10px] text-[#7893B3]">
            Credential & Session
          </span>
        </div>

        <NavLink
          to="/profile"
          onClick={onNavigate}
          className={({ isActive }) =>
            cn(
              "group relative flex items-center justify-between rounded-md px-3 py-2 text-xs font-medium transition-colors",
              isActive
                ? "bg-[#18263A] text-white border border-[#27B3A3]/30"
                : "text-[#B8C7DA] hover:bg-[#18263A]/60 hover:text-white border border-transparent"
            )
          }
        >
          <div className="flex items-center gap-3">
            <UserCheck className="h-4 w-4 text-[#7893B3]" strokeWidth={1.75} />
            <span>Investigator Profile</span>
          </div>
          <span className="font-mono text-[10px] text-[#7893B3]/70">ID-CFG</span>
        </NavLink>
      </nav>

      {/* Telemetry & Session Footer */}
      <div className="relative z-10 border-t border-[#1E2F47] bg-[#0C1524]/95 px-4 py-3.5 space-y-2.5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 label-technical text-[10px] text-[#7893B3]">
            <span className="h-2 w-2 rounded-full bg-[#27B3A3]" />
            <span>Secure Channel Active</span>
          </div>
          <span className="font-mono text-[10px] text-[#7893B3]">FIDO2</span>
        </div>

        <button
          type="button"
          onClick={handleLogout}
          className="flex w-full items-center justify-center gap-2 rounded border border-[#243752] bg-[#18263A] px-3 py-1.5 text-xs font-medium text-[#B8C7DA] hover:border-[#D64550]/50 hover:text-white transition-colors cursor-pointer"
        >
          <LogOut className="h-3.5 w-3.5 text-[#7893B3]" />
          <span>Terminate Session</span>
        </button>
      </div>
    </aside>
  );
}
