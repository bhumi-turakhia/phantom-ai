import { useState, type ReactNode } from "react";
import { useLocation } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";
import {
  ForensicGridBackground,
  type ForensicGridVariant,
} from "../ui/ForensicGridBackground";

/**
 * Resolves the page-specific animated grid treatment per Stage 1.5 rules:
 * - /dashboard: "workspace" (standard animated blueprint + subtle teal ambient illumination)
 * - /generate: "studio" (restrained steel-blue lines & subtle teal highlights; face composites unobstructed)
 * - /cases, /reports, /database, /profile: "workspace-subtle" (extra-subtle animated grid behind solid white panels)
 */
function resolveWorkspaceGridVariant(pathname: string): ForensicGridVariant {
  if (pathname.startsWith("/dashboard")) return "workspace";
  if (pathname.startsWith("/generate")) return "studio";
  return "workspace-subtle";
}

/**
 * Option C — Hybrid Forensic Workspace Shell:
 * - Dark Midnight Navy (#101B2D) Sidebar with static Steel-Blue grid
 * - Deep Slate (#18263A) TopBar header
 * - Light Grey (#F3F6FA) main workspace surface with continuous animated blueprint grid
 *   and crisp solid White (#FFFFFF) panels above the grid
 */
export function AppLayout({ children }: { children: ReactNode }) {
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const location = useLocation();
  const gridVariant = resolveWorkspaceGridVariant(location.pathname);

  return (
    <div className="flex h-screen overflow-hidden bg-[#F3F6FA] text-[#18263A]">
      {/* Desktop Sidebar */}
      <div className="hidden lg:flex">
        <Sidebar />
      </div>

      {/* Mobile & Tablet Responsive Drawer Sidebar */}
      {mobileNavOpen && (
        <div className="fixed inset-0 z-50 flex lg:hidden">
          <div
            className="fixed inset-0 bg-[#101B2D]/70 backdrop-blur-xs"
            onClick={() => setMobileNavOpen(false)}
          />
          <div className="relative z-10 flex">
            <Sidebar onNavigate={() => setMobileNavOpen(false)} />
          </div>
        </div>
      )}

      {/* Main Content Area */}
      <div className="relative flex flex-1 flex-col overflow-hidden bg-[#F3F6FA]">
        <TopBar onToggleMobileNav={() => setMobileNavOpen((prev) => !prev)} />
        <div className="relative flex-1 overflow-hidden">
          {/* Stage 1.5: Continuous Animated Forensic Blueprint Grid (strictly behind workspace content) */}
          <ForensicGridBackground variant={gridVariant} animated={true} />

          <main className="relative z-10 h-full overflow-y-auto p-4 sm:p-6">
            <div className="mx-auto max-w-[1600px]">{children}</div>
          </main>
        </div>
      </div>
    </div>
  );
}
