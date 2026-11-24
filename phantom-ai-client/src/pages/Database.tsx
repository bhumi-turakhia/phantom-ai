import { Database as DatabaseIcon, Search } from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "../components/ui/Card";
import { Input } from "../components/ui/Input";
import { StatCard } from "../components/ui/StatCard";
import { EmptyState } from "../components/ui/EmptyState";
import { PageTransition } from "../components/layout/PageTransition";

/**
 * INTEGRATION: this page is a search shell over your existing database
 * query endpoint. Wire the Input's onChange to your query hook and
 * render results in place of the EmptyState below.
 */
export default function DatabasePage() {
  return (
    <PageTransition>
      <div className="space-y-6">
        <div>
          <h1 className="text-lg font-semibold text-ink-primary">Database</h1>
          <p className="label-technical text-ink-disabled mt-1">Criminal database reference search</p>
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <StatCard label="Total Records" value={58302} icon={<DatabaseIcon className="h-4 w-4" />} />
          <StatCard label="Indexed This Week" value={412} />
          <StatCard label="Sync Status" value={100} suffix="%" />
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Search records</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="relative max-w-lg">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-disabled" />
              <Input id="db-search" placeholder="Search by attribute, case ID, or reference number…" className="pl-9" />
            </div>
            <EmptyState
              title="No query submitted"
              description="Search by physical attributes, case reference, or record ID to cross-reference the database."
            />
          </CardContent>
        </Card>
      </div>
    </PageTransition>
  );
}
