import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";

import Login from "./pages/Login";
import Register from "./pages/Register";
import Dashboard from "./pages/Dashboard";
import Cases from "./pages/Cases";
import Database from "./pages/Database";
import Reports from "./pages/Reports";
import Profile from "./pages/Profile";
import GenerateComposite from "./pages/GenerateComposite";

import { AppLayout } from "./components/layout/AppLayout";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>

        <Route path="/" element={<Login />} />
        <Route path="/register" element={<Register />} />

        <Route
          path="/dashboard"
          element={
            <AppLayout>
              <Dashboard />
            </AppLayout>
          }
        />

        <Route
          path="/cases"
          element={
            <AppLayout>
              <Cases />
            </AppLayout>
          }
        />

        <Route
          path="/generate"
          element={
            <AppLayout>
              <GenerateComposite />
            </AppLayout>
          }
        />

        <Route
          path="/database"
          element={
            <AppLayout>
              <Database />
            </AppLayout>
          }
        />

        <Route
          path="/reports"
          element={
            <AppLayout>
              <Reports />
            </AppLayout>
          }
        />

        <Route
          path="/profile"
          element={
            <AppLayout>
              <Profile />
            </AppLayout>
          }
        />

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}