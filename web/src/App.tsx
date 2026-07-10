import { Navigate, Route, HashRouter, Routes } from "react-router-dom";
import { getToken } from "./api/client";
import HomePage from "./pages/HomePage";
import LoginPage from "./pages/LoginPage";
import MeetingDetailPage from "./pages/MeetingDetailPage";

function RequireAuth({ children }: { children: JSX.Element }) {
  return getToken() ? children : <Navigate to="/login" replace />;
}

export default function App() {
  return (
    <HashRouter>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route
          path="/"
          element={
            <RequireAuth>
              <HomePage />
            </RequireAuth>
          }
        />
        <Route
          path="/meetings/:id"
          element={
            <RequireAuth>
              <MeetingDetailPage />
            </RequireAuth>
          }
        />
      </Routes>
    </HashRouter>
  );
}
