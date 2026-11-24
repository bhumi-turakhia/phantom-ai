import axios from 'axios';

let inMemoryAccessToken: string | null = null;

export const setAccessToken = (token: string | null): void => {
  inMemoryAccessToken = token;
};

export const getAccessToken = (): string | null => inMemoryAccessToken;

function getCookie(name: string): string | null {
  if (typeof document === 'undefined' || !document.cookie) return null;
  const match = document.cookie.match(
    new RegExp('(?:^|; )' + name.replace(/([.$?*|{}()[\]\\/+^])/g, '\\$1') + '=([^;]*)')
  );
  return match ? decodeURIComponent(match[1]) : null;
}

const api = axios.create({
  baseURL:
    (import.meta as any).env?.VITE_API_BASE_URL ||
    'http://localhost:5001/api/v1',
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use((config) => {
  const token =
    inMemoryAccessToken ||
    (typeof window !== 'undefined' ? localStorage.getItem('access_token') : null);
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }

  // Anti-CSRF security headers & double-submit token for cookie-authenticated requests
  config.headers['X-Requested-With'] = 'XMLHttpRequest';

  const csrfToken = getCookie('csrf_token') || getCookie('XSRF-TOKEN');
  if (csrfToken) {
    config.headers['X-CSRF-Token'] = csrfToken;
    config.headers['X-Phantom-CSRF'] = csrfToken;
  } else {
    config.headers['X-Phantom-CSRF'] = '1';
  }

  return config;
});

export default api;
