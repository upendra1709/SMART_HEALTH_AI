/**
 * Smart Health AI Diagnosis
 * script.js — Shared utilities and API helpers
 */

// ─── API Base URL ──────────────────────────────────────────
const API_BASE = 'http://localhost:5000';

// ─── API Helper ────────────────────────────────────────────
async function apiGet(endpoint) {
  const res = await fetch(`${API_BASE}${endpoint}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function apiPost(endpoint, body) {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// ─── Format symptom string ─────────────────────────────────
function formatSymptomLabel(s) {
  return s.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// ─── Nav scroll shadow ─────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const nav = document.querySelector('.nav');
  const themeToggle = document.getElementById('themeToggle');
  const themeIcon = themeToggle ? themeToggle.querySelector('.theme-toggle-icon') : null;
  const themeText = themeToggle ? themeToggle.querySelector('.theme-toggle-text') : null;

  // THEME TOGGLE: initialize and persist theme choice
  const applyTheme = (theme) => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    if (themeIcon) themeIcon.textContent = theme === 'light' ? '☀️' : '🌙';
    if (themeText) themeText.textContent = theme === 'light' ? 'Light' : 'Dark';
  };

  const storedTheme = localStorage.getItem('theme');
  const prefersLight = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
  applyTheme(storedTheme || (prefersLight ? 'light' : 'dark'));

  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') || 'dark';
      applyTheme(current === 'light' ? 'dark' : 'light');
    });
  }

  if (nav) {
    const updateNavBg = () => {
      nav.style.background = window.scrollY > 40
        ? 'var(--nav-bg-scrolled)'
        : 'var(--nav-bg)';
    };
    updateNavBg();
    window.addEventListener('scroll', updateNavBg);
  }
});
