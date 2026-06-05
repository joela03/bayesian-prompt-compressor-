/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // System palette referenced throughout the site.
        ours: '#2d5a1f',
        lingua: '#d97706',
        fail: '#b91c1c',
        ink: '#0f172a',
        muted: '#64748b',
        line: '#e2e8f0',
        canvas: '#fafaf7',
      },
      fontFamily: {
        // System stack only — no external font loads.
        sans: [
          '-apple-system',
          'BlinkMacSystemFont',
          'Segoe UI',
          'Roboto',
          'Helvetica Neue',
          'Arial',
          'sans-serif',
        ],
        mono: [
          'ui-monospace',
          'SFMono-Regular',
          'Menlo',
          'Monaco',
          'Consolas',
          'monospace',
        ],
      },
      boxShadow: {
        card: '0 1px 2px 0 rgb(15 23 42 / 0.04), 0 1px 1px 0 rgb(15 23 42 / 0.03)',
      },
    },
  },
  plugins: [],
};
