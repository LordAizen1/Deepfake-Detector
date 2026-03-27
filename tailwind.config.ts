import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        "tertiary": "#ff7b70",
        "primary-dim": "#b5b8c8",
        "secondary": "#6daa8b",
        "surface-dim": "#060e20",
        "on-tertiary": "#530004",
        "on-background": "#dee5ff",
        "on-secondary": "#002617",
        "surface-container-lowest": "#000000",
        "on-secondary-container": "#8ecdac",
        "primary-fixed-dim": "#d0d4e4",
        "on-primary": "#3c404d",
        "error-container": "#7f2927",
        "surface-bright": "#002867",
        "error": "#ee7d77",
        "on-primary-container": "#ccd0e0",
        "background": "#060e20",
        "tertiary-dim": "#ff7166",
        "on-surface-variant": "#91aaeb",
        "on-error": "#490106",
        "surface-variant": "#00225a",
        "surface-tint": "#c2c6d6",
        "error-dim": "#bb5551",
        "on-error-container": "#ff9993",
        "surface-container-low": "#06122d",
        "secondary-container": "#00452d",
        "primary-container": "#424654",
        "outline": "#5b74b1",
        "on-surface": "#dee5ff",
        "primary": "#c2c6d6",
        "surface-container-highest": "#00225a",
        "outline-variant": "#2b4680",
        "inverse-surface": "#faf8ff",
        "surface": "#060e20",
        "surface-container": "#05183c",
        "surface-container-high": "#031d4b",
      },
      fontFamily: {
        headline: ["var(--font-newsreader)", "serif"],
        body: ["var(--font-inter)", "sans-serif"],
        label: ["var(--font-inter)", "sans-serif"],
      },
      borderRadius: {
        DEFAULT: "0.125rem",
        lg: "0.25rem",
        xl: "0.5rem",
        full: "0.75rem",
      },
    },
  },
  plugins: [],
};

export default config;
