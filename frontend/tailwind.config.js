/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./templates/**/*.html",
    "./static/**/*.js",
  ],
  darkMode: 'class', // enabling class-based dark mode
  theme: {
    extend: {
      colors: {
        'light-bg': '#fdfbf7',
        'light-text': '#1a1a1a',
        'dark-bg': '#212121', // typical chatgpt dark background
        'dark-text': '#ececec',
        'sidebar-dark': '#171717',
        'sidebar-light': '#f9f9f9',
      }
    },
  },
  plugins: [],
}
