module.exports = {
  content: ["_site/**/*.html", "_site/**/*.js"],
  css: ["_site/assets/css/*.css"],
  output: "_site/assets/css/",
  skippedContentGlobs: ["_site/assets/**/*.html"],
  // Keep the engraving background plates and theme-toggle rules: their
  // selectors (body::before/::after, html[data-theme=...]) are not visible
  // to the content scan.
  safelist: {
    standard: ["body", "html"],
    greedy: [/data-theme/, /::before/, /::after/],
  },
};
