/* The three interactive pieces of the page: the sepia light/dark toggle, the
   copy button on the install line, and the docs tab strip. The theme itself is
   applied by an inline script in <head>, before first paint, so the page never
   flashes the wrong palette; this file only handles clicks. */

(function () {
  "use strict";

  var root = document.documentElement;

  /* ── Theme toggle ───────────────────────────────────────────── */

  var toggle = document.querySelector(".theme-toggle");

  function applyTheme(theme) {
    root.setAttribute("data-theme", theme);
    toggle.textContent = theme === "dark" ? "☀" : "☾";
    try {
      localStorage.setItem("tallydb-theme", theme);
    } catch (e) {
      /* Private browsing, or storage disabled: the toggle still works. */
    }
  }

  applyTheme(root.getAttribute("data-theme") === "dark" ? "dark" : "light");

  toggle.addEventListener("click", function () {
    applyTheme(root.getAttribute("data-theme") === "dark" ? "light" : "dark");
  });

  /* ── Copy the install command ───────────────────────────────── */

  var copy = document.querySelector(".copy-btn");

  copy.addEventListener("click", function () {
    var text = document.getElementById(copy.getAttribute("aria-controls"));
    try {
      navigator.clipboard.writeText(text.textContent);
    } catch (e) {
      /* No clipboard permission: fall through to the label change anyway. */
    }
    copy.textContent = "Copied";
    setTimeout(function () {
      copy.textContent = "Copy";
    }, 1600);
  });

  /* ── Docs tabs ──────────────────────────────────────────────── */

  var tabs = Array.prototype.slice.call(document.querySelectorAll(".tab"));

  tabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      tabs.forEach(function (other) {
        var selected = other === tab;
        other.setAttribute("aria-selected", selected ? "true" : "false");
        document.getElementById(other.getAttribute("aria-controls")).hidden =
          !selected;
      });
    });
  });
})();
