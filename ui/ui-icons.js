(function () {
  const icons = {
    add: "+", cloud: "☁", devices: "◉", help: "?",
    partly_cloudy_day: "⛅", psychology: "🧠", rainy: "🌧",
    save: "▣", settings: "⚙", sunny: "☀", water_drop: "💧",
    wb_sunny: "☀"
  };

  function replaceIcons(root) {
    if (!(root instanceof Element)) return;
    const nodes = root.matches(".material-symbols-outlined")
      ? [root] : root.querySelectorAll(".material-symbols-outlined");
    nodes.forEach((node) => {
      if (node.querySelector(".tooltiptext")) {
        const text = Array.from(node.childNodes).find(
          (child) => child.nodeType === Node.TEXT_NODE && child.textContent.trim()
        );
        const key = text?.textContent.trim();
        if (key && icons[key]) text.textContent = icons[key];
      } else {
        const key = node.textContent.trim();
        if (icons[key]) node.textContent = icons[key];
      }
      node.classList.add("local-icon");
      node.setAttribute("aria-hidden", "true");
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    replaceIcons(document.body);
    new MutationObserver((records) => records.forEach(
      (record) => record.addedNodes.forEach(replaceIcons)
    )).observe(document.body, { childList: true, subtree: true });
  });
})();
