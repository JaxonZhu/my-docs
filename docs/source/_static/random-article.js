(() => {
  "use strict";

  const link = document.querySelector(".random-article a");
  const data = document.currentScript?.dataset.articles;
  if (!link || !data) return;

  let articles;
  try {
    articles = JSON.parse(data);
  } catch {
    return;
  }
  if (!Array.isArray(articles) || articles.length === 0) return;

  let previous = null;
  const randomize = () => {
    const choices = articles.length > 1
      ? articles.filter((article) => article !== previous)
      : articles;
    previous = choices[Math.floor(Math.random() * choices.length)];
    link.setAttribute("href", previous);
  };

  // Keep a real link so keyboard navigation and opening a new tab still work.
  randomize();
  link.textContent = "随机逛一篇 →";
  link.classList.add("btn", "btn-neutral");
  link.addEventListener("click", randomize);
  link.addEventListener("auxclick", (event) => {
    if (event.button === 1) randomize();
  });
})();
