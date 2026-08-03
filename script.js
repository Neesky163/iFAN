const sections = [...document.querySelectorAll(".section-anchor")];
const navLinks = [...document.querySelectorAll(".toc nav a")];
const toc = document.querySelector(".toc");
const navToggle = document.querySelector(".nav-toggle");

const setActiveSection = () => {
  const current = sections.reduce((active, section) => {
    return section.getBoundingClientRect().top <= 180 ? section : active;
  }, sections[0]);

  navLinks.forEach((link) => {
    link.classList.toggle("active", link.getAttribute("href") === `#${current.id}`);
  });
};

window.addEventListener("scroll", setActiveSection, { passive: true });
setActiveSection();

navToggle?.addEventListener("click", () => {
  const open = toc.classList.toggle("open");
  navToggle.setAttribute("aria-expanded", String(open));
});

navLinks.forEach((link) => {
  link.addEventListener("click", () => {
    toc.classList.remove("open");
    navToggle?.setAttribute("aria-expanded", "false");
  });
});

document.addEventListener("click", (event) => {
  if (toc.classList.contains("open") && !toc.contains(event.target) && !navToggle.contains(event.target)) {
    toc.classList.remove("open");
    navToggle.setAttribute("aria-expanded", "false");
  }
});

const copyButton = document.querySelector(".copy-button");
copyButton?.addEventListener("click", async () => {
  const bibtex = document.querySelector("#bibtex").textContent;
  try {
    await navigator.clipboard.writeText(bibtex);
    copyButton.textContent = "Copied";
    window.setTimeout(() => { copyButton.textContent = "Copy"; }, 1600);
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.querySelector("#bibtex"));
    selection.removeAllRanges();
    selection.addRange(range);
    copyButton.textContent = "Selected";
  }
});
