(() => {
  const copyButtons = document.querySelectorAll(".copy-code-button");
  if (!copyButtons.length) {
    return;
  }

  const copyText = async (text) => {
    if ("clipboard" in navigator && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(text);
        return true;
      } catch (error) {
        console.warn("navigator.clipboard failed, falling back to execCommand", error);
      }
    }

    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "absolute";
    textarea.style.left = "-9999px";
    const yPosition = window.pageYOffset || document.documentElement.scrollTop;
    textarea.style.top = yPosition + "px";
    document.body.appendChild(textarea);

    let success = false;
    try {
      textarea.select();
      success = document.execCommand("copy");
    } catch (error) {
      success = false;
    }

    textarea.remove();
    return success;
  };

  const findCodeElement = (button) => {
    const header = button.closest(".code-header");
    if (!header) {
      return null;
    }
    let sibling = header.nextElementSibling;
    while (sibling && !sibling.classList.contains("highlighter-rouge")) {
      sibling = sibling.nextElementSibling;
    }
    if (!sibling) {
      return null;
    }
    return sibling.querySelector("pre code, code") || sibling;
  };

  copyButtons.forEach((button) => {
    const codeElement = findCodeElement(button);
    if (!codeElement) {
      console.warn("No code block found for copy button", button);
      return;
    }

    const img = button.querySelector("img");
    const defaultIcon = img ? img.src : null;
    const copiedIcon = img ? img.getAttribute("data-success-icon") || "/assets/img/check.png" : null;
    let timeoutId = null;

    button.addEventListener("click", async () => {
      const copied = await copyText(codeElement.innerText);
      if (!copied) {
        console.warn("Failed to copy code to clipboard");
        return;
      }

      if (img && copiedIcon && defaultIcon) {
        if (timeoutId) {
          clearTimeout(timeoutId);
        }
        img.src = copiedIcon;
        timeoutId = setTimeout(() => {
          img.src = defaultIcon;
          timeoutId = null;
        }, 2000);
      }
    });
  });
})();
