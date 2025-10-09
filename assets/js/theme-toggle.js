(function() {
  function getStoredTheme() {
    try {
      return localStorage.getItem('site-theme');
    } catch (e) {
      return null;
    }
  }

  function storeTheme(theme) {
    try {
      localStorage.setItem('site-theme', theme);
    } catch (e) {
      /* ignore storage failures */
    }
  }

  function systemPrefersDark() {
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  }

  function applyTheme(themeToggleButton, theme) {
    var lightStylesheet = document.getElementById('theme-light');
    var darkStylesheet = document.getElementById('theme-dark');
    if (!lightStylesheet || !darkStylesheet) {
      return;
    }

    if (theme === 'dark') {
      lightStylesheet.disabled = true;
      darkStylesheet.disabled = false;
      document.documentElement.setAttribute('data-theme', 'dark');
      themeToggleButton.classList.add('theme-toggle--dark');
      themeToggleButton.innerHTML = '<i class=\"fas fa-sun\"></i>';
      themeToggleButton.setAttribute('aria-label', 'Switch to light mode');
    } else {
      lightStylesheet.disabled = false;
      darkStylesheet.disabled = true;
      document.documentElement.setAttribute('data-theme', 'light');
      themeToggleButton.classList.remove('theme-toggle--dark');
      themeToggleButton.innerHTML = '<i class=\"fas fa-moon\"></i>';
      themeToggleButton.setAttribute('aria-label', 'Switch to dark mode');
    }
  }

  document.addEventListener('DOMContentLoaded', function() {
    var toggleButton = document.querySelector('.theme-toggle');
    if (!toggleButton) {
      return;
    }

    var storedTheme = getStoredTheme();
    var initialTheme = storedTheme || (systemPrefersDark() ? 'dark' : 'light');
    applyTheme(toggleButton, initialTheme);

    toggleButton.addEventListener('click', function() {
      var currentTheme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
      var nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
      applyTheme(toggleButton, nextTheme);
      storeTheme(nextTheme);
    });
  });
})();
