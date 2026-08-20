(function () {
  function initStudyFilters(section) {
    var chips = Array.prototype.slice.call(section.querySelectorAll("[data-study-filter]"));
    var cards = Array.prototype.slice.call(document.querySelectorAll("[data-study-card]"));
    var status = section.querySelector("[data-study-filter-status]");

    if (!chips.length || !cards.length) {
      return;
    }

    function update(filterValue, filterLabel) {
      var visibleCount = 0;

      cards.forEach(function (card) {
        var categories = (card.getAttribute("data-study-categories") || "").split("|").filter(Boolean);
        var isVisible = filterValue === "all" || categories.indexOf(filterValue) !== -1;
        card.hidden = !isVisible;
        if (isVisible) {
          visibleCount += 1;
        }
      });

      chips.forEach(function (chip) {
        var isActive = chip.getAttribute("data-study-filter") === filterValue;
        chip.classList.toggle("is-active", isActive);
        chip.setAttribute("aria-pressed", isActive ? "true" : "false");
      });

      if (status) {
        status.textContent = filterValue === "all"
          ? "All " + visibleCount + " posts"
          : filterLabel + " " + visibleCount + " posts";
      }
    }

    chips.forEach(function (chip) {
      chip.addEventListener("click", function () {
        var filterValue = chip.getAttribute("data-study-filter");
        var labelNode = chip.childNodes[0];
        var filterLabel = labelNode ? labelNode.textContent.trim() : filterValue;
        update(filterValue, filterLabel);
      });
    });
  }

  document.querySelectorAll("[data-study-filters]").forEach(initStudyFilters);
}());
