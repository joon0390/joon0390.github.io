(function () {
  function byPriority(name) {
    var order = [
      "Bayesian",
      "Statistics",
      "Machine Learning",
      "Deep Learning",
      "Graph",
      "Applied Topics",
      "Time Series",
      "Computation",
      "Optimization"
    ];
    var index = order.indexOf(name);
    return index === -1 ? order.length + 1 : index;
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function intersect(left, right) {
    var rightSet = new Set(right);
    return left.filter(function (item) { return rightSet.has(item); });
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function initMap(section) {
    var dataElement = section.querySelector("[data-category-map-data]");
    var canvas = section.querySelector("[data-category-map-canvas]");
    var svg = section.querySelector("[data-category-map-links]");
    var nodesLayer = section.querySelector("[data-category-map-nodes]");
    var panel = section.querySelector("[data-category-map-panel]");

    if (!dataElement || !canvas || !svg || !nodesLayer || !panel) {
      return;
    }

    var data = JSON.parse(dataElement.textContent);
    var palette = ["#94623B", "#6B7A4B", "#4F6F69", "#A0625B", "#6E5B8C", "#4A6A8A", "#8A744A", "#7C5A6A", "#5A6B7C"];
    var state = { categoryId: null, postId: null };
    var categoryButtons = new Map();
    var postButtons = new Map();

    data.categories = data.categories
      .sort(function (a, b) {
        return byPriority(a.name) - byPriority(b.name) || b.count - a.count || a.name.localeCompare(b.name);
      })
      .map(function (category, index) {
        category.color = palette[index % palette.length];
        return category;
      });

    var categoryById = new Map(data.categories.map(function (category) {
      return [category.id, category];
    }));

    data.posts = data.posts
      .map(function (post) {
        post.tags = (post.tags || []).filter(Boolean);
        post.categoryIds = (post.categories || [])
          .map(function (name) { return String(name).toLowerCase().replace(/[^a-z0-9-]+/g, "-").replace(/^-+|-+$/g, ""); })
          .filter(function (id) { return categoryById.has(id); });
        return post;
      })
      .filter(function (post) { return post.categoryIds.length > 0; });

    var tagFrequency = {};
    data.posts.forEach(function (post) {
      post.tags.forEach(function (tag) {
        tagFrequency[tag] = (tagFrequency[tag] || 0) + 1;
      });
    });

    var categoryPosts = new Map();
    data.categories.forEach(function (category) {
      categoryPosts.set(category.id, []);
    });

    data.posts.forEach(function (post) {
      post.categoryIds.forEach(function (categoryId) {
        if (categoryPosts.has(categoryId)) {
          categoryPosts.get(categoryId).push(post);
        }
      });
    });

    var tagEdges = [];
    for (var i = 0; i < data.posts.length; i += 1) {
      for (var j = i + 1; j < data.posts.length; j += 1) {
        var sharedTags = intersect(data.posts[i].tags, data.posts[j].tags);
        if (!sharedTags.length) {
          continue;
        }

        var score = sharedTags.reduce(function (sum, tag) {
          return sum + 1 / Math.max(1, tagFrequency[tag]);
        }, 0) + sharedTags.length * 0.8;

        tagEdges.push({
          left: data.posts[i].id,
          right: data.posts[j].id,
          sharedTags: sharedTags,
          score: score
        });
      }
    }

    tagEdges.sort(function (left, right) {
      return right.score - left.score || right.sharedTags.length - left.sharedTags.length;
    });
    tagEdges = tagEdges.slice(0, Math.min(26, tagEdges.length));

    function renderPanelDefault() {
      panel.innerHTML = [
        '<p class="category-mindmap__eyebrow">Category Atlas</p>',
        '<h3 class="category-mindmap__panel-title">Category Hubs and Post Nodes</h3>',
        '<p class="category-mindmap__panel-text">Lines indicate category membership and shared tags.</p>',
        '<ul class="category-mindmap__legend">',
        '<li><span class="category-mindmap__legend-dot category-mindmap__legend-dot--hub"></span> category hub</li>',
        '<li><span class="category-mindmap__legend-dot category-mindmap__legend-dot--post"></span> post node</li>',
        '<li><span class="category-mindmap__legend-line"></span> shared tag bridge</li>',
        '</ul>'
      ].join("");
    }

    function renderCategoryPanel(category) {
      var posts = categoryPosts.get(category.id) || [];
      var tagCounts = {};

      posts.forEach(function (post) {
        post.tags.forEach(function (tag) {
          tagCounts[tag] = (tagCounts[tag] || 0) + 1;
        });
      });

      var topTags = Object.keys(tagCounts)
        .sort(function (left, right) { return tagCounts[right] - tagCounts[left] || left.localeCompare(right); })
        .slice(0, 5);
      var postSummary = posts.length === 1
        ? "1 post is connected to this hub."
        : posts.length + " posts are connected to this hub.";
      panel.innerHTML = [
        '<p class="category-mindmap__eyebrow">Category</p>',
        '<h3 class="category-mindmap__panel-title">' + escapeHtml(category.name) + '</h3>',
        '<p class="category-mindmap__panel-text">' + postSummary + '</p>',
        topTags.length ? '<div class="category-mindmap__chips">' + topTags.map(function (tag) {
          return '<span class="category-mindmap__chip">#' + escapeHtml(tag) + '</span>';
        }).join("") + '</div>' : '',
        '<a class="category-mindmap__jump" href="#' + escapeHtml(category.id) + '">View archive</a>'
      ].join("");
    }

    function renderPostPanel(post) {
      var relatedCategories = post.categoryIds.map(function (categoryId) {
        return categoryById.get(categoryId);
      }).filter(Boolean);
      var relatedEdges = tagEdges.filter(function (edge) {
        return edge.left === post.id || edge.right === post.id;
      }).slice(0, 4);

      panel.innerHTML = [
        '<p class="category-mindmap__eyebrow">Post</p>',
        '<h3 class="category-mindmap__panel-title">' + escapeHtml(post.title) + '</h3>',
        '<p class="category-mindmap__panel-text">' + escapeHtml(post.excerpt || "No excerpt available.") + '</p>',
        relatedCategories.length ? '<div class="category-mindmap__chips">' + relatedCategories.map(function (category) {
          return '<span class="category-mindmap__chip" style="--chip-color:' + escapeHtml(category.color) + ';">' + escapeHtml(category.name) + '</span>';
        }).join("") + '</div>' : '',
        post.tags.length ? '<div class="category-mindmap__chips">' + post.tags.slice(0, 5).map(function (tag) {
          return '<span class="category-mindmap__chip">#' + escapeHtml(tag) + '</span>';
        }).join("") + '</div>' : '',
        relatedEdges.length ? '<div class="category-mindmap__list">' + relatedEdges.map(function (edge) {
          var title = edge.left === post.id
            ? data.posts.find(function (item) { return item.id === edge.right; }).title
            : data.posts.find(function (item) { return item.id === edge.left; }).title;
          return '<div class="category-mindmap__list-item">' + escapeHtml(title) + '<span>' + edge.sharedTags.map(function (tag) { return '#' + tag; }).join(', ') + '</span></div>';
        }).join("") + '</div>' : '',
        '<a class="category-mindmap__jump" href="' + escapeHtml(post.url) + '">Read post</a>'
      ].join("");
    }

    function updateSelection() {
      var activeCategoryId = state.categoryId;
      var activePostId = state.postId;

      categoryButtons.forEach(function (button, categoryId) {
        var isActive = activeCategoryId === categoryId;
        var isRelated = activePostId
          ? data.posts.find(function (post) { return post.id === activePostId; }).categoryIds.indexOf(categoryId) !== -1
          : !activeCategoryId || isActive;

        button.classList.toggle("is-active", isActive || (activePostId && isRelated));
        button.classList.toggle("is-dimmed", !isRelated);
      });

      postButtons.forEach(function (button, postId) {
        var post = data.posts.find(function (item) { return item.id === postId; });
        var isActive = activePostId === postId;
        var isInCategory = !activeCategoryId || post.categoryIds.indexOf(activeCategoryId) !== -1;
        var isRelatedToPost = !activePostId || postId === activePostId || tagEdges.some(function (edge) {
          return (edge.left === activePostId && edge.right === postId) || (edge.right === activePostId && edge.left === postId);
        });

        button.classList.toggle("is-active", isActive);
        button.classList.toggle("is-dimmed", !(isInCategory && isRelatedToPost));
      });

      svg.querySelectorAll("[data-edge-type='category']").forEach(function (line) {
        var categoryId = line.getAttribute("data-category-id");
        var postId = line.getAttribute("data-post-id");
        var related = true;

        if (activeCategoryId) {
          related = categoryId === activeCategoryId;
        }
        if (activePostId) {
          related = related && postId === activePostId;
        }

        line.classList.toggle("is-active", related && (activeCategoryId || activePostId));
        line.classList.toggle("is-dimmed", !related);
      });

      svg.querySelectorAll("[data-edge-type='tag']").forEach(function (line) {
        var left = line.getAttribute("data-left");
        var right = line.getAttribute("data-right");
        var related = !activeCategoryId && !activePostId;

        if (activeCategoryId) {
          var leftPost = data.posts.find(function (post) { return post.id === left; });
          var rightPost = data.posts.find(function (post) { return post.id === right; });
          related = leftPost.categoryIds.indexOf(activeCategoryId) !== -1 && rightPost.categoryIds.indexOf(activeCategoryId) !== -1;
        }

        if (activePostId) {
          related = left === activePostId || right === activePostId;
        }

        line.classList.toggle("is-active", related && (activeCategoryId || activePostId));
        line.classList.toggle("is-dimmed", !related);
      });

      if (activePostId) {
        renderPostPanel(data.posts.find(function (post) { return post.id === activePostId; }));
      } else if (activeCategoryId) {
        renderCategoryPanel(categoryById.get(activeCategoryId));
      } else {
        renderPanelDefault();
      }
    }

    function setCategorySelection(categoryId) {
      state.postId = null;
      state.categoryId = state.categoryId === categoryId ? null : categoryId;
      updateSelection();
    }

    function setPostSelection(postId) {
      state.categoryId = null;
      state.postId = state.postId === postId ? null : postId;
      updateSelection();
    }

    function clearSelection() {
      state.categoryId = null;
      state.postId = null;
      updateSelection();
    }

    function draw() {
      var width = canvas.clientWidth;
      var height = Math.max(540, Math.round(width * 0.62));
      var padding = 28;
      var center = { x: width / 2, y: height / 2 };
      var radiusX = Math.max(170, width * 0.35);
      var radiusY = Math.max(150, height * 0.34);
      var categoryAngles = {};

      canvas.style.height = height + "px";
      svg.setAttribute("viewBox", "0 0 " + width + " " + height);
      svg.innerHTML = "";
      nodesLayer.innerHTML = "";

      data.categories.forEach(function (category, index) {
        var angle = -Math.PI / 2 + (index / data.categories.length) * Math.PI * 2;
        categoryAngles[category.id] = angle;
        category.x = center.x + Math.cos(angle) * radiusX;
        category.y = center.y + Math.sin(angle) * radiusY;
      });

      var bucketByCategory = {};
      data.posts.forEach(function (post, index) {
        var hubs = post.categoryIds.map(function (categoryId) {
          return categoryById.get(categoryId);
        }).filter(Boolean);

        var average = hubs.reduce(function (sum, category) {
          return { x: sum.x + category.x, y: sum.y + category.y };
        }, { x: 0, y: 0 });

        average.x /= hubs.length;
        average.y /= hubs.length;

        var primaryCategoryId = post.categoryIds.slice().sort(function (left, right) {
          return byPriority(categoryById.get(left).name) - byPriority(categoryById.get(right).name);
        })[0];
        var bucket = bucketByCategory[primaryCategoryId] || 0;
        bucketByCategory[primaryCategoryId] = bucket + 1;

        var ring = 18 + Math.floor(bucket / 5) * 18;
        var spokeAngle = categoryAngles[primaryCategoryId] + (bucket % 5) * (Math.PI / 2.5) + index * 0.11;
        var anchorX = center.x + (average.x - center.x) * (hubs.length > 1 ? 0.56 : 0.66);
        var anchorY = center.y + (average.y - center.y) * (hubs.length > 1 ? 0.56 : 0.66);

        post.targetX = anchorX + Math.cos(spokeAngle) * ring;
        post.targetY = anchorY + Math.sin(spokeAngle) * ring * 0.88;
        post.x = post.targetX;
        post.y = post.targetY;
        post.radius = 7 + Math.min(4, post.categoryIds.length - 1) + Math.min(2, Math.floor(post.tags.length / 3));
        post.primaryCategoryId = primaryCategoryId;
      });

      for (var iteration = 0; iteration < 110; iteration += 1) {
        for (var a = 0; a < data.posts.length; a += 1) {
          for (var b = a + 1; b < data.posts.length; b += 1) {
            var leftPost = data.posts[a];
            var rightPost = data.posts[b];
            var dx = rightPost.x - leftPost.x;
            var dy = rightPost.y - leftPost.y;
            var distance = Math.sqrt(dx * dx + dy * dy) || 0.001;
            var minimum = leftPost.radius + rightPost.radius + 10;

            if (distance < minimum) {
              var push = (minimum - distance) / distance * 0.42;
              leftPost.x -= dx * push;
              leftPost.y -= dy * push;
              rightPost.x += dx * push;
              rightPost.y += dy * push;
            }
          }
        }

        data.posts.forEach(function (post) {
          data.categories.forEach(function (category) {
            var dx = post.x - category.x;
            var dy = post.y - category.y;
            var distance = Math.sqrt(dx * dx + dy * dy) || 0.001;
            var minimum = 50 + post.radius;

            if (distance < minimum) {
              var push = (minimum - distance) / distance * 0.18;
              post.x += dx * push;
              post.y += dy * push;
            }
          });

          post.x += (post.targetX - post.x) * 0.08;
          post.y += (post.targetY - post.y) * 0.08;
          post.x = clamp(post.x, padding, width - padding);
          post.y = clamp(post.y, padding, height - padding);
        });
      }

      var categoryLinkGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
      var tagLinkGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
      categoryLinkGroup.setAttribute("class", "category-mindmap__link-group category-mindmap__link-group--category");
      tagLinkGroup.setAttribute("class", "category-mindmap__link-group category-mindmap__link-group--tag");

      data.posts.forEach(function (post) {
        post.categoryIds.forEach(function (categoryId) {
          var category = categoryById.get(categoryId);
          var line = document.createElementNS("http://www.w3.org/2000/svg", "line");
          line.setAttribute("x1", category.x);
          line.setAttribute("y1", category.y);
          line.setAttribute("x2", post.x);
          line.setAttribute("y2", post.y);
          line.setAttribute("data-edge-type", "category");
          line.setAttribute("data-category-id", categoryId);
          line.setAttribute("data-post-id", post.id);
          line.setAttribute("class", "category-mindmap__edge category-mindmap__edge--category");
          categoryLinkGroup.appendChild(line);
        });
      });

      tagEdges.forEach(function (edge) {
        var left = data.posts.find(function (post) { return post.id === edge.left; });
        var right = data.posts.find(function (post) { return post.id === edge.right; });
        var line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", left.x);
        line.setAttribute("y1", left.y);
        line.setAttribute("x2", right.x);
        line.setAttribute("y2", right.y);
        line.setAttribute("data-edge-type", "tag");
        line.setAttribute("data-left", edge.left);
        line.setAttribute("data-right", edge.right);
        line.setAttribute("class", "category-mindmap__edge category-mindmap__edge--tag");
        tagLinkGroup.appendChild(line);
      });

      svg.appendChild(categoryLinkGroup);
      svg.appendChild(tagLinkGroup);

      data.categories.forEach(function (category) {
        var button = document.createElement("button");
        button.type = "button";
        button.className = "category-mindmap__node category-mindmap__node--category";
        button.style.left = category.x + "px";
        button.style.top = category.y + "px";
        button.style.setProperty("--node-color", category.color);
        button.setAttribute("aria-label", category.name + " category");
        button.title = category.name + " (" + category.count + ")";
        if (category.image) {
          button.classList.add("has-image");
          button.style.setProperty("--node-image", 'url("' + category.image + '")');
          button.style.setProperty("--node-image-position", category.imagePosition || "center center");
        }
        button.innerHTML = '<span class="category-mindmap__node-title">' + escapeHtml(category.name) + '</span>';
        button.addEventListener("click", function () { setCategorySelection(category.id); });
        button.addEventListener("mouseenter", function () {
          if (!state.categoryId && !state.postId) {
            renderCategoryPanel(category);
          }
        });
        button.addEventListener("mouseleave", function () {
          if (!state.categoryId && !state.postId) {
            renderPanelDefault();
          }
        });
        categoryButtons.set(category.id, button);
        nodesLayer.appendChild(button);
      });

      data.posts.forEach(function (post) {
        var button = document.createElement("button");
        button.type = "button";
        button.className = "category-mindmap__node category-mindmap__node--post";
        button.style.left = post.x + "px";
        button.style.top = post.y + "px";
        button.style.width = post.radius * 2 + "px";
        button.style.height = post.radius * 2 + "px";
        button.style.setProperty("--node-color", categoryById.get(post.primaryCategoryId).color);
        button.setAttribute("aria-label", post.title);
        button.title = post.title;
        button.addEventListener("click", function () { setPostSelection(post.id); });
        button.addEventListener("mouseenter", function () {
          if (!state.categoryId && !state.postId) {
            renderPostPanel(post);
          }
        });
        button.addEventListener("mouseleave", function () {
          if (!state.categoryId && !state.postId) {
            renderPanelDefault();
          }
        });
        postButtons.set(post.id, button);
        nodesLayer.appendChild(button);
      });

      updateSelection();
    }

    renderPanelDefault();
    draw();

    canvas.addEventListener("click", function (event) {
      if (event.target.closest(".category-mindmap__node")) {
        return;
      }

      clearSelection();
    });

    window.addEventListener("resize", function () {
      window.clearTimeout(initMap._timer);
      initMap._timer = window.setTimeout(draw, 120);
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("[data-category-map]").forEach(initMap);
  });
})();
