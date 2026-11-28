/* Grid snake.  The board is CELLS x CELLS; every coordinate below is in cells. */
(function () {
  "use strict";

  var CELLS = __GRID_SIZE__;
  var CELL_PX = __CELL_PX__;
  var STEP_MS = __STEP_MS__;
  var START_LENGTH = __START_LENGTH__;
  var SHOW_SCORE = __SHOW_SCORE__;

  var canvas = document.getElementById("board");
  var ctx = canvas.getContext("2d");
  var scoreEl = document.getElementById("score");
  var bestEl = document.getElementById("best");

  var size = CELLS * CELL_PX;
  canvas.width = size;
  canvas.height = size;

  var snake, dir, nextDir, food, score, running, paused, timer;

  function randomCell() {
    return Math.floor(Math.random() * CELLS);
  }

  function spawnFood() {
    var tries = 0;
    do {
      food = { x: randomCell(), y: randomCell() };
      tries += 1;
    } while (occupied(food.x, food.y) && tries < 500);
  }

  function occupied(x, y) {
    for (var i = 0; i < snake.length; i += 1) {
      if (snake[i].x === x && snake[i].y === y) return true;
    }
    return false;
  }

  function reset() {
    snake = [];
    var mid = Math.floor(CELLS / 2);
    for (var i = 0; i < START_LENGTH; i += 1) {
      snake.push({ x: mid - i, y: mid });
    }
    dir = { x: 1, y: 0 };
    nextDir = { x: 1, y: 0 };
    score = 0;
    paused = false;
    running = true;
    spawnFood();
    updateHud();
    if (timer) clearInterval(timer);
    timer = setInterval(tick, STEP_MS);
  }

  function updateHud() {
    if (!SHOW_SCORE) {
      scoreEl.textContent = "";
      bestEl.textContent = "";
      return;
    }
    scoreEl.textContent = "Score: " + score;
    var best = Number(window.localStorage.getItem("snakeBest") || 0);
    bestEl.textContent = "Best: " + best;
  }

  function tick() {
    if (!running || paused) return;
    dir = nextDir;
    var head = { x: snake[0].x + dir.x, y: snake[0].y + dir.y };

    /* wrap around the edges instead of crashing into a wall */
    head.x = (head.x + CELLS) % CELLS;
    head.y = (head.y + CELLS) % CELLS;

    /* self collision ends the game */
    if (occupied(head.x, head.y) && !(head.x === food.x && head.y === food.y)) {
      gameOver();
      return;
    }

    snake.unshift(head);
    if (head.x === food.x && head.y === food.y) {
      score += 1;
      if (SHOW_SCORE) {
        var best = Number(window.localStorage.getItem("snakeBest") || 0);
        if (score > best) window.localStorage.setItem("snakeBest", String(score));
      }
      spawnFood();
      updateHud();
    } else {
      snake.pop();
    }
    draw();
  }

  function gameOver() {
    running = false;
    clearInterval(timer);
    draw();
    ctx.fillStyle = "rgba(0,0,0,0.72)";
    ctx.fillRect(0, 0, size, size);
    ctx.fillStyle = "#e6edf3";
    ctx.font = Math.round(CELL_PX * 1.4) + "px monospace";
    ctx.textAlign = "center";
    ctx.fillText("Game over", size / 2, size / 2);
    ctx.font = Math.round(CELL_PX * 0.9) + "px monospace";
    ctx.fillText("Press Restart", size / 2, size / 2 + CELL_PX * 2);
  }

  function draw() {
    ctx.fillStyle = getComputedStyle(document.documentElement)
      .getPropertyValue("--bg")
      .trim() || "#0d1117";
    ctx.fillRect(0, 0, size, size);

    var foodColor = getComputedStyle(document.documentElement)
      .getPropertyValue("--food")
      .trim() || "#ff4136";
    var snakeColor = getComputedStyle(document.documentElement)
      .getPropertyValue("--accent")
      .trim() || "#2ecc40";

    ctx.fillStyle = foodColor;
    ctx.fillRect(food.x * CELL_PX + 2, food.y * CELL_PX + 2, CELL_PX - 4, CELL_PX - 4);

    for (var i = 0; i < snake.length; i += 1) {
      ctx.fillStyle = snakeColor;
      ctx.globalAlpha = i === 0 ? 1 : 0.82;
      ctx.fillRect(snake[i].x * CELL_PX + 1, snake[i].y * CELL_PX + 1, CELL_PX - 2, CELL_PX - 2);
    }
    ctx.globalAlpha = 1;
  }

  function setDirection(x, y) {
    /* ignore a reversal, which would make the snake bite its own neck */
    if (x === -dir.x && y === -dir.y) return;
    nextDir = { x: x, y: y };
  }

  document.addEventListener("keydown", function (e) {
    var k = e.key;
    if (k === "ArrowUp" || k === "w" || k === "W") setDirection(0, -1);
    else if (k === "ArrowDown" || k === "s" || k === "S") setDirection(0, 1);
    else if (k === "ArrowLeft" || k === "a" || k === "A") setDirection(-1, 0);
    else if (k === "ArrowRight" || k === "d" || k === "D") setDirection(1, 0);
    else if (k === " ") { paused = !paused; }
    if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", " "].indexOf(k) >= 0) {
      e.preventDefault();
    }
  });

  document.getElementById("pause").addEventListener("click", function () {
    paused = !paused;
  });
  document.getElementById("restart").addEventListener("click", reset);

  /* swipe support for touch screens */
  var startX = 0, startY = 0;
  canvas.addEventListener("touchstart", function (e) {
    var t = e.changedTouches[0];
    startX = t.clientX; startY = t.clientY;
  }, { passive: true });
  canvas.addEventListener("touchend", function (e) {
    var t = e.changedTouches[0];
    var dx = t.clientX - startX;
    var dy = t.clientY - startY;
    if (Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > 20) setDirection(dx > 0 ? 1 : -1, 0);
    else if (Math.abs(dy) > 20) setDirection(0, dy > 0 ? 1 : -1);
  }, { passive: true });

  reset();
  draw();
})();
