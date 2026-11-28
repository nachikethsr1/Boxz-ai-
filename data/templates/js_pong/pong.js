/* Two-paddle Pong.  Court is COURT_W x COURT_H logical pixels. */
(function () {
  "use strict";

  var COURT_W = 600;
  var COURT_H = 400;
  var PADDLE_W = 12;
  var PADDLE_H = __PADDLE_H__;
  var PADDLE_SPEED = 6;
  var BALL_R = 7;
  var BALL_SPEED = __BALL_SPEED__;
  var ACCEL = __ACCEL__;
  var WIN_SCORE = __WIN_SCORE__;
  var CPU = __CPU_ENABLED__;
  var CPU_SKILL = __CPU_SKILL__;
  var PADDLE_COLOR = "__PADDLE_COLOR__";
  var BALL_COLOR = "__BALL_COLOR__";
  var COURT_COLOR = "__COURT_COLOR__";

  var canvas = document.getElementById("court");
  var ctx = canvas.getContext("2d");
  var leftEl = document.getElementById("left");
  var rightEl = document.getElementById("right");

  var left, right, ball, score, keys, serving, over;

  function resetBall(towardRight) {
    ball = {
      x: COURT_W / 2,
      y: COURT_H / 2,
      vx: (towardRight ? 1 : -1) * BALL_SPEED,
      vy: (Math.random() - 0.5) * BALL_SPEED * 0.8
    };
    serving = true;
  }

  function reset() {
    left = { y: COURT_H / 2 - PADDLE_H / 2 };
    right = { y: COURT_H / 2 - PADDLE_H / 2 };
    score = { left: 0, right: 0 };
    over = false;
    keys = {};
    resetBall(Math.random() < 0.5);
    render();
    draw();
  }

  function render() {
    leftEl.textContent = String(score.left);
    rightEl.textContent = String(score.right);
  }

  function clampPaddle(p) {
    if (p.y < 0) p.y = 0;
    if (p.y > COURT_H - PADDLE_H) p.y = COURT_H - PADDLE_H;
  }

  function step() {
    if (over) return;

    if (keys.w) left.y -= PADDLE_SPEED;
    if (keys.s) left.y += PADDLE_SPEED;
    clampPaddle(left);

    if (CPU) {
      /* the computer tracks the ball, but only as precisely as CPU_SKILL allows */
      var target = ball.y - PADDLE_H / 2;
      var delta = target - right.y;
      right.y += Math.max(-PADDLE_SPEED, Math.min(PADDLE_SPEED, delta * CPU_SKILL));
    } else {
      if (keys.ArrowUp) right.y -= PADDLE_SPEED;
      if (keys.ArrowDown) right.y += PADDLE_SPEED;
    }
    clampPaddle(right);

    if (serving) return;

    ball.x += ball.vx;
    ball.y += ball.vy;

    /* bounce off the top and bottom walls */
    if (ball.y < BALL_R) { ball.y = BALL_R; ball.vy *= -1; }
    if (ball.y > COURT_H - BALL_R) { ball.y = COURT_H - BALL_R; ball.vy *= -1; }

    /* left paddle */
    if (ball.vx < 0 && ball.x - BALL_R < PADDLE_W &&
        ball.y > left.y && ball.y < left.y + PADDLE_H) {
      ball.x = PADDLE_W + BALL_R;
      ball.vx = Math.abs(ball.vx) * ACCEL;
      ball.vy += ((ball.y - (left.y + PADDLE_H / 2)) / PADDLE_H) * 2.2;
    }

    /* right paddle */
    if (ball.vx > 0 && ball.x + BALL_R > COURT_W - PADDLE_W &&
        ball.y > right.y && ball.y < right.y + PADDLE_H) {
      ball.x = COURT_W - PADDLE_W - BALL_R;
      ball.vx = -Math.abs(ball.vx) * ACCEL;
      ball.vy += ((ball.y - (right.y + PADDLE_H / 2)) / PADDLE_H) * 2.2;
    }

    /* scoring */
    if (ball.x < -BALL_R) {
      score.right += 1;
      pointScored();
    } else if (ball.x > COURT_W + BALL_R) {
      score.left += 1;
      pointScored();
    }
  }

  function pointScored() {
    render();
    if (score.left >= WIN_SCORE || score.right >= WIN_SCORE) {
      over = true;
      draw();
      drawOverlay((score.left > score.right ? "Left" : "Right") + " wins!");
      return;
    }
    resetBall(ball.x < 0);
    draw();
  }

  function drawOverlay(text) {
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.fillRect(0, 0, COURT_W, COURT_H);
    ctx.fillStyle = "#e6edf3";
    ctx.font = "28px monospace";
    ctx.textAlign = "center";
    ctx.fillText(text, COURT_W / 2, COURT_H / 2);
    ctx.font = "14px monospace";
    ctx.fillText("Press Space to play again", COURT_W / 2, COURT_H / 2 + 34);
  }

  function draw() {
    ctx.fillStyle = COURT_COLOR;
    ctx.fillRect(0, 0, COURT_W, COURT_H);

    /* dashed centre line */
    ctx.strokeStyle = "rgba(255,255,255,0.16)";
    ctx.setLineDash([8, 10]);
    ctx.beginPath();
    ctx.moveTo(COURT_W / 2, 0);
    ctx.lineTo(COURT_W / 2, COURT_H);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = PADDLE_COLOR;
    ctx.fillRect(0, left.y, PADDLE_W, PADDLE_H);
    ctx.fillRect(COURT_W - PADDLE_W, right.y, PADDLE_W, PADDLE_H);

    ctx.fillStyle = BALL_COLOR;
    ctx.beginPath();
    ctx.arc(ball.x, ball.y, BALL_R, 0, Math.PI * 2);
    ctx.fill();

    if (serving && !over) {
      ctx.fillStyle = "#8b9bab";
      ctx.font = "13px monospace";
      ctx.textAlign = "center";
      ctx.fillText("Press Space to serve", COURT_W / 2, COURT_H / 2 + 42);
    }
  }

  function frame() {
    step();
    draw();
    window.requestAnimationFrame(frame);
  }

  document.addEventListener("keydown", function (e) {
    keys[e.key] = true;
    if (e.key === " ") {
      if (over) { reset(); return; }
      serving = false;
      e.preventDefault();
    }
    if (["ArrowUp", "ArrowDown", " "].indexOf(e.key) >= 0) e.preventDefault();
  });

  document.addEventListener("keyup", function (e) {
    keys[e.key] = false;
  });

  /* touch: drag anywhere on the canvas to move the left paddle */
  canvas.addEventListener("touchmove", function (e) {
    var rect = canvas.getBoundingClientRect();
    var y = ((e.touches[0].clientY - rect.top) / rect.height) * COURT_H;
    left.y = y - PADDLE_H / 2;
    clampPaddle(left);
    if (serving && !over) serving = false;
    e.preventDefault();
  }, { passive: false });

  reset();
  frame();
})();
