/* A drifting particle field.  Particles link to near neighbours and burst on click. */
(function () {
  "use strict";

  var COUNT = __PARTICLE_COUNT__;
  var SPEED = __SPEED__;
  var RADIUS = __PARTICLE_RADIUS__;
  var LINK_DISTANCE = __LINK_DISTANCE__;
  var LINK_LINES = __LINK_LINES__;
  var COLOR = "__PARTICLE_COLOR__";
  var LINK_COLOR = "__LINK_COLOR__";

  var canvas = document.getElementById("fx");
  var ctx = canvas.getContext("2d");
  var width = 0;
  var height = 0;
  var dpr = 1;
  var particles = [];
  var pointer = { x: -9999, y: -9999 };
  var running = true;

  function resize() {
    dpr = window.devicePixelRatio || 1;
    width = canvas.clientWidth;
    height = canvas.clientHeight;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function makeParticle(x, y) {
    return {
      x: x === undefined ? Math.random() * width : x,
      y: y === undefined ? Math.random() * height : y,
      vx: (Math.random() - 0.5) * SPEED,
      vy: (Math.random() - 0.5) * SPEED,
      r: RADIUS * (0.6 + Math.random() * 0.8)
    };
  }

  function seed() {
    particles = [];
    for (var i = 0; i < COUNT; i += 1) particles.push(makeParticle());
  }

  function step() {
    for (var i = 0; i < particles.length; i += 1) {
      var p = particles[i];
      p.x += p.vx;
      p.y += p.vy;

      /* bounce off the edges so nothing drifts away forever */
      if (p.x < 0 || p.x > width) { p.vx *= -1; p.x = Math.max(0, Math.min(width, p.x)); }
      if (p.y < 0 || p.y > height) { p.vy *= -1; p.y = Math.max(0, Math.min(height, p.y)); }

      /* the pointer gently repels nearby particles */
      var dx = p.x - pointer.x;
      var dy = p.y - pointer.y;
      var dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 110 && dist > 0.001) {
        var push = (110 - dist) / 110;
        p.x += (dx / dist) * push * 1.6;
        p.y += (dy / dist) * push * 1.6;
      }
    }
  }

  function draw() {
    ctx.clearRect(0, 0, width, height);

    if (LINK_LINES) {
      ctx.strokeStyle = LINK_COLOR;
      for (var i = 0; i < particles.length; i += 1) {
        for (var j = i + 1; j < particles.length; j += 1) {
          var a = particles[i];
          var b = particles[j];
          var dx = a.x - b.x;
          var dy = a.y - b.y;
          var d2 = dx * dx + dy * dy;
          if (d2 < LINK_DISTANCE * LINK_DISTANCE) {
            var alpha = 1 - Math.sqrt(d2) / LINK_DISTANCE;
            ctx.globalAlpha = alpha * 0.5;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }
      ctx.globalAlpha = 1;
    }

    ctx.fillStyle = COLOR;
    for (var k = 0; k < particles.length; k += 1) {
      var p = particles[k];
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function frame() {
    if (running) {
      step();
      draw();
    }
    window.requestAnimationFrame(frame);
  }

  window.addEventListener("resize", function () {
    resize();
    seed();
  });

  window.addEventListener("pointermove", function (e) {
    pointer.x = e.clientX;
    pointer.y = e.clientY;
  });

  window.addEventListener("pointerleave", function () {
    pointer.x = -9999;
    pointer.y = -9999;
  });

  window.addEventListener("pointerdown", function (e) {
    /* spawn a short-lived burst at the click point */
    for (var i = 0; i < 24; i += 1) {
      var angle = (Math.PI * 2 * i) / 24;
      var q = makeParticle(e.clientX, e.clientY);
      q.vx = Math.cos(angle) * SPEED * 3;
      q.vy = Math.sin(angle) * SPEED * 3;
      particles.push(q);
    }
    /* keep the population bounded so long sessions stay smooth */
    while (particles.length > COUNT * 2) particles.shift();
  });

  resize();
  seed();
  frame();
})();
