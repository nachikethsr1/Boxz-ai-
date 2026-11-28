/* Stopwatch (counts up) plus a countdown (counts down), sharing one display. */
(function () {
  "use strict";

  var TICK_MS = __TICK_MS__;
  var RING_SECONDS = 60;

  var display = document.getElementById("display");
  var bar = document.getElementById("bar");
  var lapsEl = document.getElementById("laps");
  var countdownInput = document.getElementById("countdown");

  /* elapsed is in milliseconds; direction is +1 (stopwatch) or -1 (countdown) */
  var elapsed = 0;
  var limit = 0;
  var running = false;
  var handle = null;
  var last = 0;
  var lapCount = 0;

  function format(ms) {
    if (ms < 0) ms = 0;
    var tenths = Math.floor(ms / 100);
    var minutes = Math.floor(tenths / 600);
    var seconds = Math.floor((tenths % 600) / 10);
    var tenth = tenths % 10;
    return String(minutes).padStart(2, "0") + ":" +
           String(seconds).padStart(2, "0") + "." + tenth;
  }

  function render() {
    display.textContent = format(elapsed);
    var fraction = 0;
    if (limit > 0) {
      fraction = Math.max(0, Math.min(1, elapsed / limit));
    } else {
      fraction = (elapsed % (RING_SECONDS * 1000)) / (RING_SECONDS * 1000);
    }
    var circumference = 2 * Math.PI * 52;
    bar.style.strokeDashoffset = String(circumference * (1 - fraction));
  }

  function stop() {
    running = false;
    if (handle) {
      clearInterval(handle);
      handle = null;
    }
  }

  function start() {
    if (running) return;
    running = true;
    last = Date.now();
    handle = setInterval(function () {
      var now = Date.now();
      var delta = now - last;
      last = now;
      elapsed += delta;
      if (limit > 0 && elapsed >= limit) {
        elapsed = limit;
        render();
        stop();
        flashDone();
        return;
      }
      render();
    }, TICK_MS);
  }

  function flashDone() {
    display.textContent = "Time!";
    display.animate(
      [{ opacity: 1 }, { opacity: 0.25 }, { opacity: 1 }],
      { duration: 600, iterations: 3 }
    );
  }

  document.getElementById("start").addEventListener("click", function () {
    if (limit > 0 && elapsed >= limit) elapsed = 0;
    start();
  });

  document.getElementById("pause").addEventListener("click", stop);

  document.getElementById("reset").addEventListener("click", function () {
    stop();
    elapsed = 0;
    limit = 0;
    lapCount = 0;
    lapsEl.innerHTML = "";
    render();
  });

  document.getElementById("lap").addEventListener("click", function () {
    if (elapsed === 0) return;
    lapCount += 1;
    var li = document.createElement("li");
    var label = document.createElement("span");
    label.textContent = "Lap " + lapCount;
    var value = document.createElement("span");
    value.textContent = format(elapsed);
    li.appendChild(label);
    li.appendChild(value);
    lapsEl.insertBefore(li, lapsEl.firstChild);
  });

  document.getElementById("startCountdown").addEventListener("click", function () {
    var seconds = Number(countdownInput.value);
    if (isNaN(seconds) || seconds <= 0) {
      countdownInput.focus();
      return;
    }
    stop();
    elapsed = 0;
    limit = seconds * 1000;
    render();
    start();
  });

  render();
})();
