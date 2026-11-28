/* A counter with +, - and reset.  The value optionally survives a refresh. */
(function () {
  "use strict";

  var START = __START_VALUE__;
  var STEP = __STEP__;
  var MIN = __MIN_VALUE__;
  var MAX = __MAX_VALUE__;
  var PERSIST = __PERSIST__;
  var STORE_KEY = "boxz-counter-v1";

  var countEl = document.getElementById("count");
  var messageEl = document.getElementById("message");

  var value = START;

  if (PERSIST) {
    var saved = window.localStorage.getItem(STORE_KEY);
    if (saved !== null && saved !== "") {
      var parsed = Number(saved);
      if (!isNaN(parsed)) value = parsed;
    }
  }

  function clamp(n) {
    if (n < MIN) return MIN;
    if (n > MAX) return MAX;
    return n;
  }

  function render() {
    countEl.textContent = String(value);
    if (value === MAX) {
      messageEl.textContent = "Maximum reached";
    } else if (value === MIN) {
      messageEl.textContent = "Minimum reached";
    } else {
      messageEl.textContent = "";
    }
    if (PERSIST) {
      window.localStorage.setItem(STORE_KEY, String(value));
    }
  }

  function change(delta) {
    var before = value;
    value = clamp(value + delta);
    render();
    if (value === before) {
      /* tell the user the limit stopped the change instead of doing nothing */
      messageEl.textContent = delta > 0 ? "Maximum reached" : "Minimum reached";
    }
  }

  document.getElementById("inc").addEventListener("click", function () {
    change(STEP);
  });

  document.getElementById("dec").addEventListener("click", function () {
    change(-STEP);
  });

  document.getElementById("reset").addEventListener("click", function () {
    value = START;
    render();
  });

  /* keyboard shortcuts, but only when the focus is not already on a button */
  document.addEventListener("keydown", function (e) {
    if (e.target && e.target.tagName === "BUTTON") return;
    if (e.key === "+" || e.key === "=") change(STEP);
    else if (e.key === "-") change(-STEP);
    else if (e.key === "0") { value = START; render(); }
  });

  render();
})();
