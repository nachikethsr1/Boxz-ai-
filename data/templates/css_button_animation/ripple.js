/* Spawn a ripple element at the pointer position on every button press. */
(function () {
  "use strict";

  var ENABLED = __RIPPLE_ENABLED__;
  var RIPPLE_MS = 600;

  if (!ENABLED) return;

  var buttons = document.querySelectorAll(".btn");

  Array.prototype.forEach.call(buttons, function (button) {
    button.addEventListener("pointerdown", function (e) {
      var rect = button.getBoundingClientRect();
      var size = Math.max(rect.width, rect.height);

      var ripple = document.createElement("span");
      ripple.className = "ripple";
      ripple.style.width = size + "px";
      ripple.style.height = size + "px";
      ripple.style.left = (e.clientX - rect.left - size / 2) + "px";
      ripple.style.top = (e.clientY - rect.top - size / 2) + "px";

      button.appendChild(ripple);
      window.setTimeout(function () {
        if (ripple.parentNode) ripple.parentNode.removeChild(ripple);
      }, RIPPLE_MS);
    });
  });
})();
