// Lightbox for publication preview thumbnails.
//
// medium-zoom cannot be used for these: it clones the <img> at the
// thumbnail's fixed 200x150 plate box, so an image with a different natural
// aspect ratio comes out stretched, and its zoom scale is computed from that
// box rather than the image. This overlay shows the original file at its
// natural size, capped to the viewport.
document.addEventListener("DOMContentLoaded", function () {
  const previews = document.querySelectorAll(".pub-thumb img.preview");
  if (previews.length === 0) return;

  let overlay = null;

  function close() {
    if (overlay === null) return;
    overlay.remove();
    overlay = null;
    document.removeEventListener("keydown", onKeydown);
  }

  function onKeydown(event) {
    if (event.key === "Escape") close();
  }

  previews.forEach(function (thumb) {
    thumb.addEventListener("click", function () {
      if (overlay !== null) return;
      const full = new Image();
      full.src = thumb.currentSrc || thumb.src;
      full.alt = thumb.alt || "";
      overlay = document.createElement("div");
      overlay.className = "pub-lightbox";
      overlay.appendChild(full);
      overlay.addEventListener("click", close);
      document.addEventListener("keydown", onKeydown);
      document.body.appendChild(overlay);
    });
  });
});
