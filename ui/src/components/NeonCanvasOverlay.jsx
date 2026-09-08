import { useRef, useEffect, useState, useCallback } from 'react';

/* ─────────────────────────────────────────────────────────────
   PALETTE DEFINITIONS  – { name, r, g, b, glow, edge }
   r/g/b  = the neon tint color (0-255)
   glow   = CSS rgba string for the outer bloom
   edge   = CSS color string for the hard outline
──────────────────────────────────────────────────────────────── */
export const PALETTES = {
  redcyan: {
    name: 'Red–Cyan',
    r: 0, g: 229, b: 255,           // cyan channel
    rChannel: 255, gChannel: 0, bChannel: 0,  // red for original shift
    glow: 'rgba(0,229,255,',
    edge: '#00e5ff',
    label: '🔵',
    description: 'Classic Red-Cyan anaglyph blending',
    chipColor: '#00e5ff',
  },
  neongreen: {
    name: 'Toxic Neon',
    r: 57, g: 255, b: 20,
    rChannel: 57, gChannel: 255, bChannel: 20,
    glow: 'rgba(57,255,20,',
    edge: '#39ff14',
    label: '🟢',
    description: 'Toxic neon green heat contour',
    chipColor: '#39ff14',
  },
  violet: {
    name: 'Bio-Violet',
    r: 191, g: 95, b: 255,
    rChannel: 191, gChannel: 95, bChannel: 255,
    glow: 'rgba(191,95,255,',
    edge: '#bf5fff',
    label: '🟣',
    description: 'Bioluminescent violet spectrum',
    chipColor: '#bf5fff',
  },
  solar: {
    name: 'Solar Heat',
    r: 255, g: 149, b: 0,
    rChannel: 255, gChannel: 149, bChannel: 0,
    glow: 'rgba(255,149,0,',
    edge: '#ff9500',
    label: '🟠',
    description: 'Thermal solar infrared mapping',
    chipColor: '#ff9500',
  },
};

/* ─────────────────────────────────────────────────────────────
   RENDER MODES
──────────────────────────────────────────────────────────────── */
export const MODES = {
  overlay:   'overlay',
  spotlight: 'spotlight',
  split:     'split',
  extract:   'extract',
};

/* ═══════════════════════════════════════════════════════════════
   CORE RENDER FUNCTION – called every animation frame
═══════════════════════════════════════════════════════════════ */
function renderFrame({
  ctx,
  imgData,       // original ImageData from source image
  maskData,      // Float32Array 0-1 values, same dims as canvas
  W, H,
  palette,
  threshold,
  glowRadius,
  glowOpacity,
  pulseOn,
  pulsePhase,
  mode,
  splitX,        // for split mode: 0-1
  spotX, spotY,  // for spotlight mode
  spotRadius,
}) {
  ctx.clearRect(0, 0, W, H);

  // ── 1. Draw base image ───────────────────────────────────────
  ctx.putImageData(imgData, 0, 0);

  if (!maskData) return;  // no mask yet, just show image

  const pal = palette;

  // ── 2. Build per-pixel neon overlay on offscreen canvas ──────
  const overlayImg = ctx.createImageData(W, H);
  const od = overlayImg.data;
  const sd = imgData.data;

  for (let i = 0; i < W * H; i++) {
    const v = maskData[i]; // 0-1 mask confidence

    if (v < threshold) continue;  // below threshold → skip

    const si = i * 4;
    const alpha = (v - threshold) / (1 - threshold); // re-normalize above threshold

    // ── Red-Cyan anaglyph split for "redcyan" palette ────────
    if (pal === PALETTES.redcyan) {
      // Shift red channel to original * tint, cyan channel boosted
      const boost = alpha * 1.6;
      od[si]     = Math.min(255, sd[si]     * 0.3 + 255 * boost);  // R boosted → red tint
      od[si + 1] = Math.min(255, sd[si + 1] * 0.15);               // G crushed
      od[si + 2] = Math.min(255, sd[si + 2] * 0.3 + 255 * boost);  // B boosted → cyan tint
      od[si + 3] = Math.min(255, alpha * 200);
    } else {
      // Generic neon color channel blend
      od[si]     = pal.r;
      od[si + 1] = pal.g;
      od[si + 2] = pal.b;
      od[si + 3] = Math.min(255, alpha * 180);
    }
  }

  // ── 3. Apply mode masking ────────────────────────────────────
  if (mode === MODES.split) {
    const cutX = Math.floor(splitX * W);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        if (x >= cutX) {
          const i = (y * W + x) * 4;
          od[i + 3] = 0;  // hide overlay on right side
        }
      }
    }
  }

  if (mode === MODES.spotlight) {
    const R = spotRadius;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const dist = Math.sqrt((x - spotX) ** 2 + (y - spotY) ** 2);
        if (dist > R) {
          const i = (y * W + x) * 4;
          od[i + 3] = 0;
        } else {
          // soft edge
          const i = (y * W + x) * 4;
          const edgeFade = Math.max(0, 1 - (dist / R) ** 2);
          od[i + 3] = Math.floor(od[i + 3] * edgeFade);
        }
      }
    }
  }

  // ── 4. Composite overlay onto canvas ─────────────────────────
  const offCtx = document.createElement('canvas');
  offCtx.width = W; offCtx.height = H;
  const oCtx = offCtx.getContext('2d');
  oCtx.putImageData(overlayImg, 0, 0);

  ctx.save();
  ctx.globalCompositeOperation = 'screen';
  ctx.globalAlpha = glowOpacity;
  ctx.drawImage(offCtx, 0, 0);
  ctx.restore();

  // ── 5. Neon glow blur layers (layered bloom) ─────────────────
  const bloomRadii = [glowRadius, glowRadius * 2.5, glowRadius * 5];
  const bloomAlphas = [0.55, 0.3, 0.15];

  bloomRadii.forEach((radius, idx) => {
    const blurCtx = document.createElement('canvas');
    blurCtx.width = W; blurCtx.height = H;
    const bc = blurCtx.getContext('2d');
    bc.filter = `blur(${radius}px)`;
    bc.drawImage(offCtx, 0, 0);

    ctx.save();
    ctx.globalCompositeOperation = 'screen';
    ctx.globalAlpha = bloomAlphas[idx] * glowOpacity;
    ctx.drawImage(blurCtx, 0, 0);
    ctx.restore();
  });

  // ── 6. Hard neon edge contour ─────────────────────────────────
  const edgeCtx = document.createElement('canvas');
  edgeCtx.width = W; edgeCtx.height = H;
  const ec = edgeCtx.getContext('2d');

  const edgeImg = ec.createImageData(W, H);
  const ed = edgeImg.data;

  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const c  = maskData[y * W + x];
      const n  = maskData[(y - 1) * W + x];
      const s  = maskData[(y + 1) * W + x];
      const ww = maskData[y * W + (x - 1)];
      const e  = maskData[y * W + (x + 1)];

      const grad = Math.abs(c - n) + Math.abs(c - s) + Math.abs(c - ww) + Math.abs(c - e);
      if (grad > 0.08 && c > threshold) {
        const i = (y * W + x) * 4;
        ed[i]     = pal.r;
        ed[i + 1] = pal.g;
        ed[i + 2] = pal.b;
        ed[i + 3] = Math.min(255, grad * 600);
      }
    }
  }

  ec.putImageData(edgeImg, 0, 0);

  // Edge glow
  const edgeGlowCtx = document.createElement('canvas');
  edgeGlowCtx.width = W; edgeGlowCtx.height = H;
  const egc = edgeGlowCtx.getContext('2d');
  egc.filter = `blur(${glowRadius * 0.8}px)`;
  egc.drawImage(edgeCtx, 0, 0);

  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  ctx.globalAlpha = 0.9 * glowOpacity;
  ctx.drawImage(edgeGlowCtx, 0, 0);
  ctx.restore();

  // Hard bright core
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  ctx.globalAlpha = Math.min(1, glowOpacity * 1.2);
  ctx.drawImage(edgeCtx, 0, 0);
  ctx.restore();

  // ── 7. Pulsing radar aura ─────────────────────────────────────
  if (pulseOn && maskData) {
    // Find centroid of detected region
    let cx = 0, cy = 0, cnt = 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        if (maskData[y * W + x] > threshold + 0.1) {
          cx += x; cy += y; cnt++;
        }
      }
    }
    if (cnt > 0) {
      cx /= cnt; cy /= cnt;
      const maxR = Math.max(W, H) * 0.6;
      const pR = ((pulsePhase % 1) * maxR);
      const pulseA = Math.max(0, 1 - pR / maxR) * 0.35;

      ctx.save();
      ctx.globalCompositeOperation = 'screen';
      const grad = ctx.createRadialGradient(cx, cy, Math.max(0, pR - 25), cx, cy, pR);
      grad.addColorStop(0, `${pal.glow}${pulseA})`);
      grad.addColorStop(1, `${pal.glow}0)`);
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, W, H);
      ctx.restore();
    }
  }

  // ── 8. Split divider line ─────────────────────────────────────
  if (mode === MODES.split) {
    const cx = splitX * W;
    ctx.save();
    ctx.strokeStyle = pal.edge;
    ctx.lineWidth = 2;
    ctx.shadowBlur = 16;
    ctx.shadowColor = pal.edge;
    ctx.beginPath();
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, H);
    ctx.stroke();
    ctx.restore();
  }

  // ── 9. Spotlight ring ─────────────────────────────────────────
  if (mode === MODES.spotlight) {
    ctx.save();
    ctx.strokeStyle = pal.edge;
    ctx.lineWidth = 2;
    ctx.shadowBlur = 20;
    ctx.shadowColor = pal.edge;
    ctx.globalAlpha = 0.8;
    ctx.beginPath();
    ctx.arc(spotX, spotY, spotRadius, 0, Math.PI * 2);
    ctx.stroke();

    // Cross-hair ticks
    const tickLen = 10;
    [[0, -spotRadius], [0, spotRadius], [-spotRadius, 0], [spotRadius, 0]].forEach(([dx, dy]) => {
      ctx.beginPath();
      ctx.moveTo(spotX + dx, spotY + dy);
      ctx.lineTo(
        spotX + dx * (1 - tickLen / spotRadius),
        spotY + dy * (1 - tickLen / spotRadius)
      );
      ctx.stroke();
    });
    ctx.restore();
  }
}

/* ═══════════════════════════════════════════════════════════════
   COMPONENT
═══════════════════════════════════════════════════════════════ */
export default function NeonCanvasOverlay({
  srcImage,        // HTMLImageElement or null
  maskArray,       // Float32Array or null
  maskW, maskH,    // original mask dimensions
  palette = PALETTES.redcyan,
  threshold = 0.3,
  glowRadius = 6,
  glowOpacity = 0.85,
  pulseOn = true,
  mode = MODES.overlay,
  onExtractedURL,  // callback(dataURL)
}) {
  const canvasRef  = useRef(null);
  const rafRef     = useRef(null);
  const stateRef   = useRef({});

  const [splitX, setSplitX] = useState(0.5);
  const [spotPos, setSpotPos] = useState({ x: 0.5, y: 0.5 });
  const [spotRadius] = useState(90);
  const isDragging = useRef(false);

  // ── Prepare scaled mask ───────────────────────────────────────
  const [scaledMask, setScaledMask] = useState(null);
  const [imgData, setImgData]   = useState(null);
  const [canvasW, setCanvasW]   = useState(0);
  const [canvasH, setCanvasH]   = useState(0);

  // Resize & draw image + scale mask when source changes
  useEffect(() => {
    if (!srcImage) { setImgData(null); setScaledMask(null); return; }

    const MAX = 800;
    let w = srcImage.naturalWidth  || srcImage.width  || 640;
    let h = srcImage.naturalHeight || srcImage.height || 480;
    if (w > MAX) { h = Math.round(h * MAX / w); w = MAX; }
    if (h > MAX) { w = Math.round(w * MAX / h); h = MAX; }

    setCanvasW(w); setCanvasH(h);

    const tmp = document.createElement('canvas');
    tmp.width = w; tmp.height = h;
    const tc = tmp.getContext('2d');
    tc.drawImage(srcImage, 0, 0, w, h);
    setImgData(tc.getImageData(0, 0, w, h));

    // Scale mask bilinearly
    if (maskArray && maskW && maskH) {
      const scaled = new Float32Array(w * h);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const sx = (x / w) * maskW;
          const sy = (y / h) * maskH;
          const ix = Math.floor(sx), iy = Math.floor(sy);
          const fx = sx - ix, fy = sy - iy;
          const i00 = iy * maskW + ix;
          const i10 = iy * maskW + Math.min(ix + 1, maskW - 1);
          const i01 = Math.min(iy + 1, maskH - 1) * maskW + ix;
          const i11 = Math.min(iy + 1, maskH - 1) * maskW + Math.min(ix + 1, maskW - 1);
          scaled[y * w + x] =
            maskArray[i00] * (1 - fx) * (1 - fy) +
            maskArray[i10] * fx * (1 - fy) +
            maskArray[i01] * (1 - fx) * fy +
            maskArray[i11] * fx * fy;
        }
      }
      setScaledMask(scaled);
    } else {
      setScaledMask(null);
    }
  }, [srcImage, maskArray, maskW, maskH]);

  // ── Extract isolated object ────────────────────────────────────
  useEffect(() => {
    if (mode !== MODES.extract || !imgData || !scaledMask) return;
    const { width: W, height: H } = imgData;
    const out = new ImageData(W, H);
    const od = out.data, sd = imgData.data;
    for (let i = 0; i < W * H; i++) {
      const v = scaledMask[i];
      if (v > threshold) {
        const si = i * 4;
        od[si]     = sd[si];
        od[si + 1] = sd[si + 1];
        od[si + 2] = sd[si + 2];
        od[si + 3] = Math.min(255, (v - threshold) / (1 - threshold) * 255 * 1.4);
      }
    }
    const tmp = document.createElement('canvas');
    tmp.width = W; tmp.height = H;
    tmp.getContext('2d').putImageData(out, 0, 0);
    onExtractedURL && onExtractedURL(tmp.toDataURL());
  }, [mode, imgData, scaledMask, threshold, onExtractedURL]);

  // ── Animation loop ─────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imgData) return;

    canvas.width  = canvasW;
    canvas.height = canvasH;
    const ctx = canvas.getContext('2d');

    let startTime = null;

    function loop(ts) {
      if (!startTime) startTime = ts;
      const elapsed = (ts - startTime) / 1000;

      renderFrame({
        ctx, imgData,
        maskData: scaledMask,
        W: canvasW, H: canvasH,
        palette,
        threshold,
        glowRadius,
        glowOpacity,
        pulseOn,
        pulsePhase: elapsed * 0.35,  // pulse speed
        mode,
        splitX,
        spotX: spotPos.x * canvasW,
        spotY: spotPos.y * canvasH,
        spotRadius,
      });

      rafRef.current = requestAnimationFrame(loop);
    }

    rafRef.current = requestAnimationFrame(loop);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [imgData, scaledMask, canvasW, canvasH, palette, threshold, glowRadius, glowOpacity, pulseOn, mode, splitX, spotPos, spotRadius]);

  // ── Mouse interaction ──────────────────────────────────────────
  const handleMouseMove = useCallback((e) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;

    if (mode === MODES.spotlight) {
      setSpotPos({ x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)) });
    }
    if (mode === MODES.split && isDragging.current) {
      setSplitX(Math.max(0.05, Math.min(0.95, x)));
    }
  }, [mode]);

  const handleMouseDown = useCallback((e) => {
    isDragging.current = true;
    if (mode === MODES.split) {
      const rect = canvasRef.current.getBoundingClientRect();
      setSplitX(Math.max(0.05, Math.min(0.95, (e.clientX - rect.left) / rect.width)));
    }
  }, [mode]);

  const handleMouseUp = useCallback(() => { isDragging.current = false; }, []);

  const canvasStyle = mode === MODES.spotlight
    ? { cursor: 'none' }
    : mode === MODES.split
    ? { cursor: 'ew-resize' }
    : {};

  return (
    <div className="canvas-wrapper">
      {!imgData && (
        <div className="canvas-placeholder">
          <span style={{ fontSize: 52, filter: 'drop-shadow(0 0 16px rgba(0,229,255,0.5))', animation: 'float 3s ease-in-out infinite' }}>🔍</span>
          <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Upload image or select a preset</span>
        </div>
      )}
      <canvas
        ref={canvasRef}
        style={{ display: imgData ? 'block' : 'none', ...canvasStyle }}
        onMouseMove={handleMouseMove}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />
      <div className="scanlines" />
      {/* Split label hints */}
      {mode === MODES.split && imgData && (
        <>
          <div style={{ position:'absolute', top: 12, left: 16, fontSize: 10, fontWeight: 700, color: 'rgba(255,255,255,0.5)', letterSpacing: '0.1em', textTransform: 'uppercase', pointerEvents: 'none' }}>OVERLAY</div>
          <div style={{ position:'absolute', top: 12, right: 16, fontSize: 10, fontWeight: 700, color: 'rgba(255,255,255,0.5)', letterSpacing: '0.1em', textTransform: 'uppercase', pointerEvents: 'none' }}>ORIGINAL</div>
        </>
      )}
      {mode === MODES.spotlight && imgData && (
        <div style={{ position:'absolute', bottom: 14, left: '50%', transform: 'translateX(-50%)', fontSize: 10, fontWeight: 700, color: 'rgba(0,229,255,0.7)', letterSpacing: '0.1em', textTransform: 'uppercase', pointerEvents: 'none', whiteSpace: 'nowrap' }}>
          🔦 MOVE CURSOR TO SCAN
        </div>
      )}
    </div>
  );
}
