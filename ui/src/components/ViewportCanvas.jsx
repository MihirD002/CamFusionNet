import React, { useRef, useEffect, useState, useCallback } from 'react';
import { extractContour, getThermalColor } from '../utils/colormap';

export default function ViewportCanvas({
  imageSrc,
  maskSrc,
  opacity,
  glowRadius,
  threshold,
  visualMode = 'glow',
  splitPosition = 50,
  setSplitPosition,
  onSnapshot,
  onDropImage,
  onTriggerUpload
}) {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const baseImgRef = useRef(null);
  const maskImgRef = useRef(null);

  const [zoom, setZoom] = useState(1);
  const [stats, setStats] = useState({ width: 0, height: 0, camouflagedPercent: 0 });
  const [imagesLoaded, setImagesLoaded] = useState(false);
  const [isDraggingOver, setIsDraggingOver] = useState(false);
  const [isDraggingCurtain, setIsDraggingCurtain] = useState(false);

  // Load images safely
  useEffect(() => {
    let active = true;
    setImagesLoaded(false);

    if (!imageSrc) return;

    const base = new Image();
    const mask = new Image();

    if (imageSrc.startsWith('http')) {
      base.crossOrigin = 'anonymous';
    }
    if (maskSrc && maskSrc.startsWith('http')) {
      mask.crossOrigin = 'anonymous';
    }

    let baseReady = false;
    let maskReady = false;

    const tryFinalize = () => {
      if (!active) return;
      if (baseReady && maskReady) {
        baseImgRef.current = base;
        maskImgRef.current = mask;
        setImagesLoaded(true);
        setStats(prev => ({
          ...prev,
          width: base.naturalWidth || 800,
          height: base.naturalHeight || 600
        }));
      }
    };

    base.onload = () => {
      if (base.naturalWidth > 0) {
        baseReady = true;
        tryFinalize();
      }
    };
    base.onerror = () => {
      console.warn('Base image failed to load:', imageSrc);
    };

    const setBlankFallbackMask = () => {
      const fallbackCanvas = document.createElement('canvas');
      fallbackCanvas.width = 64;
      fallbackCanvas.height = 64;
      const fCtx = fallbackCanvas.getContext('2d');
      fCtx.fillStyle = '#000';
      fCtx.fillRect(0, 0, 64, 64);
      mask.onload = () => {
        maskReady = true;
        tryFinalize();
      };
      mask.src = fallbackCanvas.toDataURL();
    };

    mask.onload = () => {
      if (mask.naturalWidth > 0) {
        maskReady = true;
        tryFinalize();
      }
    };
    mask.onerror = () => {
      console.warn('Mask image failed to load, creating blank fallback mask');
      setBlankFallbackMask();
    };

    base.src = imageSrc;
    if (maskSrc) {
      mask.src = maskSrc;
    } else {
      setBlankFallbackMask();
    }

    return () => {
      active = false;
    };
  }, [imageSrc, maskSrc]);

  // Main Render Engine supporting all dynamic visualization modes
  const renderCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const baseImg = baseImgRef.current;
    const maskImg = maskImgRef.current;

    if (!canvas || !baseImg || !maskImg || !imagesLoaded) {
      return;
    }
    if (!baseImg.complete || baseImg.naturalWidth === 0) {
      return;
    }
    if (!maskImg.complete || maskImg.naturalWidth === 0) {
      return;
    }

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const width = baseImg.naturalWidth || 800;
    const height = baseImg.naturalHeight || 600;

    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    // 1. Read Mask Pixel Data safely
    const offCanvas = document.createElement('canvas');
    offCanvas.width = width;
    offCanvas.height = height;
    const offCtx = offCanvas.getContext('2d', { willReadFrequently: true });
    try {
      offCtx.drawImage(maskImg, 0, 0, width, height);
    } catch (err) {
      console.warn('Could not draw mask element:', err);
      return;
    }

    let maskData;
    try {
      maskData = offCtx.getImageData(0, 0, width, height);
    } catch (e) {
      console.error('Mask read error', e);
      return;
    }

    // 2. Compute camouflaged target coverage percentage
    let camoPixels = 0;
    const totalPixels = width * height;
    const threshVal = threshold * 255;
    for (let i = 0; i < maskData.data.length; i += 4) {
      if (maskData.data[i] >= threshVal) camoPixels++;
    }
    const camoPct = ((camoPixels / totalPixels) * 100).toFixed(1);

    // 3. Extract Sobel Contour Edges
    const contourEdges = extractContour(maskData.data, width, height, threshVal);

    // Helper: Draw Red-Cyan Glow Layer
    const createGlowLayer = () => {
      const gCanvas = document.createElement('canvas');
      gCanvas.width = width;
      gCanvas.height = height;
      const gCtx = gCanvas.getContext('2d');
      const gImgData = gCtx.createImageData(width, height);
      const cyanR = 0, cyanG = 245, cyanB = 212;
      const redR = 255, redG = 0, redB = 85;

      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = y * width + x;
          const dataIdx = idx * 4;
          const maskVal = maskData.data[dataIdx];
          const isEdge = contourEdges[idx] > 0;

          if (maskVal >= threshVal) {
            if (isEdge) {
              gImgData.data[dataIdx] = cyanR;
              gImgData.data[dataIdx + 1] = cyanG;
              gImgData.data[dataIdx + 2] = cyanB;
              gImgData.data[dataIdx + 3] = 255;
            } else {
              gImgData.data[dataIdx] = redR;
              gImgData.data[dataIdx + 1] = redG;
              gImgData.data[dataIdx + 2] = redB;
              gImgData.data[dataIdx + 3] = Math.round(190 * opacity);
            }
          }
        }
      }
      gCtx.putImageData(gImgData, 0, 0);
      return gCanvas;
    };

    // --- MODE 1: RED-CYAN BLOOM ---
    if (visualMode === 'glow') {
      ctx.clearRect(0, 0, width, height);
      ctx.drawImage(baseImgRef.current, 0, 0, width, height);
      const glowCanvas = createGlowLayer();

      ctx.save();
      ctx.globalCompositeOperation = 'screen';
      ctx.shadowColor = '#00f5d4';
      ctx.shadowBlur = glowRadius * 1.8;
      ctx.drawImage(glowCanvas, 0, 0);

      ctx.shadowColor = '#ff0055';
      ctx.shadowBlur = glowRadius * 0.8;
      ctx.drawImage(glowCanvas, 0, 0);

      ctx.shadowBlur = 0;
      ctx.drawImage(glowCanvas, 0, 0);
      ctx.restore();
    }

    // --- MODE 2: THERMAL HEATMAP (FLIR Jet) ---
    else if (visualMode === 'thermal') {
      ctx.clearRect(0, 0, width, height);
      ctx.drawImage(baseImgRef.current, 0, 0, width, height);

      const tCanvas = document.createElement('canvas');
      tCanvas.width = width;
      tCanvas.height = height;
      const tCtx = tCanvas.getContext('2d');
      const tImgData = tCtx.createImageData(width, height);

      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = y * width + x;
          const dataIdx = idx * 4;
          const maskVal = maskData.data[dataIdx];

          if (maskVal >= threshVal) {
            const normalized = (maskVal - threshVal) / (255 - threshVal + 1e-6);
            const { r, g, b } = getThermalColor(normalized);
            tImgData.data[dataIdx] = r;
            tImgData.data[dataIdx + 1] = g;
            tImgData.data[dataIdx + 2] = b;
            tImgData.data[dataIdx + 3] = Math.round(240 * opacity);
          }
        }
      }
      tCtx.putImageData(tImgData, 0, 0);

      ctx.save();
      ctx.globalAlpha = opacity;
      ctx.drawImage(tCanvas, 0, 0);
      ctx.restore();
    }

    // --- MODE 3: CYBER NEON CONTOUR ---
    else if (visualMode === 'contour') {
      ctx.clearRect(0, 0, width, height);
      ctx.drawImage(baseImgRef.current, 0, 0, width, height);

      const cCanvas = document.createElement('canvas');
      cCanvas.width = width;
      cCanvas.height = height;
      const cCtx = cCanvas.getContext('2d');
      const cImgData = cCtx.createImageData(width, height);

      // Electric neon green/cyan edge
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = y * width + x;
          const dataIdx = idx * 4;
          if (contourEdges[idx] > 0) {
            cImgData.data[dataIdx] = 0;
            cImgData.data[dataIdx + 1] = 255;
            cImgData.data[dataIdx + 2] = 170;
            cImgData.data[dataIdx + 3] = 255;
          }
        }
      }
      cCtx.putImageData(cImgData, 0, 0);

      ctx.save();
      ctx.shadowColor = '#00ffaa';
      ctx.shadowBlur = glowRadius * 2;
      ctx.drawImage(cCanvas, 0, 0);
      ctx.shadowBlur = glowRadius;
      ctx.drawImage(cCanvas, 0, 0);
      ctx.shadowBlur = 0;
      ctx.drawImage(cCanvas, 0, 0);
      ctx.restore();
    }

    // --- MODE 4: SPOTLIGHT FOCUS ---
    else if (visualMode === 'spotlight') {
      // 1. Draw grayscale darkened background
      ctx.clearRect(0, 0, width, height);
      const bCanvas = document.createElement('canvas');
      bCanvas.width = width;
      bCanvas.height = height;
      const bCtx = bCanvas.getContext('2d');
      bCtx.drawImage(baseImgRef.current, 0, 0, width, height);
      const bImgData = bCtx.getImageData(0, 0, width, height);

      for (let i = 0; i < bImgData.data.length; i += 4) {
        const r = bImgData.data[i];
        const g = bImgData.data[i + 1];
        const b = bImgData.data[i + 2];
        const maskVal = maskData.data[i];
        
        // If not in mask, desaturate & darken by 70%
        if (maskVal < threshVal) {
          const gray = (0.299 * r + 0.587 * g + 0.114 * b) * 0.25;
          bImgData.data[i] = gray;
          bImgData.data[i + 1] = gray;
          bImgData.data[i + 2] = gray;
        }
      }
      ctx.putImageData(bImgData, 0, 0);

      // Add a subtle rim highlight around the spotlighted subject
      const edgeCanvas = document.createElement('canvas');
      edgeCanvas.width = width;
      edgeCanvas.height = height;
      const eCtx = edgeCanvas.getContext('2d');
      const eData = eCtx.createImageData(width, height);
      for (let i = 0; i < contourEdges.length; i++) {
        if (contourEdges[i] > 0) {
          eData.data[i * 4] = 255;
          eData.data[i * 4 + 1] = 215;
          eData.data[i * 4 + 2] = 0;
          eData.data[i * 4 + 3] = 220;
        }
      }
      eCtx.putImageData(eData, 0, 0);
      ctx.save();
      ctx.shadowColor = '#ffd700';
      ctx.shadowBlur = 10;
      ctx.drawImage(edgeCanvas, 0, 0);
      ctx.restore();
    }

    // --- MODE 5: TARGET CUTOUT ---
    else if (visualMode === 'cutout') {
      ctx.clearRect(0, 0, width, height);

      // Draw dark checkerboard
      const size = 16;
      for (let y = 0; y < height; y += size) {
        for (let x = 0; x < width; x += size) {
          ctx.fillStyle = ((x / size + y / size) % 2 === 0) ? '#0f1420' : '#070a10';
          ctx.fillRect(x, y, size, size);
        }
      }

      // Draw base image only where mask >= threshVal
      const cutCanvas = document.createElement('canvas');
      cutCanvas.width = width;
      cutCanvas.height = height;
      const cutCtx = cutCanvas.getContext('2d');
      cutCtx.drawImage(baseImgRef.current, 0, 0, width, height);
      const cutImgData = cutCtx.getImageData(0, 0, width, height);

      for (let i = 0; i < cutImgData.data.length; i += 4) {
        if (maskData.data[i] < threshVal) {
          cutImgData.data[i + 3] = 0; // Transparent
        }
      }
      cutCtx.putImageData(cutImgData, 0, 0);

      ctx.save();
      ctx.shadowColor = 'rgba(0, 245, 212, 0.4)';
      ctx.shadowBlur = 20;
      ctx.drawImage(cutCanvas, 0, 0);
      ctx.restore();
    }

    // --- MODE 6: SPLIT CURTAIN SWIPE ---
    else if (visualMode === 'curtain') {
      ctx.clearRect(0, 0, width, height);
      const splitX = Math.round((width * splitPosition) / 100);

      // 1. Draw raw base image on the left side
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, splitX, height);
      ctx.clip();
      ctx.drawImage(baseImgRef.current, 0, 0, width, height);
      ctx.restore();

      // 2. Draw overlay on the right side
      ctx.save();
      ctx.beginPath();
      ctx.rect(splitX, 0, width - splitX, height);
      ctx.clip();
      ctx.drawImage(baseImgRef.current, 0, 0, width, height);

      const glowCanvas = createGlowLayer();
      ctx.globalCompositeOperation = 'screen';
      ctx.shadowColor = '#00f5d4';
      ctx.shadowBlur = glowRadius * 1.5;
      ctx.drawImage(glowCanvas, 0, 0);
      ctx.restore();

      // 3. Draw neon vertical divider line
      ctx.save();
      ctx.strokeStyle = '#00f5d4';
      ctx.lineWidth = 3;
      ctx.shadowColor = '#00f5d4';
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.moveTo(splitX, 0);
      ctx.lineTo(splitX, height);
      ctx.stroke();

      // Draw slider handle button in the center
      const handleY = height / 2;
      ctx.fillStyle = '#070a10';
      ctx.strokeStyle = '#00f5d4';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(splitX, handleY, 18, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Draw arrows in the handle
      ctx.fillStyle = '#00f5d4';
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('◄►', splitX, handleY);
      ctx.restore();
    }

    setStats(prev => ({ ...prev, camouflagedPercent: camoPct }));
  }, [imagesLoaded, opacity, glowRadius, threshold, visualMode, splitPosition]);

  useEffect(() => {
    renderCanvas();
  }, [renderCanvas]);

  // Direct canvas drag interaction for Curtain mode
  const handleCanvasMouseDown = (e) => {
    if (visualMode !== 'curtain' || !setSplitPosition) return;
    setIsDraggingCurtain(true);
    updateCurtainFromMouse(e);
  };

  const handleCanvasMouseMove = (e) => {
    if (!isDraggingCurtain || visualMode !== 'curtain' || !setSplitPosition) return;
    updateCurtainFromMouse(e);
  };

  const handleCanvasMouseUp = () => {
    setIsDraggingCurtain(false);
  };

  const updateCurtainFromMouse = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSplitPosition(pct);
  };

  // Stage Drag & Drop Handlers
  const handleStageDragOver = (e) => {
    e.preventDefault();
    setIsDraggingOver(true);
  };

  const handleStageDragLeave = (e) => {
    e.preventDefault();
    setIsDraggingOver(false);
  };

  const handleStageDrop = (e) => {
    e.preventDefault();
    setIsDraggingOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/') && onDropImage) {
      const reader = new FileReader();
      reader.onload = (event) => {
        onDropImage(event.target.result, file.name);
      };
      reader.readAsDataURL(file);
    }
  };

  const exportHighResPNG = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const link = document.createElement('a');
    link.download = `fapnet_${visualMode}_${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
    if (onSnapshot) onSnapshot();
  };

  return (
    <main 
      className={`viewport-container ${isDraggingOver ? 'drag-active' : ''}`} 
      ref={containerRef}
      onDragOver={handleStageDragOver}
      onDragLeave={handleStageDragLeave}
      onDrop={handleStageDrop}
    >
      {/* Dropzone drag indicator */}
      {isDraggingOver && (
        <div className="canvas-drop-backdrop">
          <div className="canvas-drop-modal">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <h2>Drop Image to Detect Camouflage</h2>
            <p>FAPNet will immediately run inference on CUDA</p>
          </div>
        </div>
      )}

      {/* Floating HUD Badges */}
      <div className="viewport-hud-top">
        <div className="hud-badge mode-indicator">
          <span className="dot pulse"></span>
          <span>MODE: {visualMode.toUpperCase()}</span>
        </div>
        <div className="hud-badge resolution">
          <span>{stats.width} × {stats.height} PX</span>
        </div>
        <div className="hud-badge coverage">
          <span>TARGET: {stats.camouflagedPercent}% COVERAGE</span>
        </div>
      </div>

      {/* Interactive Main Canvas */}
      <div 
        className="canvas-wrapper" 
        style={{ transform: `scale(${zoom})` }}
        onMouseDown={handleCanvasMouseDown}
        onMouseMove={handleCanvasMouseMove}
        onMouseUp={handleCanvasMouseUp}
        onMouseLeave={handleCanvasMouseUp}
      >
        <canvas 
          ref={canvasRef} 
          className={`main-shader-canvas ${visualMode === 'curtain' ? 'curtain-cursor' : ''}`} 
        />
      </div>

      {/* Viewport Floating Footer Controls */}
      <div className="viewport-floating-footer">
        <div className="zoom-controls">
          <button onClick={() => setZoom(z => Math.max(0.4, z - 0.15))}>−</button>
          <span className="zoom-val">{Math.round(zoom * 100)}%</span>
          <button onClick={() => setZoom(z => Math.min(2.5, z + 0.15))}>+</button>
          <button className="reset-btn" onClick={() => setZoom(1)}>Reset</button>
        </div>

        {/* Upload Picture Button */}
        {onTriggerUpload && (
          <button className="upload-quick-btn" onClick={onTriggerUpload} title="Upload a new picture">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <span>Upload New Picture</span>
          </button>
        )}

        <button className="export-hd-btn" onClick={exportHighResPNG}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="7 10 12 15 17 10"/>
            <line x1="12" y1="15" x2="12" y2="3"/>
          </svg>
          Export {visualMode.toUpperCase()} Result (PNG)
        </button>
      </div>
    </main>
  );
}
