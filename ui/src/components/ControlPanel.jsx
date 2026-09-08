import React, { useRef, useState } from 'react';
import { PRESET_SAMPLES } from '../utils/presets';

export const VISUALIZATION_MODES = [
  {
    id: 'glow',
    name: 'Red-Cyan Bloom',
    icon: '🔮',
    tag: 'Recommended',
    desc: 'Holographic cyan contour with crimson heat core & bloom'
  },
  {
    id: 'thermal',
    name: 'Thermal Heatmap',
    icon: '🌡️',
    tag: 'FLIR Jet',
    desc: 'Full-spectrum military infrared confidence grading'
  },
  {
    id: 'contour',
    name: 'Cyber Neon Contour',
    icon: '⚡',
    tag: 'Edge Only',
    desc: 'High-contrast glowing neon edge outlining hidden borders'
  },
  {
    id: 'spotlight',
    name: 'Spotlight Focus',
    icon: '🔦',
    tag: 'Highlight',
    desc: 'Desaturates background & illuminates target in full color'
  },
  {
    id: 'cutout',
    name: 'Target Cutout',
    icon: '✂️',
    tag: 'Isolate',
    desc: 'Strips away background to isolate the camouflaged subject'
  },
  {
    id: 'curtain',
    name: 'Split Curtain Swipe',
    icon: '↔️',
    tag: 'Before/After',
    desc: 'Interactive sliding divider comparing raw vs detected'
  }
];

export default function ControlPanel({
  opacity,
  setOpacity,
  glowRadius,
  setGlowRadius,
  threshold,
  setThreshold,
  visualMode,
  setVisualMode,
  splitPosition,
  setSplitPosition,
  activePresetId,
  onSelectPreset,
  onImageUpload,
  onMaskUpload,
  serverOnline,
  onRunInference,
  isInferencing,
  hasCustomMask,
  onTriggerUpload
}) {
  const imageInputRef = useRef(null);
  const maskInputRef = useRef(null);
  const [isDraggingImage, setIsDraggingImage] = useState(false);
  const [isDraggingMask, setIsDraggingMask] = useState(false);

  const processImageFile = (file) => {
    if (!file || !file.type.startsWith('image/')) {
      alert('Please upload a valid image file (JPG, PNG, WEBP, etc.)');
      return;
    }
    const reader = new FileReader();
    reader.onload = (event) => {
      onImageUpload(event.target.result, file.name);
    };
    reader.readAsDataURL(file);
  };

  const processMaskFile = (file) => {
    if (!file || !file.type.startsWith('image/')) {
      alert('Please upload a valid image file for the mask');
      return;
    }
    const reader = new FileReader();
    reader.onload = (event) => {
      onMaskUpload(event.target.result, file.name);
    };
    reader.readAsDataURL(file);
  };

  const handleImageChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      processImageFile(file);
    }
    e.target.value = '';
  };

  const handleMaskChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      processMaskFile(file);
    }
    e.target.value = '';
  };

  const isCustomImageActive = activePresetId === null;

  return (
    <aside className="control-sidebar minimal">
      {/* 1. Custom Image Upload */}
      <section className="panel-section">
        <div className="section-header-flex">
          <h2 className="section-title">CUSTOM INPUT</h2>
          {isCustomImageActive && (
            <button 
              className="reset-preset-link" 
              onClick={() => onSelectPreset(PRESET_SAMPLES[0].id)}
              title="Return to default sample presets"
            >
              ↺ Reset to Sample
            </button>
          )}
        </div>

        <div className="upload-deck">
          {/* Main Photo Dropzone */}
          <div 
            className={`upload-dropzone primary-zone ${isCustomImageActive ? 'loaded' : ''} ${isDraggingImage ? 'dragging' : ''}`} 
            onClick={() => {
              if (onTriggerUpload) {
                onTriggerUpload();
              } else if (imageInputRef.current) {
                imageInputRef.current.value = '';
                imageInputRef.current.click();
              }
            }}
            onDragOver={(e) => { e.preventDefault(); setIsDraggingImage(true); }}
            onDragLeave={(e) => { e.preventDefault(); setIsDraggingImage(false); }}
            onDrop={(e) => {
              e.preventDefault();
              setIsDraggingImage(false);
              const file = e.dataTransfer.files?.[0];
              if (file) processImageFile(file);
            }}
          >
            <input 
              type="file" 
              ref={imageInputRef} 
              style={{ display: 'none' }} 
              accept="image/*"
              onClick={(e) => e.stopPropagation()}
              onChange={handleImageChange}
            />
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
              <circle cx="8.5" cy="8.5" r="1.5"/>
              <polyline points="21 15 16 10 5 21"/>
            </svg>
            <span className="upload-title">
              {isCustomImageActive ? 'Upload / Replace Picture' : 'Click or Drop Custom Image'}
            </span>
            <span className="upload-subtitle">
              {isCustomImageActive ? 'Click anytime to load another picture' : 'Auto-triggers FAPNet neural inference'}
            </span>
          </div>

          {/* Optional Mask Dropzone */}
          <div 
            className={`upload-dropzone secondary-zone ${hasCustomMask ? 'loaded' : ''} ${isDraggingMask ? 'dragging' : ''}`} 
            onClick={() => {
              if (maskInputRef.current) {
                maskInputRef.current.value = '';
                maskInputRef.current.click();
              }
            }}
            onDragOver={(e) => { e.preventDefault(); setIsDraggingMask(true); }}
            onDragLeave={(e) => { e.preventDefault(); setIsDraggingMask(false); }}
            onDrop={(e) => {
              e.preventDefault();
              setIsDraggingMask(false);
              const file = e.dataTransfer.files?.[0];
              if (file) processMaskFile(file);
            }}
          >
            <input 
              type="file" 
              ref={maskInputRef} 
              style={{ display: 'none' }} 
              accept="image/*"
              onClick={(e) => e.stopPropagation()}
              onChange={handleMaskChange}
            />
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
            </svg>
            <span className="upload-title">{hasCustomMask ? 'Mask Loaded ✓' : 'Upload Ground Truth (Optional)'}</span>
            <span className="upload-subtitle">Grayscale PNG mask</span>
          </div>
        </div>

        {/* Manual Re-Inference Button */}
        <button 
          className={`run-inference-btn ${isInferencing ? 'loading' : ''}`}
          onClick={onRunInference}
          disabled={isInferencing}
        >
          {isInferencing ? '⚡ Running FAPNet Neural Model...' : '🔮 Run Camouflage Detection'}
        </button>
      </section>

      {/* 2. Dynamic Visualization Selector */}
      <section className="panel-section">
        <div className="section-header-flex">
          <h2 className="section-title">VISUALIZATION MODE</h2>
          <span className="active-mode-badge">{VISUALIZATION_MODES.find(m => m.id === visualMode)?.name}</span>
        </div>
        
        <div className="vis-modes-grid">
          {VISUALIZATION_MODES.map(mode => (
            <button
              key={mode.id}
              className={`vis-mode-card ${visualMode === mode.id ? 'active' : ''}`}
              onClick={() => setVisualMode(mode.id)}
            >
              <div className="vis-card-top">
                <span className="vis-icon">{mode.icon}</span>
                <span className="vis-tag">{mode.tag}</span>
              </div>
              <span className="vis-name">{mode.name}</span>
              <span className="vis-desc">{mode.desc}</span>
            </button>
          ))}
        </div>

        {/* Curtain Split Slider (Visible only in Split Curtain mode) */}
        {visualMode === 'curtain' && (
          <div className="slider-group curtain-slider-box">
            <div className="slider-header">
              <label>↔️ Split Reveal Position</label>
              <span className="slider-val">{Math.round(splitPosition)}%</span>
            </div>
            <input 
              type="range" 
              min="0" 
              max="100" 
              step="1" 
              value={splitPosition}
              onChange={e => setSplitPosition(parseFloat(e.target.value))}
            />
            <div className="split-labels">
              <span>Original Photo</span>
              <span>Detected Camouflage</span>
            </div>
          </div>
        )}
      </section>

      {/* 3. Quick Sample Presets */}
      <section className="panel-section">
        <h2 className="section-title">QUICK SAMPLES</h2>
        <div className="preset-cards-grid">
          {PRESET_SAMPLES.map(p => (
            <button
              key={p.id}
              className={`preset-card ${activePresetId === p.id ? 'active' : ''}`}
              onClick={() => onSelectPreset(p.id)}
            >
              <img src={p.image} alt={p.title} className="preset-thumb" />
              <div className="preset-info">
                <span className="preset-title">{p.title}</span>
                <span className="tag rate">{p.detectionRate} detected</span>
              </div>
            </button>
          ))}
        </div>
      </section>

      {/* 4. Shader & Sensitivity Fine Tuning */}
      <section className="panel-section">
        <h2 className="section-title">SHADER ADJUSTMENTS</h2>

        <div className="controls-list">
          {/* Opacity */}
          <div className="slider-group">
            <div className="slider-header">
              <label>Overlay Opacity</label>
              <span className="slider-val">{Math.round(opacity * 100)}%</span>
            </div>
            <input 
              type="range" 
              min="0.1" 
              max="1.0" 
              step="0.02" 
              value={opacity}
              onChange={e => setOpacity(parseFloat(e.target.value))}
            />
          </div>

          {/* Glow Radius */}
          <div className="slider-group">
            <div className="slider-header">
              <label>Bloom & Glow Radius</label>
              <span className="slider-val">{glowRadius}px</span>
            </div>
            <input 
              type="range" 
              min="4" 
              max="35" 
              step="1" 
              value={glowRadius}
              onChange={e => setGlowRadius(parseInt(e.target.value))}
            />
          </div>

          {/* Detection Sensitivity */}
          <div className="slider-group">
            <div className="slider-header">
              <label>Confidence Threshold</label>
              <span className="slider-val">{threshold.toFixed(2)}</span>
            </div>
            <input 
              type="range" 
              min="0.1" 
              max="0.9" 
              step="0.02" 
              value={threshold}
              onChange={e => setThreshold(parseFloat(e.target.value))}
            />
          </div>
        </div>
      </section>
    </aside>
  );
}
