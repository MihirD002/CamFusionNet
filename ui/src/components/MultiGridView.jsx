import React from 'react';

export default function MultiGridView({ imageSrc, maskSrc, paletteKey, threshold }) {
  return (
    <div className="multi-grid-container">
      <div className="grid-cell">
        <div className="grid-header">
          <span className="grid-index">01</span>
          <span className="grid-title">RAW INPUT IMAGE</span>
          <span className="grid-tag">Original RGB</span>
        </div>
        <div className="grid-img-wrap">
          <img src={imageSrc} alt="Raw Input" className="grid-img" />
          <div className="grid-overlay-label">Hidden Subject Embedded</div>
        </div>
      </div>

      <div className="grid-cell">
        <div className="grid-header">
          <span className="grid-index">02</span>
          <span className="grid-title">PREDICTED ATTENTION MASK</span>
          <span className="grid-tag">Sigmoid Output</span>
        </div>
        <div className="grid-img-wrap">
          <img src={maskSrc} alt="Predicted Mask" className="grid-img mask-invert" />
          <div className="grid-overlay-label">Feature Aggregation Peak</div>
        </div>
      </div>

      <div className="grid-cell">
        <div className="grid-header">
          <span className="grid-index">03</span>
          <span className="grid-title">THERMAL SALIENCY HEATMAP</span>
          <span className="grid-tag">Contour Analysis</span>
        </div>
        <div className="grid-img-wrap thermal-wrap">
          <img src={maskSrc} alt="Heatmap" className="grid-img thermal-filter" />
          <div className="grid-overlay-label">Multi-Scale Boundary Gradient</div>
        </div>
      </div>

      <div className="grid-cell highlight-cell">
        <div className="grid-header">
          <span className="grid-index">04</span>
          <span className="grid-title">RED-CYAN GLOW COMPOSITE</span>
          <span className="grid-tag highlight">Unmasked Reveal</span>
        </div>
        <div className="grid-img-wrap composite-wrap">
          <img src={imageSrc} alt="Base" className="grid-img composite-base" />
          <img src={maskSrc} alt="Overlay" className="grid-img composite-mask" />
          <div className="grid-overlay-label">Holographic Neon Fusion</div>
        </div>
      </div>
    </div>
  );
}
