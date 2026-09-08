import React from 'react';

export default function Header({ serverOnline, onTestServer, isInferencing, onTriggerUpload }) {
  return (
    <header className="app-header">
      <div className="header-left">
        <div className="brand-badge">
          <span className="brand-dot"></span>
          <span className="brand-tag">FAPNet</span>
        </div>
        <div className="title-group">
          <h1>Camouflaged Object Vision</h1>
          <p className="subtitle">Red-Cyan / Color Blended Overlay</p>
        </div>
      </div>

      <div className="header-right">
        {/* Quick Upload Button directly in navigation */}
        <button 
          className="header-btn upload-header-btn"
          onClick={onTriggerUpload}
          title="Upload new picture (JPG, PNG, WEBP)"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="17 8 12 3 7 8"/>
            <line x1="12" y1="3" x2="12" y2="15"/>
          </svg>
          <span>Upload Image</span>
        </button>

        <div className="status-pill">
          <span className={`status-indicator ${serverOnline ? 'online' : 'standalone'}`}></span>
          <span className="status-text">
            {serverOnline ? 'Backend Online' : 'Canvas Engine'}
          </span>
          {isInferencing && <span className="inferencing-spinner">⚡ Processing...</span>}
        </div>

        <button 
          className="header-btn" 
          onClick={onTestServer}
          title="Test backend connection"
        >
          {serverOnline ? 'Connected' : 'Check Server'}
        </button>
      </div>
    </header>
  );
}
