import { useRef, useState, useCallback } from 'react';

// Relative paths to sample images from the CHAMELEON dataset
// served by the Python backend at /api/preset-image?file=...
const SAMPLE_PRESETS = [
  { id: 'animal-1',  label: 'Chameleon 01', file: 'animal-1.jpg' },
  { id: 'animal-4',  label: 'Chameleon 04', file: 'animal-4.jpg' },
  { id: 'animal-7',  label: 'Chameleon 07', file: 'animal-7.jpg' },
  { id: 'animal-20', label: 'Chameleon 20', file: 'animal-20.jpg' },
  { id: 'animal-30', label: 'Chameleon 30', file: 'animal-30.jpg' },
  { id: 'animal-45', label: 'Chameleon 45', file: 'animal-45.jpg' },
];

export default function ImageUploader({ onImageLoaded, activePresetId, onPresetSelect, backendOnline }) {
  const fileInputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [thumbURLs, setThumbURLs] = useState({});

  // ── File handling ──────────────────────────────────────────────
  const loadFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      onImageLoaded({ img, file, url, name: file.name, isCustom: true });
    };
    img.src = url;
  }, [onImageLoaded]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    loadFile(e.dataTransfer.files[0]);
  }, [loadFile]);

  const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = () => setIsDragging(false);

  // ── Preset selection ───────────────────────────────────────────
  const handlePreset = useCallback((preset) => {
    onPresetSelect(preset.id);
    const url = backendOnline
      ? `http://localhost:5050/api/preset-image?file=${preset.file}`
      : `https://picsum.photos/seed/${preset.id}/480/360`;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      onImageLoaded({ img, url, name: preset.label, isCustom: false, presetFile: preset.file });
    };
    img.src = url;

    // Thumb
    setThumbURLs(prev => ({ ...prev, [preset.id]: url }));
  }, [backendOnline, onImageLoaded, onPresetSelect]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

      {/* ── Drag-and-drop zone ── */}
      <div
        className={`upload-zone${isDragging ? ' drag-over' : ''}`}
        onClick={() => fileInputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <span className="upload-icon">📂</span>
        <p style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>
          Drop your image here
        </p>
        <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          PNG · JPG · WebP — or click to browse
        </p>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          style={{ display: 'none' }}
          onChange={e => loadFile(e.target.files[0])}
        />
      </div>

      {/* ── Preset Samples ── */}
      <div>
        <p className="section-label">Sample Presets — CHAMELEON</p>
        <div className="preset-grid">
          {SAMPLE_PRESETS.map(preset => (
            <div
              key={preset.id}
              className={`preset-thumb${activePresetId === preset.id ? ' active' : ''}`}
              onClick={() => handlePreset(preset)}
              title={preset.label}
            >
              {thumbURLs[preset.id] ? (
                <img src={thumbURLs[preset.id]} alt={preset.label} />
              ) : (
                <div style={{
                  width: '100%', height: '100%', minHeight: 64,
                  background: `hsl(${parseInt(preset.id.split('-')[1]) * 42}, 25%, 12%)`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 22, color: 'rgba(255,255,255,0.2)'
                }}>🦎</div>
              )}
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
