import { PALETTES, MODES } from './NeonCanvasOverlay';

export default function ControlsPanel({
  palette, onPaletteChange,
  threshold, onThresholdChange,
  glowRadius, onGlowRadiusChange,
  glowOpacity, onGlowOpacityChange,
  pulseOn, onPulseToggle,
  mode, onModeChange,
}) {

  // ── Slider helper ───────────────────────────────────────────────
  function Slider({ label, value, min, max, step = 0.01, onChange, format }) {
    const pct = ((value - min) / (max - min)) * 100;
    return (
      <div className="slider-track">
        <div className="slider-header">
          <span className="slider-label">{label}</span>
          <span className="slider-value">{format ? format(value) : value}</span>
        </div>
        <input
          type="range"
          min={min} max={max} step={step}
          value={value}
          style={{ '--pct': `${pct}%` }}
          onChange={e => onChange(parseFloat(e.target.value))}
        />
      </div>
    );
  }

  return (
    <>
      {/* ── Mode tabs ── */}
      <div>
        <p className="section-label">Render Mode</p>
        <div className="mode-tabs">
          {[
            { id: MODES.overlay,   icon: '🎨', title: 'Overlay' },
            { id: MODES.split,     icon: '⚡', title: 'Split' },
            { id: MODES.spotlight, icon: '🔦', title: 'Scan' },
            { id: MODES.extract,   icon: '🔬', title: 'Extract' },
          ].map(m => (
            <button
              key={m.id}
              className={`mode-tab${mode === m.id ? ' active' : ''}`}
              onClick={() => onModeChange(m.id)}
              title={m.title}
            >
              {m.icon} {m.title}
            </button>
          ))}
        </div>
      </div>

      <div className="divider" />

      {/* ── Neon Palette ── */}
      <div>
        <p className="section-label">Neon Palette</p>
        <div className="palette-grid">
          {Object.entries(PALETTES).map(([key, pal]) => (
            <div
              key={key}
              className={`palette-chip${palette === pal ? ' active' : ''}`}
              style={{ '--chip-color': pal.chipColor }}
              onClick={() => onPaletteChange(pal)}
            >
              <div
                className="palette-dot"
                style={{
                  background: `radial-gradient(circle, ${pal.chipColor}, ${pal.chipColor}88)`,
                  boxShadow: palette === pal ? `0 0 10px ${pal.chipColor}` : 'none',
                }}
              />
              <span style={{ color: palette === pal ? pal.chipColor : 'var(--text-secondary)' }}>
                {pal.name}
              </span>
              {palette === pal && (
                <span style={{ marginLeft: 'auto', fontSize: 8, color: pal.chipColor }}>●</span>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="divider" />

      {/* ── Signal Controls ── */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
        <p className="section-label" style={{ marginBottom: 0 }}>Signal Controls</p>

        <Slider
          label="Detection Threshold"
          value={threshold}
          min={0.05} max={0.95}
          format={v => v.toFixed(2)}
          onChange={onThresholdChange}
        />

        <Slider
          label="Glow Radius"
          value={glowRadius}
          min={1} max={20} step={0.5}
          format={v => `${v}px`}
          onChange={onGlowRadiusChange}
        />

        <Slider
          label="Glow Intensity"
          value={glowOpacity}
          min={0.1} max={1.0}
          format={v => `${Math.round(v * 100)}%`}
          onChange={onGlowOpacityChange}
        />
      </div>

      <div className="divider" />

      {/* ── Pulse Aura Toggle ── */}
      <div>
        <p className="section-label">Radar Pulse</p>
        <label className="toggle-wrap">
          <span className="toggle">
            <input type="checkbox" checked={pulseOn} onChange={e => onPulseToggle(e.target.checked)} />
            <span className="toggle-slider" />
          </span>
          <span style={{ fontSize: 13, color: pulseOn ? 'var(--neon-cyan)' : 'var(--text-muted)' }}>
            {pulseOn ? 'Radar Active' : 'Radar Off'}
          </span>
        </label>
        {pulseOn && (
          <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6, lineHeight: 1.5 }}>
            Pulsing centroid aura tracks the center of the detected camouflaged region.
          </p>
        )}
      </div>

      {/* Mode hints */}
      <div className="glass-card" style={{ padding: 12, fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6 }}>
        {mode === MODES.overlay   && <><b style={{ color:'var(--neon-cyan)' }}>Overlay Mode</b> – Full neon contour + Red-Cyan chromatic blend over image.</>}
        {mode === MODES.split     && <><b style={{ color:'var(--neon-cyan)' }}>Split Mode</b> – Drag on canvas to sweep between original and overlay.</>}
        {mode === MODES.spotlight && <><b style={{ color:'var(--neon-cyan)' }}>Scan Mode</b> – Move cursor over canvas — spotlight reveals hidden camouflage.</>}
        {mode === MODES.extract   && <><b style={{ color:'var(--neon-cyan)' }}>Extract Mode</b> – Isolates detected object on transparent background below.</>}
      </div>
    </>
  );
}
