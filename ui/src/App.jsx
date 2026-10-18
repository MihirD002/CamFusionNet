import React, { useState, useEffect, useCallback, useRef } from 'react';
import Header from './components/Header';
import ViewportCanvas from './components/ViewportCanvas';
import ControlPanel from './components/ControlPanel';
import { PRESET_SAMPLES } from './utils/presets';
import { generateApproximateMask } from './utils/colormap';
import './App.css';

function App() {
  // Preset or Custom Image State
  const [activePresetId, setActivePresetId] = useState(PRESET_SAMPLES[0].id);
  const [currentImage, setCurrentImage] = useState(PRESET_SAMPLES[0].image);
  const [currentMask, setCurrentMask] = useState(PRESET_SAMPLES[0].mask);
  const [hasCustomMask, setHasCustomMask] = useState(false);

  // Dynamic Visualization Modes: 'glow' | 'thermal' | 'contour' | 'spotlight' | 'cutout' | 'curtain'
  const [visualMode, setVisualMode] = useState('glow');
  const [splitPosition, setSplitPosition] = useState(50);

  // Minimal Overlay Parameters
  const [opacity, setOpacity] = useState(0.85);
  const [glowRadius, setGlowRadius] = useState(16);
  const [threshold, setThreshold] = useState(0.40);

  // Backend connection state
  const [serverOnline, setServerOnline] = useState(false);
  const [isInferencing, setIsInferencing] = useState(false);
  const [notification, setNotification] = useState(null);

  const showToast = (msg, type = 'info') => {
    setNotification({ msg, type });
    setTimeout(() => {
      setNotification(null);
    }, 4000);
  };

  // Check Python backend connection (uses Vite proxy → localhost:8000)
  const checkServer = useCallback(async () => {
    try {
      const res = await fetch('/api/health', { method: 'GET', signal: AbortSignal.timeout(2000) });
      if (res.ok) {
        const data = await res.json();
        setServerOnline(true);
        return true;
      }
    } catch (e) {
      setServerOnline(false);
    }
    return false;
  }, []);

  // Poll every 5 seconds to auto-detect server going online/offline
  useEffect(() => {
    checkServer();
    const interval = setInterval(checkServer, 5000);
    return () => clearInterval(interval);
  }, [checkServer]);


  // Handle Preset Selection
  const handleSelectPreset = (id) => {
    const preset = PRESET_SAMPLES.find(p => p.id === id);
    if (preset) {
      setActivePresetId(id);
      setCurrentImage(preset.image);
      setCurrentMask(preset.mask);
      setHasCustomMask(false);
      showToast(`Loaded preset: ${preset.title}`);
    }
  };

  // Ref for global file upload input
  const fileInputRef = useRef(null);

  const handleTriggerUpload = () => {
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
      fileInputRef.current.click();
    }
  };

  // Trigger FAPNet inference with specific image data URL
  const runModelInference = async (imgDataUrl) => {
    if (!imgDataUrl) return;
    setIsInferencing(true);
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: imgDataUrl })
      });

      if (res.ok) {
        const data = await res.json();
        if (data.mask_base64) {
          setCurrentMask(`data:image/png;base64,${data.mask_base64}`);
          setHasCustomMask(true);
          setServerOnline(true);
          showToast(`✓ FAPNet (ConvNeXt)  detected in ${data.inference_time_ms || 120}ms on ${data.device || 'GPU'}`, 'success');
        }
      } else {
        showToast('Inference returned error — check server console', 'warning');
      }
    } catch (err) {
      console.warn('Backend inference unreachable:', err);
      setServerOnline(false);
      showToast('Backend offline — using client-side edge saliency', 'warning');
      // Client-side fallback preview (downscaled to avoid freezing UI thread)
      generateFastClientMask(imgDataUrl);
    } finally {
      setIsInferencing(false);
    }
  };

  // Fast client-side fallback mask generated on a downscaled canvas
  const generateFastClientMask = (dataUrl) => {
    const img = new Image();
    img.onload = () => {
      const maxDim = 256;
      const scale = Math.min(1, maxDim / Math.max(img.naturalWidth || 1, img.naturalHeight || 1));
      const w = Math.max(16, Math.round((img.naturalWidth || 800) * scale));
      const h = Math.max(16, Math.round((img.naturalHeight || 600) * scale));

      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, w, h);
      const imgData = ctx.getImageData(0, 0, w, h);
      const maskData = generateApproximateMask(imgData, w, h);

      const maskImgData = ctx.createImageData(w, h);
      maskImgData.data.set(maskData);
      ctx.putImageData(maskImgData, 0, 0);

      setCurrentMask(canvas.toDataURL('image/png'));
    };
    img.src = dataUrl;
  };

  // Handle Custom Image Upload (supports unlimited subsequent uploads)
  const handleImageUpload = (dataUrl, fileName = 'custom_image.png') => {
    setActivePresetId(null);
    setCurrentImage(dataUrl);
    setCurrentMask(null); // Clear previous mask immediately so no stale mask lingers
    setHasCustomMask(false);

    showToast(`Loaded ${fileName}. Running neural detection...`, 'info');
    runModelInference(dataUrl);
  };

  // Handle Ground Truth Mask Upload
  const handleMaskUpload = (dataUrl, fileName) => {
    setCurrentMask(dataUrl);
    setHasCustomMask(true);
    showToast(`Custom mask applied from ${fileName}`, 'success');
  };

  return (
    <div className="app-container">
      {/* Hidden Master File Input for Seamless Upload from Anywhere */}
      <input 
        type="file" 
        ref={fileInputRef} 
        style={{ display: 'none' }} 
        accept="image/*"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) {
            const reader = new FileReader();
            reader.onload = (ev) => handleImageUpload(ev.target.result, file.name);
            reader.readAsDataURL(file);
          }
          e.target.value = '';
        }}
      />

      {/* Top Navigation */}
      <Header 
        serverOnline={serverOnline}
        onTestServer={checkServer}
        isInferencing={isInferencing}
        onTriggerUpload={handleTriggerUpload}
      />

      {/* Main Workspace */}
      <div className="studio-layout">
        {/* Left Interactive Canvas Stage */}
        <ViewportCanvas 
          imageSrc={currentImage}
          maskSrc={currentMask}
          opacity={opacity}
          glowRadius={glowRadius}
          threshold={threshold}
          visualMode={visualMode}
          splitPosition={splitPosition}
          setSplitPosition={setSplitPosition}
          onSnapshot={() => showToast('High-res result exported!', 'success')}
          onDropImage={handleImageUpload}
          onTriggerUpload={handleTriggerUpload}
        />

        {/* Right Minimal Sidebar */}
        <ControlPanel 
          opacity={opacity}
          setOpacity={setOpacity}
          glowRadius={glowRadius}
          setGlowRadius={setGlowRadius}
          threshold={threshold}
          setThreshold={setThreshold}
          visualMode={visualMode}
          setVisualMode={setVisualMode}
          splitPosition={splitPosition}
          setSplitPosition={setSplitPosition}
          activePresetId={activePresetId}
          onSelectPreset={handleSelectPreset}
          onImageUpload={handleImageUpload}
          onMaskUpload={handleMaskUpload}
          serverOnline={serverOnline}
          onRunInference={() => runModelInference(currentImage)}
          isInferencing={isInferencing}
          hasCustomMask={hasCustomMask}
          onTriggerUpload={handleTriggerUpload}
        />
      </div>

      {/* Global Toast Notification */}
      {notification && (
        <div className={`toast-notification ${notification.type}`}>
          <div className="toast-icon">
            {notification.type === 'success' && '✓'}
            {notification.type === 'error' && '✕'}
            {notification.type === 'warning' && '⚠️'}
            {notification.type === 'info' && '⚡'}
          </div>
          <span className="toast-msg">{notification.msg}</span>
        </div>
      )}
    </div>
  );
}

export default App;
