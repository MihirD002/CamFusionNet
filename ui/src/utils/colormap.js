// Colormap and image processing shaders for glowing neon and heat contour overlays

export const PALETTES = {
  'red-cyan': {
    name: 'Red-Cyan Hologram',
    glowColor: '#00f5d4', // Neon Cyan Outline
    fillColor: '#ff0055', // Crimson Red Core
    accentColor: '#00bbf9',
    type: 'anaglyph',
    description: 'High-contrast 3D holographic anaglyph with cyan outer edge & crimson heat core',
    gradient: 'linear-gradient(135deg, #ff0055 0%, #7928ca 50%, #00f5d4 100%)',
    tags: ['Recommended', 'Sci-Fi Anaglyph']
  },
  'cyberpunk': {
    name: 'Cyberpunk Neon',
    glowColor: '#00f2fe',
    fillColor: '#ff0844',
    accentColor: '#f72585',
    type: 'neon',
    description: 'Electrifying neon pink and hyper-cyan contour with intense bloom glow',
    gradient: 'linear-gradient(135deg, #ff0844 0%, #4facfe 100%)',
    tags: ['High Energy', 'Neon']
  },
  'thermal': {
    name: 'Thermal Infrared (Jet)',
    glowColor: '#ffea00',
    fillColor: '#ff0033',
    accentColor: '#0066ff',
    type: 'colormap',
    description: 'Standard military FLIR / Thermal color grading from cold blue to fiery yellow',
    gradient: 'linear-gradient(135deg, #0011ff 0%, #00f5d4 35%, #ffcc00 70%, #ff0033 100%)',
    tags: ['FLIR', 'Heatmap']
  },
  'solar': {
    name: 'Solar Flare Amber',
    glowColor: '#ffe600',
    fillColor: '#ff4800',
    accentColor: '#ff0055',
    type: 'heat',
    description: 'Blazing golden amber outer halo with plasma orange interior',
    gradient: 'linear-gradient(135deg, #ff0055 0%, #ff4800 50%, #ffe600 100%)',
    tags: ['High Visibility']
  },
  'emerald': {
    name: 'Toxic Emerald (Matrix)',
    glowColor: '#00ff66',
    fillColor: '#006633',
    accentColor: '#adff2f',
    type: 'single',
    description: 'Bioluminescent radioactive matrix green with ultra-sharp edge definition',
    gradient: 'linear-gradient(135deg, #003311 0%, #00cc44 60%, #adff2f 100%)',
    tags: ['Matrix', 'Biolum']
  },
  'synthwave': {
    name: 'Synthwave Purple',
    glowColor: '#ff71ce',
    fillColor: '#01cdfe',
    accentColor: '#b967ff',
    type: 'neon',
    description: 'Retro 80s neon purple, magenta, and electric blue aura',
    gradient: 'linear-gradient(135deg, #01cdfe 0%, #b967ff 50%, #ff71ce 100%)',
    tags: ['Retro Wave']
  }
};

// Convert hex color to {r, g, b}
export function hexToRgb(hex) {
  const cleanHex = hex.replace('#', '');
  const bigint = parseInt(cleanHex, 16);
  return {
    r: (bigint >> 16) & 255,
    g: (bigint >> 8) & 255,
    b: bigint & 255
  };
}

// Thermal / Jet Colormap lookup function (0.0 to 1.0)
export function getThermalColor(val) {
  const v = Math.min(Math.max(val, 0), 1);
  let r = 0, g = 0, b = 0;

  if (v < 0.25) {
    // Blue to Cyan
    r = 0;
    g = Math.round(255 * (v / 0.25));
    b = 255;
  } else if (v < 0.5) {
    // Cyan to Green
    r = 0;
    g = 255;
    b = Math.round(255 * (1 - (v - 0.25) / 0.25));
  } else if (v < 0.75) {
    // Green to Yellow
    r = Math.round(255 * ((v - 0.5) / 0.25));
    g = 255;
    b = 0;
  } else {
    // Yellow to Red
    r = 255;
    g = Math.round(255 * (1 - (v - 0.75) / 0.25));
    b = 0;
  }
  return { r, g, b };
}

// Extract edges / contour using Sobel operator on mask data
export function extractContour(maskData, width, height, threshold = 128) {
  const edgeData = new Uint8Array(width * height);
  const binary = new Uint8Array(width * height);

  // First binarize / smooth
  for (let i = 0; i < maskData.length; i += 4) {
    // Mask is grayscale, use R or A channel
    const val = maskData[i];
    binary[i / 4] = val >= threshold ? 255 : 0;
  }

  // Sobel convolution
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const gx =
        -binary[idx - width - 1] + binary[idx - width + 1] +
        -2 * binary[idx - 1]     + 2 * binary[idx + 1] +
        -binary[idx + width - 1] + binary[idx + width + 1];

      const gy =
        -binary[idx - width - 1] - 2 * binary[idx - width] - binary[idx - width + 1] +
        binary[idx + width - 1]  + 2 * binary[idx + width] + binary[idx + width + 1];

      const mag = Math.min(255, Math.hypot(gx, gy));
      edgeData[idx] = mag > 60 ? 255 : 0;
    }
  }

  return edgeData;
}

// Generate Saliency / Camouflage difference approximation client-side
// if a custom image is uploaded with no mask provided
export function generateApproximateMask(imageData, width, height) {
  const mask = new Uint8ClampedArray(width * height * 4);
  const data = imageData.data;
  
  // Calculate average luminance & color variance to highlight high-entropy or central clusters
  const centerX = width / 2;
  const centerY = height / 2;
  const maxDist = Math.hypot(centerX, centerY);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];

      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      const sat = Math.max(r, g, b) - Math.min(r, g, b);
      const dist = Math.hypot(x - centerX, y - centerY) / maxDist;
      
      // Center bias combined with saturation/contrast gradient
      let score = (sat * 1.5 + Math.abs(lum - 128) * 0.8) * (1 - dist * 0.6);
      score = Math.min(255, Math.max(0, score * 1.2));

      mask[idx] = score;
      mask[idx + 1] = score;
      mask[idx + 2] = score;
      mask[idx + 3] = 255;
    }
  }
  return mask;
}
