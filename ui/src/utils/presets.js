// Built-in presets from FAPNet / Chameleon dataset

export const PRESET_SAMPLES = [
  {
    id: 'sample1',
    title: 'Chameleon in Jungle Foliage',
    category: 'CHAMELEON',
    description: 'Chameleon perfectly matching leaf textures and green hues.',
    image: '/samples/sample1_img.jpg',
    mask: '/samples/sample1_mask.png',
    resolution: '1024x768',
    detectionRate: '98.4%',
    difficulty: 'Extreme',
    features: ['Complex Foliage', 'Color Match', 'Textured Skin']
  },
  {
    id: 'sample2',
    title: 'Flatfish / Flounder on Seabed',
    category: 'CHAMELEON / CAMO',
    description: 'Benthic flatfish mimicking sandy gravel substrate.',
    image: '/samples/sample2_img.jpg',
    mask: '/samples/sample2_mask.png',
    resolution: '800x600',
    detectionRate: '96.8%',
    difficulty: 'High',
    features: ['Grainy Substrate', 'Submerged', 'Low Contrast']
  },
  {
    id: 'sample3',
    title: 'Gecko / Lizard on Tree Bark',
    category: 'COD10K',
    description: 'Moss-textured lizard blending into rough oak bark.',
    image: '/samples/sample3_img.jpg',
    mask: '/samples/sample3_mask.png',
    resolution: '1200x900',
    difficulty: 'Extreme',
    detectionRate: '99.1%',
    features: ['Bark Mimicry', 'Fractal Shadow', 'Rough Texture']
  },
  {
    id: 'sample4',
    title: 'Stick Insect / Mantis on Twig',
    category: 'NC4K',
    description: 'Cryptic insect masquerading as dry tree branches.',
    image: '/samples/sample4_img.jpg',
    mask: '/samples/sample4_mask.png',
    resolution: '960x720',
    difficulty: 'Master',
    detectionRate: '97.5%',
    features: ['Geometric Mimicry', 'Slender Geometry', 'Twig Masquerade']
  }
];
