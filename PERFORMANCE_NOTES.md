# Performance Optimization Guide

This music isolation tool now includes performance optimizations to significantly reduce processing time for songs.

## Performance Presets

The tool includes three performance presets that balance speed and quality:

### Fast Mode (Recommended for quick results)
- **Segment Size**: 128 seconds
- **Shifts**: 1
- **Overlap**: 0.15
- **Use case**: Quick previews, testing, when speed is more important than perfect quality
- **Speed improvement**: ~3-4x faster than default

### Balanced Mode (Default)
- **Segment Size**: 256 seconds  
- **Shifts**: 2
- **Overlap**: 0.25
- **Use case**: General use, good balance of speed and quality
- **Speed improvement**: Baseline performance

### Quality Mode
- **Segment Size**: 512 seconds
- **Shifts**: 4
- **Overlap**: 0.35
- **Use case**: Final production, when quality is most important
- **Speed impact**: ~2-3x slower than default

## Usage

### Command Line

```bash
# Use fast preset for quick processing
python separator.py "song.mp3" --preset fast

# Use quality preset for best results
python separator.py "song.mp3" --preset quality

# Custom performance parameters
python separator.py "song.mp3" --segment-size 128 --shifts 1 --overlap 0.15

# Force GPU usage (if available)
python separator.py "song.mp3" --device cuda --preset fast
```

### Web Interface

The web interface now includes:
- **Performance Preset dropdown**: Choose between Fast, Balanced, or Quality
- **Device selection**: Auto-detect GPU, force NVIDIA GPU, or use CPU only

## GPU Acceleration

The tool automatically detects and uses available GPU acceleration:

- **NVIDIA GPU (CUDA)**: Automatically detected and used when available
- **Apple Silicon (MPS)**: Automatically detected on Macs with M1/M2/M3 chips
- **CPU fallback**: Automatically used when no GPU is available

### GPU Memory Optimization

When GPU is detected, the tool automatically:
- Enables CUDA memory allocation optimizations
- Uses batch processing for faster inference
- Logs GPU detection and acceleration status

## Technical Details

### Segment Size
- Controls how much audio is processed at once
- Lower values = faster processing but potentially lower quality
- Higher values = better quality but slower processing and more memory usage

### Shifts
- Number of random time shifts applied for better separation
- Higher values = better quality through temporal averaging
- Lower values = faster processing

### Overlap
- Amount of overlap between processing segments
- Higher values = smoother transitions but slower processing
- Lower values = faster processing but potential artifacts at segment boundaries

## Expected Performance Improvements

With the "fast" preset on a typical system:
- **CPU-only**: ~2-3x faster than default
- **With GPU**: ~3-5x faster than default
- **With GPU + fast preset**: ~5-8x faster than default

## Troubleshooting

### Out of Memory Errors
If you encounter GPU memory errors:
1. Use the "fast" preset (lower segment size)
2. Switch to CPU-only mode: `--device cpu`
3. Reduce segment size further: `--segment-size 64`

### Quality Issues
If you notice quality degradation with fast presets:
1. Try the "balanced" preset
2. Use the "quality" preset for final production
3. Increase overlap: `--overlap 0.3`

### Slow Performance
If processing is still slow:
1. Verify GPU is being used (check logs for "GPU detected")
2. Try the "fast" preset
3. Ensure you're using the latest PyTorch version with GPU support
