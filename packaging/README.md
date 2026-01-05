# Abramelin macOS Packaging

Build scripts for creating a macOS `.pkg` installer for Abramelin.

## Prerequisites

1. **macOS** with Xcode command line tools:
   ```bash
   xcode-select --install
   ```

2. **librsvg** for icon generation (optional but recommended):
   ```bash
   brew install librsvg
   ```

3. **Developer ID** (optional, for code signing and notarization)

## Quick Build

```bash
cd packaging
./build_pkg.sh
```

The installer will be created at `packaging/dist/Abramelin-VERSION-arm64.pkg`.

## Build with Code Signing

For distribution outside the Mac App Store, you need to sign the package:

```bash
./build_pkg.sh --sign "Developer ID Installer: Your Name (TEAMID)"
```

## Notarization

After signing, notarize the package for distribution:

```bash
xcrun notarytool submit dist/Abramelin-*-arm64.pkg \
    --apple-id "your@email.com" \
    --team-id "YOUR_TEAM_ID" \
    --password "app-specific-password" \
    --wait

# Staple the notarization ticket
xcrun stapler staple dist/Abramelin-*-arm64.pkg
```

## Directory Structure

```
packaging/
├── build_pkg.sh          # Main build script
├── README.md             # This file
├── resources/
│   ├── Info.plist        # App bundle metadata
│   ├── Abramelin         # Launcher script (copied to Contents/MacOS/)
│   └── entitlements.plist # Code signing entitlements
├── scripts/
│   ├── preinstall        # Runs before installation
│   └── postinstall       # Runs after installation
├── build/                # Temporary build artifacts (gitignored)
└── dist/                 # Final .pkg output (gitignored)
```

## How the Installer Works

1. **Installation**: The `.pkg` installs `Abramelin.app` to `/Applications`

2. **First Launch**:
   - Checks for Apple Silicon (required for MLX)
   - Checks for macOS 14+ (Sonoma)
   - Checks for Python 3.12 (prompts to install if missing)
   - Creates a virtual environment in `Abramelin.app/Contents/Resources/venv/`
   - Installs all Python dependencies (PyTorch, MLX, FastAPI, etc.)

3. **Model Download**: On first generation, ESM3 weights (~5GB) are downloaded from HuggingFace to `~/.cache/huggingface/`

4. **Runtime**: The app starts a FastAPI server on `localhost:8000` and opens the browser

## Troubleshooting

- **Logs**: `~/Library/Logs/Abramelin/`
- **Cache**: `~/.cache/huggingface/` (model weights)
- **Venv**: `Abramelin.app/Contents/Resources/venv/`

## Size Estimates

- Installer `.pkg`: ~50-100MB
- Installed app (after first run): ~2-3GB (includes venv)
- Model weights: ~5GB (downloaded separately)

## Requirements for End Users

- Apple Silicon Mac (M1, M2, M3, or M4)
- macOS 14 (Sonoma) or later
- Python 3.12 (installer will prompt if missing)
- ~10GB free disk space
- Internet connection (for model download)
