#!/bin/bash
# Abramelin macOS Package Builder
# Creates a .pkg installer for Abramelin protein design workbench
#
# Usage: ./build_pkg.sh [--sign IDENTITY]
#
# Requirements:
#   - macOS with Xcode command line tools
#   - rsvg-convert (brew install librsvg) for icon generation
#   - Optional: Developer ID for code signing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
OUTPUT_DIR="$SCRIPT_DIR/dist"

# Version from pyproject.toml
VERSION=$(grep 'version = ' "$PROJECT_DIR/pyproject.toml" | head -1 | cut -d'"' -f2)
VERSION=${VERSION:-"1.0.0"}

APP_NAME="Abramelin"
BUNDLE_ID="com.taontronic.abramelin"
PKG_NAME="${APP_NAME}-${VERSION}-arm64"

# Parse arguments
SIGN_IDENTITY=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --sign)
            SIGN_IDENTITY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Building $APP_NAME $VERSION"
echo "========================================"

# Clean previous build
echo "Cleaning previous build..."
rm -rf "$BUILD_DIR" "$OUTPUT_DIR"
mkdir -p "$BUILD_DIR" "$OUTPUT_DIR"

# Create app bundle structure
echo "Creating app bundle structure..."
APP_BUNDLE="$BUILD_DIR/$APP_NAME.app"
mkdir -p "$APP_BUNDLE/Contents/MacOS"
mkdir -p "$APP_BUNDLE/Contents/Resources"

# Copy Info.plist
echo "Copying Info.plist..."
cp "$SCRIPT_DIR/resources/Info.plist" "$APP_BUNDLE/Contents/"

# Update version in Info.plist
sed -i '' "s/<string>1.0.0<\/string>/<string>$VERSION<\/string>/" "$APP_BUNDLE/Contents/Info.plist"

# Copy launcher script
echo "Copying launcher..."
cp "$SCRIPT_DIR/resources/Abramelin" "$APP_BUNDLE/Contents/MacOS/"
chmod +x "$APP_BUNDLE/Contents/MacOS/Abramelin"

# Generate icon from SVG
echo "Generating app icon..."
ICON_SVG="$PROJECT_DIR/esm/web/static/pics/abramelin1.svg"
ICONSET_DIR="$BUILD_DIR/AppIcon.iconset"
mkdir -p "$ICONSET_DIR"

if command -v rsvg-convert &>/dev/null; then
    # Generate all required icon sizes
    for size in 16 32 64 128 256 512; do
        rsvg-convert -w $size -h $size "$ICON_SVG" -o "$ICONSET_DIR/icon_${size}x${size}.png"
        rsvg-convert -w $((size*2)) -h $((size*2)) "$ICON_SVG" -o "$ICONSET_DIR/icon_${size}x${size}@2x.png"
    done

    # Create icns file
    iconutil -c icns "$ICONSET_DIR" -o "$APP_BUNDLE/Contents/Resources/AppIcon.icns"
    echo "Icon generated successfully"
else
    echo "Warning: rsvg-convert not found. Install with: brew install librsvg"
    echo "Using placeholder icon..."
    # Create a minimal placeholder icon (will show generic icon)
fi

# Copy ESM package
echo "Copying ESM package..."
ESM_DEST="$APP_BUNDLE/Contents/Resources/esm"
mkdir -p "$ESM_DEST"

# Copy Python source (excluding cache and venv)
rsync -av --exclude='__pycache__' \
          --exclude='*.pyc' \
          --exclude='.git' \
          --exclude='.venv' \
          --exclude='*.egg-info' \
          --exclude='build' \
          --exclude='dist' \
          --exclude='packaging/build' \
          --exclude='packaging/dist' \
          "$PROJECT_DIR/esm" "$ESM_DEST/../"

# Note: ESM package is loaded via PYTHONPATH, no pyproject.toml needed

# Copy data files
if [[ -d "$PROJECT_DIR/esm/data" ]]; then
    cp -r "$PROJECT_DIR/esm/data" "$ESM_DEST/"
fi

# Create PkgInfo
echo "APPL????" > "$APP_BUNDLE/Contents/PkgInfo"

# Note: App bundle code signing requires "Developer ID Application" certificate
# The package signing below uses "Developer ID Installer" certificate
# For now, we skip app bundle signing - the signed package is sufficient for distribution

# Create component package
echo "Creating component package..."
pkgbuild --root "$BUILD_DIR" \
         --identifier "$BUNDLE_ID" \
         --version "$VERSION" \
         --install-location "/Applications" \
         --scripts "$SCRIPT_DIR/scripts" \
         "$BUILD_DIR/component.pkg"

# Create distribution XML
echo "Creating distribution..."
cat > "$BUILD_DIR/distribution.xml" << EOF
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
    <title>Abramelin</title>
    <organization>com.taontronic</organization>
    <license file="license.txt"/>
    <readme file="readme.txt"/>
    <welcome file="welcome.txt"/>
    <conclusion file="conclusion.txt"/>

    <options customize="never" require-scripts="false" hostArchitectures="arm64"/>

    <domains enable_anywhere="false" enable_currentUserHome="false" enable_localSystem="true"/>

    <volume-check>
        <allowed-os-versions>
            <os-version min="14.0"/>
        </allowed-os-versions>
    </volume-check>

    <choices-outline>
        <line choice="default">
            <line choice="com.taontronic.abramelin"/>
        </line>
    </choices-outline>

    <choice id="default"/>
    <choice id="com.taontronic.abramelin" visible="false">
        <pkg-ref id="com.taontronic.abramelin"/>
    </choice>

    <pkg-ref id="com.taontronic.abramelin" version="$VERSION">component.pkg</pkg-ref>
</installer-gui-script>
EOF

# Create installer text files
cat > "$BUILD_DIR/welcome.txt" << 'EOF'
Welcome to Abramelin

Abramelin is an interactive protein design workbench powered by ESM3 with Apple Silicon acceleration.

Features:
- Protein sequence generation and structure prediction
- Interactive 3D visualization with 3Dmol.js
- Real-time generation streaming via WebSocket
- Function annotation prediction
- Ensemble structure generation

Requirements:
- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 14 (Sonoma) or later
- Python 3.12 (will prompt to install if missing)
- ~10GB free disk space (for models)

Click Continue to proceed with installation.
EOF

cat > "$BUILD_DIR/readme.txt" << 'EOF'
About Abramelin

Abramelin uses ESM3 (Evolutionary Scale Modeling), a frontier protein language model developed by EvolutionaryScale, optimized for Apple Silicon using MLX.

On first launch:
1. Abramelin will set up a Python virtual environment
2. Install required dependencies (~5 minutes)
3. Download ESM3 model weights from HuggingFace (~5GB)

After setup, subsequent launches are instant.

Usage:
- Launch Abramelin from Applications
- Your default browser will open to http://localhost:8000
- Enter a protein sequence or use masks (_) for generation
- Click Generate to create sequences and structures

For more information, visit the documentation.
EOF

cat > "$BUILD_DIR/license.txt" << 'EOF'
Abramelin License

MIT License

Copyright (c) 2024 Taontronic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

ESM3 Model License

ESM3 is developed by EvolutionaryScale and is subject to its own license terms.
See: https://github.com/evolutionaryscale/esm

---

Third-Party Licenses

This software includes the following open-source components:
- MLX by Apple (MIT License)
- FastAPI (MIT License)
- 3Dmol.js (BSD License)
- PyTorch (BSD License)
- And other dependencies as listed in pyproject.toml
EOF

cat > "$BUILD_DIR/conclusion.txt" << 'EOF'
Installation Complete!

Abramelin has been installed to your Applications folder.

To get started:
1. Open Abramelin from Applications
2. Wait for first-time setup (Python environment and dependencies)
3. Your browser will open when the server is ready

First-time model download:
The ESM3 model (~5GB) will be downloaded automatically on first generation.
This is a one-time download and will be cached for future use.

Troubleshooting:
- Logs are stored in ~/Library/Logs/Abramelin/
- If you encounter issues, check the latest log file

Enjoy designing proteins with Abramelin!
EOF

# Build the final product package
echo "Building final package..."
productbuild --distribution "$BUILD_DIR/distribution.xml" \
             --package-path "$BUILD_DIR" \
             --resources "$BUILD_DIR" \
             "$OUTPUT_DIR/$PKG_NAME.pkg"

# Sign the package (if identity provided)
if [[ -n "$SIGN_IDENTITY" ]]; then
    echo "Signing package..."
    productsign --sign "$SIGN_IDENTITY" \
                "$OUTPUT_DIR/$PKG_NAME.pkg" \
                "$OUTPUT_DIR/$PKG_NAME-signed.pkg"
    mv "$OUTPUT_DIR/$PKG_NAME-signed.pkg" "$OUTPUT_DIR/$PKG_NAME.pkg"
fi

# Create checksum
echo "Creating checksum..."
cd "$OUTPUT_DIR"
shasum -a 256 "$PKG_NAME.pkg" > "$PKG_NAME.pkg.sha256"

echo ""
echo "========================================"
echo "Build complete!"
echo "========================================"
echo "Package: $OUTPUT_DIR/$PKG_NAME.pkg"
echo "Size: $(du -h "$OUTPUT_DIR/$PKG_NAME.pkg" | cut -f1)"
echo "SHA256: $(cat "$PKG_NAME.pkg.sha256")"
echo ""
echo "To install: double-click the .pkg file"
echo "To notarize (required for distribution):"
echo "  xcrun notarytool submit $OUTPUT_DIR/$PKG_NAME.pkg --apple-id YOUR_APPLE_ID --team-id YOUR_TEAM_ID --password YOUR_APP_PASSWORD"
