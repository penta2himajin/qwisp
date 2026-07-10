#!/usr/bin/env bash
# scripts/release.sh vX.Y.Z [--notes-file PATH] [--dry-run]
#
# --notes-file PATH prepends your release notes (changelog) above the standard install/asset
# footer in the GitHub release body.
#
# Build, package, and publish a qwisp release, then bump the Homebrew tap formula.
#
# Prereqs:
#   - macOS with Xcode 26 / Swift 6.3 (the ONLY toolchain that builds qwisp — hosted CI can't).
#   - clean `main`, and `gh` authed as a repo admin (the `v*` tag ruleset requires admin to
#     create the tag; a non-admin token is blocked by design).
#   - the model is NOT needed (the smoke test is GPU-free / model-free).
#
# Steps: build Release -> assemble binary + SwiftPM resource bundles -> configtest smoke ->
#        tarball + sha256 -> gh release create -> rewrite tap formula (url/sha256/version) -> push.
#
# --dry-run does everything up to (and including) the tarball + sha256, but skips the two
# side-effecting steps (GitHub release, tap push). Use it to validate the build/package half.
set -euo pipefail

VERSION="" DRY=0 NOTES_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)        DRY=1; shift;;
    --notes-file)     NOTES_FILE="${2:-}"; shift 2;;
    --notes-file=*)   NOTES_FILE="${1#*=}"; shift;;
    v*)               VERSION="$1"; shift;;
    *)                echo "unknown arg: $1"; exit 1;;
  esac
done
[[ "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "usage: scripts/release.sh vX.Y.Z [--notes-file PATH] [--dry-run]"; exit 1; }
[[ -z "$NOTES_FILE" || -f "$NOTES_FILE" ]] || { echo "ERROR: --notes-file '$NOTES_FILE' not found"; exit 1; }
BARE="${VERSION#v}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
TAP_REMOTE="https://github.com/penta2himajin/homebrew-qwisp.git"
ASSET="qwisp-${VERSION}-macos-arm64.tar.gz"
PROD="swift/.xcode-build-rel/Build/Products/Release"

# 0. sanity
[[ "$(git branch --show-current)" == "main" ]] || { echo "ERROR: not on main"; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "ERROR: working tree is dirty"; exit 1; }
if [[ $DRY -eq 0 ]] && gh release view "$VERSION" >/dev/null 2>&1; then
  echo "ERROR: release $VERSION already exists"; exit 1
fi

# 1. build Release (qwisp scheme only — qwisp-poc is not shipped)
echo "==> building Release $VERSION (Xcode 26 required)"
pkill -f 'Release/qwisp' 2>/dev/null || true
( cd swift && xcodebuild build -scheme qwisp -configuration Release -destination 'platform=macOS' \
    -derivedDataPath ./.xcode-build-rel -skipPackagePluginValidation ) > "$(mktemp -t qwisp-build.XXXX)" 2>&1 \
  || { echo "ERROR: build failed"; exit 1; }

# 2. assemble dist: the binary + EVERY SwiftPM resource bundle it loads at runtime
#    (mlx-swift_Cmlx.bundle carries MLX's default.metallib; all must sit next to the binary).
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
DIST="$WORK/qwisp-${VERSION}-macos-arm64"
mkdir -p "$DIST"
cp "$PROD/qwisp" "$DIST/"
cp -R "$PROD"/*.bundle "$DIST/"
echo "==> bundled: $(cd "$DIST" && ls -d *.bundle | tr '\n' ' ')"

# 3. smoke: the assembled binary runs standalone (GPU-free, no model)
echo -n "==> smoke: " && "$DIST/qwisp" configtest | tail -1

# 4. tarball + sha256
TAR="$WORK/$ASSET"
tar czf "$TAR" -C "$WORK" "qwisp-${VERSION}-macos-arm64"
SHA="$(shasum -a 256 "$TAR" | awk '{print $1}')"
echo "==> $ASSET  ($(du -h "$TAR" | awk '{print $1}'))  sha256=$SHA"

if [[ $DRY -eq 1 ]]; then
  echo "==> --dry-run: skipping GitHub release + tap push. Artifact: $TAR"
  exit 0
fi

# 5. GitHub release (admin token bypasses the v* tag ruleset)
NOTES="$WORK/notes.md"
: > "$NOTES"
# Prepend the caller's release notes (changelog etc.), then the standard install/asset footer.
if [[ -n "$NOTES_FILE" ]]; then cat "$NOTES_FILE" >> "$NOTES"; printf '\n\n---\n\n' >> "$NOTES"; fi
cat >> "$NOTES" <<'EOF'
```bash
brew install penta2himajin/qwisp/qwisp
qwisp pull            # download the default model (~20 GB) + write config
qwisp chat "..."      # or: brew services start qwisp
```

Apple Silicon (arm64), macOS 14+. Greedy/lossless by default; sampling honored when requested.
Asset: `__ASSET__` — the `qwisp` binary + its colocated resource bundles.
EOF
/usr/bin/sed -i '' "s|__ASSET__|$ASSET|g" "$NOTES"
gh release create "$VERSION" "$TAR" --target main --title "qwisp $VERSION" --notes-file "$NOTES"

# 6. bump the Homebrew tap formula (url / sha256 / version)
echo "==> updating tap formula"
TAP="$WORK/homebrew-qwisp"
git clone -q "$TAP_REMOTE" "$TAP"
URL="https://github.com/penta2himajin/qwisp/releases/download/${VERSION}/${ASSET}"
/usr/bin/sed -i '' \
  -e "s|^  url \".*\"|  url \"$URL\"|" \
  -e "s|^  sha256 \".*\"|  sha256 \"$SHA\"|" \
  -e "s|^  version \".*\"|  version \"$BARE\"|" \
  "$TAP/Formula/qwisp.rb"
( cd "$TAP" && git add Formula/qwisp.rb \
    && git -c user.name="Kenya Nara" -c user.email="penta2himajin@gmail.com" commit -q -m "qwisp $BARE" \
    && git push -q )

echo "==> done: https://github.com/penta2himajin/qwisp/releases/tag/$VERSION"
