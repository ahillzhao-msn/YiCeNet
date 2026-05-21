#!/usr/bin/env bash
# ============================================================
# YiCeNet GitHub Release Publisher
# Usage: publish_release.sh <version> [--dry-run]
#
# Creates a GitHub release with the specified checkpoint version.
# Only run manually (NOT automated — releases are on-demand).
#
# Prerequisites: gh CLI (apt install gh, then gh auth login)
# ============================================================
set -euo pipefail

VERSION="${1:?Usage: publish_release.sh <version> [--dry-run]}"
DRY_RUN="${2:-}"
YICENET_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT_DIR="$YICENET_ROOT/checkpoints"
RELEASE_DIR="$YICENET_ROOT/release"

# ── Verify checkpoint exists ──
CKPT_FILE="yicenet_${VERSION}.pt"
CKPT_PATH="$CHECKPOINT_DIR/$CKPT_FILE"
if [ ! -f "$CKPT_PATH" ]; then
    echo "✗ Checkpoint not found: $CKPT_PATH"
    echo "  Available checkpoints:"
    ls "$CHECKPOINT_DIR"/yicenet_v*.pt 2>/dev/null || echo "  (none)"
    exit 1
fi

# ── Build release tarball ──
echo "📦 Building ${VERSION} release..."
mkdir -p "$RELEASE_DIR"
TARBALL="$RELEASE_DIR/yicenet_${VERSION}_release.tar.gz"

tar -czf "$TARBALL" \
    -C "$YICENET_ROOT" \
    --exclude='checkpoints/*' \
    --exclude='data/*' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='logs' \
    src/ tests/ scripts/ requirements.txt \
    README.md \
    yicenet_architecture.md \
    "$CKPT_PATH"

echo "  ✓ Tarball: $TARBALL ($(du -h "$TARBALL" | cut -f1))"

# ── Git tag ──
if git -C "$YICENET_ROOT" tag | grep -q "^${VERSION}$"; then
    echo "  ! Tag ${VERSION} already exists, skipping tag"
else
    git -C "$YICENET_ROOT" tag -a "$VERSION" -m "YiCeNet ${VERSION} release"
    echo "  ✓ Tagged ${VERSION}"
fi

# ── Publish to GitHub ──
if [ "$DRY_RUN" = "--dry-run" ]; then
    echo ""
    echo "⚠️  DRY RUN — would run:"
    echo "  git push origin $VERSION"
    echo "  gh release create $VERSION '$TARBALL' --title 'YiCeNet ${VERSION}' --notes 'See release notes'"
else
    echo "🚀 Publishing ${VERSION} to GitHub..."
    git -C "$YICENET_ROOT" push origin "$VERSION"
    gh release create "$VERSION" "$TARBALL" \
        --repo "$(git -C "$YICENET_ROOT" remote get-url origin | sed 's|.*github.com/||;s|\.git$||')" \
        --title "YiCeNet ${VERSION}" \
        --notes ""
    echo "  ✓ Published ${VERSION} to GitHub Releases"
fi

echo ""
echo "Done."
