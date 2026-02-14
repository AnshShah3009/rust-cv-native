#!/usr/bin/env bash
# Performance validation script
# Compares our implementation with basic expectations

echo "=========================================="
echo "Rust CV Native - Performance Validation"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Running benchmarks..."
echo ""

# Run benchmarks
cargo bench 2>&1 | tee benchmark_results.txt

echo ""
echo "=========================================="
echo "Benchmark Complete"
echo "=========================================="
echo ""
echo "Results saved to benchmark_results.txt"
echo ""

# Check for GPU availability
echo "Checking GPU availability..."
if cargo test -p cv-stereo --features gpu gpu_available 2>/dev/null; then
    echo -e "${GREEN}✓${NC} GPU available for testing"
else
    echo -e "${YELLOW}⚠${NC} GPU not available or tests failed"
fi

echo ""
echo "Key Performance Indicators:"
echo "- Stereo block matching 512x512: Should complete in < 100ms (CPU)"
echo "- Stereo SGM 512x512: Should complete in < 500ms (CPU)"
echo "- FAST detection 1024x1024: Should complete in < 50ms"
echo "- Gaussian blur 1024x1024: Should complete in < 20ms"
echo ""
echo "Note: These are rough estimates. Integrated GPU should be 2-5x faster for parallelizable tasks."
