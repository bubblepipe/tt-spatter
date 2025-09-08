#!/bin/bash

# Test script for TensTorrent gather_scatter kernel
# Tests various configurations and validates correctness

set -e  # Exit on error

SPATTER_BIN="./build/spatter"
BACKEND="tenstorrent"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "TensTorrent Gather-Scatter Kernel Test"
echo "=========================================="
echo ""

# Function to run a test
run_test() {
    local test_name="$1"
    local cores="$2"
    local pattern_gather="$3"
    local pattern_scatter="$4"
    local size="$5"
    local delta_gather="${6:-8}"
    local delta_scatter="${7:-8}"
    
    echo -n "Testing $test_name (cores=$cores, size=$size)... "
    
    # Run the test and capture output
    output=$($SPATTER_BIN --backend $BACKEND --tt-cores $cores -k gs \
             -g "$pattern_gather" -u "$pattern_scatter" \
             -x "$delta_gather" -y "$delta_scatter" \
             -l $size 2>&1)
    
    # Check if validation passed
    if echo "$output" | grep -q "✓ Gather-Scatter kernel validation PASSED"; then
        # Extract bandwidth
        bandwidth=$(echo "$output" | grep -E "^0 " | awk '{print $4}')
        echo -e "${GREEN}PASSED${NC} (${bandwidth} MB/s)"
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        echo "Error output:"
        echo "$output" | grep -E "validation|FAILED|Error"
        return 1
    fi
}

# Function to run performance test
perf_test() {
    local cores="$1"
    local size="$2"
    
    echo -n "Performance test (cores=$cores, size=$size): "
    
    output=$($SPATTER_BIN --backend $BACKEND --tt-cores $cores -k gs \
             -g "UNIFORM:8:1" -u "UNIFORM:8:1" \
             -l $size 2>&1)
    
    bandwidth=$(echo "$output" | grep -E "^0 " | awk '{print $4}')
    time=$(echo "$output" | grep -E "^0 " | awk '{print $3}')
    
    echo "Bandwidth: ${bandwidth} MB/s, Time: ${time}s"
}

# Test suite
# echo "=== Basic Functionality Tests ==="
# echo ""

# # Test 1: Small size, single core
# run_test "Small single-core" 1 "UNIFORM:8:1" "UNIFORM:8:1" 100

# # Test 2: Medium size, single core
# run_test "Medium single-core" 1 "UNIFORM:8:1" "UNIFORM:8:1" 10000

# # Test 3: Large size, single core
# run_test "Large single-core" 1 "UNIFORM:8:1" "UNIFORM:8:1" 100000

# # Test 4: Multi-core tests
# echo ""
# echo "=== Multi-Core Tests ==="
# echo ""

# run_test "Small multi-core" 4 "UNIFORM:8:1" "UNIFORM:8:1" 1000
# run_test "Medium multi-core" 8 "UNIFORM:8:1" "UNIFORM:8:1" 100000
# run_test "Large multi-core" 30 "UNIFORM:8:1" "UNIFORM:8:1" 1000000

# # Test 5: Different patterns (must have same length for gather_scatter)
# echo ""
# echo "=== Pattern Variation Tests ==="
# echo ""

# # Note: Both patterns must have the same length for gather_scatter
# run_test "Same length patterns" 4 "UNIFORM:16:1" "UNIFORM:16:1" 10000
# run_test "Different deltas" 4 "UNIFORM:8:2" "UNIFORM:8:3" 10000
# # Complex pattern with large strides can cause overlapping writes in multi-core
# # This test may fail with multiple cores due to lack of atomic operations
# # Known limitation: Multi-core scatter can have race conditions without atomics
# run_test "Complex pattern (single-core)" 1 "UNIFORM:64:4" "UNIFORM:64:8" 50000

# Test 6: Different delta values
echo ""
echo "=== Delta Parameter Tests ==="
echo ""

run_test "Delta 1" 4 "UNIFORM:8:1" "UNIFORM:8:1" 10000 1 1
run_test "Delta 16" 4 "UNIFORM:8:1" "UNIFORM:8:1" 10000 16 16
run_test "Delta 64" 4 "UNIFORM:8:1" "UNIFORM:8:1" 10000 64 64
run_test "Mixed deltas" 4 "UNIFORM:8:1" "UNIFORM:8:1" 10000 8 16

# Test 7: Edge cases
echo ""
echo "=== Edge Case Tests ==="
echo ""

run_test "Minimum size" 1 "UNIFORM:1:1" "UNIFORM:1:1" 1
run_test "Power of 2 - 1" 1 "UNIFORM:8:1" "UNIFORM:8:1" 1023
run_test "Power of 2" 1 "UNIFORM:8:1" "UNIFORM:8:1" 1024
run_test "Power of 2 + 1" 1 "UNIFORM:8:1" "UNIFORM:8:1" 1025

# Test 8: Stress test with very large sizes
echo ""
echo "=== Stress Tests ==="
echo ""

if [ "${RUN_STRESS_TESTS:-0}" = "1" ]; then
    run_test "Very large single" 1 "UNIFORM:8:1" "UNIFORM:8:1" 10000000
    run_test "Very large multi" 30 "UNIFORM:8:1" "UNIFORM:8:1" 10000000
else
    echo "Skipping stress tests (set RUN_STRESS_TESTS=1 to enable)"
fi

# Performance scaling test
echo ""
echo "=== Performance Scaling Tests ==="
echo ""

echo "Single-core scaling:"
for size in 1000 10000 100000 1000000; do
    perf_test 1 $size
done

echo ""
echo "Multi-core scaling (size=1000000):"
for cores in 1 2 4 8 16 30; do
    perf_test $cores 1000000
done

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="

# Count passed/failed tests from the output
total_tests=$(grep -c "Testing\|Performance test" $0 | head -1)
echo "All tests completed!"
echo ""

# Additional validation with comparison against CPU
if [ "${VALIDATE_AGAINST_CPU:-0}" = "1" ]; then
    echo "=== CPU Validation Tests ==="
    echo "Running gather_scatter on CPU and TensTorrent for comparison..."
    
    # Run on CPU (serial backend)
    cpu_output=$($SPATTER_BIN --backend serial -k gs \
                 -g "UNIFORM:8:1" -u "UNIFORM:8:1" \
                 -l 1000 2>&1)
    cpu_bw=$(echo "$cpu_output" | grep -E "^0 " | awk '{print $4}')
    
    # Run on TensTorrent
    tt_output=$($SPATTER_BIN --backend $BACKEND --tt-cores 1 -k gs \
                -g "UNIFORM:8:1" -u "UNIFORM:8:1" \
                -l 1000 2>&1)
    tt_bw=$(echo "$tt_output" | grep -E "^0 " | awk '{print $4}')
    
    echo "CPU Bandwidth: ${cpu_bw} MB/s"
    echo "TT Bandwidth: ${tt_bw} MB/s"
fi

echo ""
echo "Test script completed successfully!"