#!/bin/bash

# Benchmark script for TensTorrent Blackhole scaling
# Tests 1, 65, and 130 cores across all kernels

# Configuration
KERNELS=("gather" "scatter" "gs" "multigather" "multiscatter")
CORE_COUNTS=(1 2 4 8 70 140)  # Blackhole has 130 cores (13x10 grid)
SIZES=(1048576 16777216)
PATTERN="UNIFORM:8:1"
PATTERN_GATHER="UNIFORM:4:1"
PATTERN_SCATTER="UNIFORM:4:1"
WRAP=2
DELTA=8

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "============================================================"
echo "   TensTorrent Blackhole Multi-Core Scaling Benchmark"
echo "============================================================"
echo ""

# Function to run benchmark
run_benchmark() {
    local kernel=$1
    local cores=$2
    local size=$3
    local extra_args=""
    
    # Add kernel-specific patterns
    case $kernel in
        "multigather")
            extra_args="-g $PATTERN_GATHER"
            ;;
        "multiscatter")
            extra_args="-u $PATTERN_SCATTER"
            ;;
        "gs")
            extra_args="-g $PATTERN_GATHER -u $PATTERN_SCATTER"
            ;;
    esac
    
    # Run the benchmark
    output=$(./build/spatter --backend tenstorrent --tt-cores $cores \
             -k $kernel -p $PATTERN -d $DELTA -w $WRAP \
             -l $size $extra_args 2>&1)
    
    if echo "$output" | grep -q "config.*bytes.*time.*bw"; then
        bandwidth=$(echo "$output" | grep -E "^0 " | awk '{print $4}')
        time=$(echo "$output" | grep -E "^0 " | awk '{print $3}')
        echo "${bandwidth}:${time}"
    else
        echo "ERROR:0"
    fi
}

# Test each kernel
for kernel in "${KERNELS[@]}"; do
    echo -e "${CYAN}=== Kernel: $kernel ===${NC}"
    echo ""
    
    # Create table header
    printf "%-12s" "Size"
    for cores in "${CORE_COUNTS[@]}"; do
        printf "%-20s" "${cores} cores (MB/s)"
    done
    echo ""
    echo "-------------------------------------------------------------------------------"
    
    for size in "${SIZES[@]}"; do
        printf "%-12s" "$size"
        
        for cores in "${CORE_COUNTS[@]}"; do
            result=$(run_benchmark "$kernel" "$cores" "$size")
            bw=$(echo $result | cut -d: -f1)
            
            if [ "$bw" == "ERROR" ]; then
                printf "%-20s" "FAILED"
            else
                printf "%-20s" "$bw"
            fi
        done
        
        echo ""
    done
    echo ""
done

# # Detailed analysis for largest size
# echo "============================================================"
# echo "   Detailed Analysis (Size = 10M elements)"
# echo "============================================================"
# echo ""

# for kernel in "${KERNELS[@]}"; do
#     echo -e "${GREEN}Kernel: $kernel${NC}"
#     echo "-------------------------------"
    
#     for cores in "${CORE_COUNTS[@]}"; do
#         echo -n "  $cores cores: "
        
#         # Add kernel-specific patterns
#         extra_args=""
#         case $kernel in
#             "multigather")
#                 extra_args="-g $PATTERN_GATHER"
#                 ;;
#             "multiscatter")
#                 extra_args="-u $PATTERN_SCATTER"
#                 ;;
#             "gs")
#                 extra_args="-g $PATTERN_GATHER -u $PATTERN_SCATTER"
#                 ;;
#         esac
        
#         # Run with verbose output to see core allocation
#         output=$(./build/spatter --backend tenstorrent --tt-cores $cores \
#                  -k $kernel -p $PATTERN -d $DELTA -w $WRAP \
#                  -l 10000000 $extra_args -v 2 2>&1)
        
#         # Extract performance and core info
#         bandwidth=$(echo "$output" | grep -E "^0 " | awk '{print $4}')
#         actual_cores=$(echo "$output" | grep "Cores actually used:" | awk '{print $NF}')
        
#         if [ -n "$bandwidth" ]; then
#             echo "${bandwidth} MB/s (actually using ${actual_cores:-$cores} cores)"
#         else
#             echo -e "${RED}FAILED${NC}"
#         fi
#     done
#     echo ""
# done

# # Find maximum available cores
# echo "============================================================"
# echo "   System Information"
# echo "============================================================"
# echo ""

# # Try to determine actual core count from device info
# output=$(./build/spatter --backend tenstorrent --tt-cores 200 \
#          -k gather -p UNIFORM:1:1 -l 100 -v 2 2>&1)

# compute_grid=$(echo "$output" | grep "Device grid size:" | head -1)
# effective_grid=$(echo "$output" | grep "Effective grid size:" | head -1)
# actual_used=$(echo "$output" | grep "Cores actually used:" | head -1)

# if [ -n "$compute_grid" ]; then
#     echo "Device information:"
#     echo "  $compute_grid"
#     echo "  $effective_grid"
#     echo "  $actual_used"
# else
#     echo "Could not determine device core configuration"
# fi

# echo ""
# echo "============================================================"
# echo "   Performance Summary"
# echo "============================================================"
# echo ""

# # Best performance for each kernel
# echo "Best bandwidth achieved (10M elements):"
# for kernel in "${KERNELS[@]}"; do
#     best_bw=0
#     best_cores=0
    
#     for cores in "${CORE_COUNTS[@]}"; do
#         result=$(run_benchmark "$kernel" "$cores" 10000000)
#         bw=$(echo $result | cut -d: -f1)
        
#         if [ "$bw" != "ERROR" ]; then
#             if (( $(echo "$bw > $best_bw" | bc -l 2>/dev/null || echo 0) )); then
#                 best_bw=$bw
#                 best_cores=$cores
#             fi
#         fi
#     done
    
#     printf "  %-15s: %s MB/s @ %d cores\n" "$kernel" "$best_bw" "$best_cores"
# done

echo ""
echo "Notes:"
echo "- Blackhole uses BFloat16 (2 bytes) vs double precision (8 bytes)"
echo "- Actual core count may be limited by device configuration"
echo "- Performance scaling depends on workload size per core"
echo "- Minimum efficient workload is ~10K-20K elements per core"