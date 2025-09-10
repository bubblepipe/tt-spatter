#!/bin/bash

# Benchmark script for OpenMP backend scaling
# Tests 1, 2, 4, 8, 16, 32 threads across all kernels

# Configuration
KERNELS=("gather" "scatter" "gs" "multigather" "multiscatter")
THREAD_COUNTS=(1 2 4 6 9 12)  # Common thread counts for OpenMP
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
echo "        OpenMP Multi-Thread Scaling Benchmark"
echo "============================================================"
echo ""

# Check if OpenMP backend is available
if ! ./build_openmp/spatter --backend openmp -k gather -p UNIFORM:1:1 -l 10 &>/dev/null; then
    echo -e "${RED}ERROR: OpenMP backend not available!${NC}"
    echo "Make sure Spatter was built with OpenMP support:"
    echo "  cmake -DUSE_OPENMP=1 -B build_openmp -S ."
    echo "  cd build_openmp && make -j"
    exit 1
fi

# Function to run benchmark
run_benchmark() {
    local kernel=$1
    local threads=$2
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
    
    # Run the benchmark with OpenMP
    output=$(./build_openmp/spatter --backend openmp -t $threads \
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
    for threads in "${THREAD_COUNTS[@]}"; do
        printf "%-20s" "${threads} threads (MB/s)"
    done
    echo ""
    echo "-------------------------------------------------------------------------------"
    
    for size in "${SIZES[@]}"; do
        printf "%-12s" "$size"
        
        for threads in "${THREAD_COUNTS[@]}"; do
            result=$(run_benchmark "$kernel" "$threads" "$size")
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

# Detailed analysis for largest size
echo "============================================================"
echo "   Detailed Analysis (Size = 10M elements)"
echo "============================================================"
echo ""

for kernel in "${KERNELS[@]}"; do
    echo -e "${GREEN}Kernel: $kernel${NC}"
    echo "-------------------------------"
    
    for threads in "${THREAD_COUNTS[@]}"; do
        echo -n "  $threads threads: "
        
        # Add kernel-specific patterns
        extra_args=""
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
        
        # Run with verbose output
        output=$(./build_openmp/spatter --backend openmp -t $threads \
                 -k $kernel -p $PATTERN -d $DELTA -w $WRAP \
                 -l 10000000 $extra_args -v 2 2>&1)
        
        # Extract performance
        bandwidth=$(echo "$output" | grep -E "^0 " | awk '{print $4}')
        
        if [ -n "$bandwidth" ]; then
            echo "${bandwidth} MB/s"
        else
            echo -e "${RED}FAILED${NC}"
        fi
    done
    echo ""
done

# System information
echo "============================================================"
echo "   System Information"
echo "============================================================"
echo ""

# Get CPU information
echo "CPU Information:"
if [ -f /proc/cpuinfo ]; then
    model=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | sed 's/^ *//')
    cores=$(grep -c "processor" /proc/cpuinfo)
    echo "  Model: $model"
    echo "  Total cores: $cores"
else
    echo "  Unable to determine CPU information"
fi

echo ""
echo "OpenMP Information:"
# Check OpenMP version
OMP_VERSION=$(./build_openmp/spatter --backend openmp -t 1 -k gather -p UNIFORM:1:1 -l 10 -v 2 2>&1 | grep -i "openmp" | head -1)
if [ -n "$OMP_VERSION" ]; then
    echo "  $OMP_VERSION"
fi

# Check max threads
echo "  OMP_NUM_THREADS=${OMP_NUM_THREADS:-not set}"
echo "  Maximum threads available: $(nproc)"

echo ""
echo "============================================================"
echo "   Performance Summary"
echo "============================================================"
echo ""

# Best performance for each kernel
echo "Best bandwidth achieved (10M elements):"
for kernel in "${KERNELS[@]}"; do
    best_bw=0
    best_threads=0
    
    for threads in "${THREAD_COUNTS[@]}"; do
        result=$(run_benchmark "$kernel" "$threads" 10000000)
        bw=$(echo $result | cut -d: -f1)
        
        if [ "$bw" != "ERROR" ]; then
            if (( $(echo "$bw > $best_bw" | bc -l 2>/dev/null || echo 0) )); then
                best_bw=$bw
                best_threads=$threads
            fi
        fi
    done
    
    printf "  %-15s: %s MB/s @ %d threads\n" "$kernel" "$best_bw" "$best_threads"
done

echo ""
echo "============================================================"
echo "   Comparison with Serial Backend"
echo "============================================================"
echo ""

echo "Comparing OpenMP (best thread count) vs Serial for 1M elements:"
for kernel in "${KERNELS[@]}"; do
    # Get serial performance
    extra_args=""
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
    
    serial_output=$(./build_openmp/spatter --backend serial \
                    -k $kernel -p $PATTERN -d $DELTA -w $WRAP \
                    -l 1000000 $extra_args 2>&1)
    serial_bw=$(echo "$serial_output" | grep -E "^0 " | awk '{print $4}')
    
    # Find best OpenMP performance
    best_openmp_bw=0
    best_threads=0
    for threads in "${THREAD_COUNTS[@]}"; do
        result=$(run_benchmark "$kernel" "$threads" 1000000)
        bw=$(echo $result | cut -d: -f1)
        if [ "$bw" != "ERROR" ]; then
            if (( $(echo "$bw > $best_openmp_bw" | bc -l 2>/dev/null || echo 0) )); then
                best_openmp_bw=$bw
                best_threads=$threads
            fi
        fi
    done
    
    if [ -n "$serial_bw" ] && [ "$best_openmp_bw" != "0" ]; then
        speedup=$(echo "scale=2; $best_openmp_bw / $serial_bw" | bc 2>/dev/null || echo "N/A")
        printf "  %-15s: Serial=%s MB/s, OpenMP=%s MB/s @ %d threads (%.2fx speedup)\n" \
               "$kernel" "$serial_bw" "$best_openmp_bw" "$best_threads" "$speedup"
    fi
done

echo ""
echo "Notes:"
echo "- OpenMP uses double precision (8 bytes)"
echo "- Performance scaling depends on memory bandwidth and cache hierarchy"
echo "- Thread count should not exceed physical cores for best performance"
echo "- NUMA effects may impact performance on multi-socket systems"