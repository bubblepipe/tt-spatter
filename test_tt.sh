#!/bin/bash

# Test script to identify which problem sizes cause crashes or validation failures with TensTorrent backend
# Tests powers of 2 from 2^4 to 2^30
# Exit codes:
#   0   - Success
#   1   - Validation failure (kernel produced incorrect results)
#   124 - Timeout
#   139 - Segmentation fault
#   Others - Various crash types

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CORES_TO_TEST=(1 2 4 8 16 32 64)  
LOG_FILE="crash_test_results_$(date +%Y%m%d_%H%M%S).log"
TIMEOUT_SEC=30  # Timeout for each test

echo "TensTorrent Crash Test - Testing problem sizes from 2^4 to 2^30"
echo "============================================================="
echo "Log file: $LOG_FILE"
echo ""

# Header for results
printf "%-10s %-15s %-10s %-15s %-20s\n" "Power" "Size" "Cores" "Status" "Time (seconds)" | tee $LOG_FILE
printf "%-10s %-15s %-10s %-15s %-20s\n" "-----" "----" "-----" "------" "--------------" | tee -a $LOG_FILE

# Function to run a single test
run_test() {
    local power=$1
    local cores=$2
    local size=$((2**power))
    
    # Run the command with timeout
    start_time=$(date +%s.%N)
    
    # Redirect stderr to stdout to capture all output
    timeout $TIMEOUT_SEC bash -c "TT_METAL_DPRINT_CORES=all ./build/spatter --backend tenstorrent --tt-cores $cores -k gather -pUNIFORM:8:3 -l$size" &>/dev/null
    
    exit_code=$?
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc)
    
    # Determine status based on exit code
    if [ $exit_code -eq 0 ]; then
        status="${GREEN}SUCCESS${NC}"
        status_plain="SUCCESS"
    elif [ $exit_code -eq 1 ]; then
        status="${RED}VALIDATION_FAIL${NC}"
        status_plain="VALIDATION_FAIL"
    elif [ $exit_code -eq 124 ]; then
        status="${YELLOW}TIMEOUT${NC}"
        status_plain="TIMEOUT"
    elif [ $exit_code -eq 139 ]; then
        # 139 = 128 + 11 (SIGSEGV)
        status="${RED}SEGFAULT${NC}"
        status_plain="SEGFAULT"
    else
        status="${RED}CRASHED_$exit_code${NC}"
        status_plain="CRASHED_$exit_code"
    fi
    
    # Print and log result
    printf "%-10s %-15s %-10s " "2^$power" "$size" "$cores" | tee -a $LOG_FILE
    printf "${status}" 
    printf "%-15s %-20s\n" "$status_plain" "$duration" | tee -a $LOG_FILE
    
    return $exit_code
}

# Main test loop
for cores in "${CORES_TO_TEST[@]}"; do
    echo "" | tee -a $LOG_FILE
    echo "Testing with --tt-cores $cores" | tee -a $LOG_FILE
    echo "--------------------------------" | tee -a $LOG_FILE
    
    for power in {4..30}; do
        run_test $power $cores
        
        # Track crashes but don't interrupt the test
        # if [ $? -ne 0 ] && [ $? -ne 124 ]; then
        #     crash_count=$((crash_count + 1))
        # else
        #     crash_count=0
        # fi
    done
done

echo "" | tee -a $LOG_FILE
echo "=============================================================" | tee -a $LOG_FILE
echo "Test complete. Results saved to: $LOG_FILE" | tee -a $LOG_FILE

# Summary analysis
echo "" | tee -a $LOG_FILE
echo "Summary Analysis:" | tee -a $LOG_FILE
echo "-----------------" | tee -a $LOG_FILE

for cores in "${CORES_TO_TEST[@]}"; do
    success_count=$(grep "$cores.*SUCCESS" $LOG_FILE | wc -l)
    validation_fail_count=$(grep "$cores.*VALIDATION_FAIL" $LOG_FILE | wc -l)
    segfault_count=$(grep "$cores.*SEGFAULT" $LOG_FILE | wc -l)
    crash_count=$(grep "$cores.*CRASHED_" $LOG_FILE | wc -l)
    timeout_count=$(grep "$cores.*TIMEOUT" $LOG_FILE | wc -l)
    
    echo "Cores=$cores: Success=$success_count, ValidationFail=$validation_fail_count, Segfault=$segfault_count, Crashed=$crash_count, Timeout=$timeout_count" | tee -a $LOG_FILE
done

# Find boundaries where different failures start
echo "" | tee -a $LOG_FILE
echo "Failure boundaries:" | tee -a $LOG_FILE
for cores in "${CORES_TO_TEST[@]}"; do
    first_validation_fail=$(grep "$cores.*VALIDATION_FAIL" $LOG_FILE | head -1 | awk '{print $1}')
    first_segfault=$(grep "$cores.*SEGFAULT" $LOG_FILE | head -1 | awk '{print $1}')
    first_crash=$(grep "$cores.*CRASHED_" $LOG_FILE | head -1 | awk '{print $1}')
    
    if [ ! -z "$first_validation_fail" ] || [ ! -z "$first_segfault" ] || [ ! -z "$first_crash" ]; then
        echo "  Cores=$cores:" | tee -a $LOG_FILE
        [ ! -z "$first_validation_fail" ] && echo "    First validation fail at $first_validation_fail" | tee -a $LOG_FILE
        [ ! -z "$first_segfault" ] && echo "    First segfault at $first_segfault" | tee -a $LOG_FILE
        [ ! -z "$first_crash" ] && echo "    First crash at $first_crash" | tee -a $LOG_FILE
    else
        echo "  Cores=$cores: No failures detected" | tee -a $LOG_FILE
    fi
done