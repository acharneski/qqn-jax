#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# List of reports to run.
# Comment out (prefix with #) any line to disable that report.
REPORTS=(
#    "mnist_comparison"
    "fashion_mnist_mlp_comparison"
#    "mnist_sparse_benchmark"
)

for REPORT in "${REPORTS[@]}"; do
    LOGFILE="results/${REPORT}_${TIMESTAMP}.log"
    echo "=== Running ${REPORT} ==="
    python3 "./examples/${REPORT}.py" | tee "${LOGFILE}"
done