#!/bin/bash

NPROC_LIST=(2 3 4 5 6 7 8 9 10 11 12)

# params: 
#   n - grid size
#   total_flips - total flip attempts during the algorithm
#   iterations - number of synchronization steps (this can be fixed to 100)
#   kt - for testing we may assume 2.3
#   verbose - always False (we don't pass it for testing)
PARAMS_LIST=(
    "200 10000000"
    "500 10000000"
    "200 100000000"
    "500 100000000"
)

OUTFILE="ising_times.csv"
echo "grid_size,total_flips,nproc,time_sec" > $OUTFILE

for PARAMS in "${PARAMS_LIST[@]}"; do
    read grid_size total_flips  <<< "$PARAMS"
    
    DURATION_BASE=$(uv run src/sequential.py --n $grid_size --total_flips $total_flips | tail -n 1)
    echo "$grid_size,$total_flips,1,$DURATION_BASE" | tee -a $OUTFILE

    for NPROC in "${NPROC_LIST[@]}"; do
        
        echo "Running n=$grid_size, total_flips=$total_flips with $NPROC MPI processes"

        DURATION=$(mpiexec -n $NPROC uv run src/partition_plus_sm.py \
            --n $grid_size --total_flips $total_flips | tail -n 1)

        echo "$grid_size,$total_flips,$NPROC,$DURATION" | tee -a $OUTFILE
    done
done