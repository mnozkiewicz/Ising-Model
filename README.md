# Ising-Model

## Setting up

To setup the project, you will need [uv](https://github.com/astral-sh/uv). You can download it via `pip` or `pipx`. After instalation you can run:

```
uv sync
source .venv/bin/activate
python main.py
```


Also [open-mpi](https://www.open-mpi.org/) is required.


## Scripts and notebooks

[visualisation.ipynb](./notebooks/visualisation.ipynb) - This notebook provides the theoretical background for the Ising Model and enables the generation of visualizations and GIFs.

[sequential.py](./src/sequential.py) - Contains a single-threaded (sequential) implementation of the Metropolis algorithm. To view available parameters, you can run:
```bash
uv run src/sequential.py --help
```

[partition_plus_sm.py](./src/partition_plus_sm.py) - Contains a multi-process implementation of the Metropolis algorithm using the mpi4py library (a Python wrapper for OpenMPI). The algorithm partitions the grid evenly among worker processes and utilizes shared memory. To run the script:

```bash
mpiexec -n $NUMBER_OF_PARALLEL_PROCESSES uv run src/partition_plus_sm.py
```

To check possible parameters:
```bash
uv run src/partition_plus_sm.py --help
```

[test_script.sh](./test_script.sh) - A Bash script used to benchmark the parallel method. It iterates over various input combinations, executes the single-threaded program to establish a baseline, and then runs the multi-process version to examine the impact of increasing the number of processes. Time measurements are saved to [ising_times.csv](./ising_times.csv) file. To run the script:

```bash
chmod +x test_script.sh
./test_script.sh
```

[metrics.ipynb](./notebooks/metrics.ipynb) - A notebook analyzing the results from [ising_times.csv](./ising_times.csv). t plots standard metrics for benchmarking parallel programs.