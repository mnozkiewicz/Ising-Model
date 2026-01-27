import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import numba
import click
from time import perf_counter

@numba.jit(nopython=True, nogil=True, cache=True)
def metropolis_phase(
        n: int,
        lattice,
        J: float,
        beta: float,
        task: np.ndarray,
        start,
        end
    ):

    grid_numbers = task[start:end]
    for number in grid_numbers:
        i = number // n
        j = number % n
        spin = lattice[i, j]
        neighbor_sum = (
            lattice[(i + 1) % n, j] +
            lattice[(i - 1) % n, j] +
            lattice[i, (j + 1) % n] +
            lattice[i, (j - 1) % n]
        )

        dE = 2 * J * spin * neighbor_sum
        if dE <= 0:
            lattice[i, j] *= -1
        elif np.random.random() < np.exp(-dE * beta):
            lattice[i, j] *= -1
            
        

@click.command()
@click.option("--n", type=int, default=200, help="Grid size (NxN)")
@click.option("--total_flips", type=int, default=100_000_000, help="Total attempts across ALL processes")
@click.option("--iterations", type=int, default=10, help="Number of synchronization steps")
@click.option("--kt", type=float, default=2.3, help="Temperature")
@click.option("--density", type=float, default=0.5, help="Initial Density of white pixels")
@click.option("--verbose", type=bool, default=False)
def main(n: int, total_flips: int, iterations: int, kt: float, density: float, verbose: bool):
    

    comm = MPI.COMM_WORLD
    shm_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    rank = shm_comm.Get_rank()
    size = shm_comm.Get_size()

    flips_per_process = total_flips // size
    flips_per_synchronization_loop = flips_per_process // iterations


    nbytes = (n * n) * np.dtype(np.float64).itemsize if rank == 0 else 0
    win = MPI.Win.Allocate_shared(nbytes, np.dtype(np.float64).itemsize, comm=shm_comm)
    buf, _ = win.Shared_query(0)
    lattice = np.ndarray(buffer=buf, dtype=np.float64, shape=(n, n))

    # task buffer 
    nbytes = flips_per_synchronization_loop * np.dtype(np.int32).itemsize if rank == 0 else 0
    win2 = MPI.Win.Allocate_shared(nbytes, np.dtype(np.int32).itemsize, comm=shm_comm)
    buf2, _ = win2.Shared_query(0)
    tasks = np.ndarray(buffer=buf2, dtype=np.int32, shape=(flips_per_synchronization_loop,))
    

    if rank == 0:
        np.random.seed(0)
        init_data = np.random.choice([1.0, -1.0], size=(n, n), p=[density, 1 - density])
        lattice[:] = init_data[:]
        
    shm_comm.Barrier()
    
    
        
    J = 1.0
    beta = 1.0 / kt
    
    start_time = perf_counter()
    
    for i in range(iterations):
        if rank == 0:
            current_seed = (rank * 100000) + i
            np.random.seed(current_seed)
            grid_numbers = np.random.choice(n*n, size=flips_per_synchronization_loop, replace=False).astype(np.int32)
            
            black_numbers_mask = (grid_numbers % 2).astype(bool)
            black_numbers:np.ndarray = grid_numbers[black_numbers_mask]
            red_numbers = grid_numbers[~black_numbers_mask]
            

            task_size = black_numbers.shape[0]
            tasks[:task_size] = black_numbers[:]


            for i in range(1, size):
                comm.send(task_size, dest=i)
        else:
            task_size = comm.recv(source=0)

        per_node_tasks = task_size // size
        start = rank * per_node_tasks
        end = (rank + 1) * per_node_tasks if rank < n - 1 else task_size
                        
        metropolis_phase(
            n, lattice, J, beta, tasks, start, end
        )
        shm_comm.Barrier()

        if rank == 0:
            task_size = red_numbers.shape[0]
            tasks[:task_size] = red_numbers[:]

            for i in range(1, size):
                comm.send(task_size, dest=i)
        else:
            task_size = comm.recv(source=0)

        per_node_tasks = task_size // size
        start = rank * per_node_tasks
        end = (rank + 1) * per_node_tasks if rank < n - 1 else task_size
        
        
        metropolis_phase(
            n, lattice, J, beta, tasks, start, end
        )
        shm_comm.Barrier()

    end_time = perf_counter()

    # magnetizations = []
    # for step in range(100_000 + 1):
    #     i, j = np.random.randint(0, n), np.random.randint(0, n)
        
    #     spin = lattice[i, j]
    #     neighbor_sum = (
    #         lattice[(i + 1) % n, j] +
    #         lattice[(i - 1) % n, j] +
    #         lattice[i, (j + 1) % n] +
    #         lattice[i, (j - 1) % n]
    #     )

    #     dE = 2 * J * spin * neighbor_sum
    #     if dE <= 0:
    #         lattice[i, j] *= -1
    #     elif np.random.random() < np.exp(-dE * beta):
    #         lattice[i, j] *= -1

    #     avg_mag = np.abs(np.sum(lattice)) / (n * n)
    #     if step % 1000 == 0:
    #         magnetizations.append(avg_mag)
    
    print(f"{(end_time - start_time):.3f}")
    if rank == 0 and verbose:

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.imshow(lattice, cmap='gray', interpolation='nearest')
        ax1.set_title(f"Final Configuration (T={kt})")
        ax1.axis('off')

        ax2.plot(magnetizations, linewidth=0.5, color='blue')
        ax2.set_title(f"Magnetization Fluctuations during Measurement")
        ax2.set_xlabel("Sample Points")
        ax2.set_ylabel("|Magnetization|")
        ax2.set_ylim(0, 1)
        ax2.grid(True, linestyle='--', linewidth=0.5)


        mean_magnetization = np.mean(magnetizations)
        ax3.hist(magnetizations, bins=30, color='skyblue', edgecolor='black')
        ax3.axvline(mean_magnetization, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_magnetization:.4f}')
        ax3.set_title("Histogram of Average Magnetization")
        ax3.set_xlabel("Average Magnetization")
        ax3.set_ylabel("Count")
        ax3.legend()

        plt.savefig("../screenshots/ising_sequential_random.png")
        plt.tight_layout()
        plt.show()
            
    win.Free()
    win2.Free()

if __name__ == '__main__':
    main()