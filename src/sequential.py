import numpy as np
import matplotlib.pyplot as plt
import numba
import click
from time import perf_counter


@numba.jit(nopython=True, nogil=True)
def metropolis_phase(
        n: int,
        lattice: np.ndarray,
        J: float,
        beta: float,
        n_samples: int,
    ):
    
    for _ in range(n_samples):
        i, j = np.random.randint(0, n), np.random.randint(0, n)
        
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
@click.option("--total_flips", type=int, default=100_000_000, help="Total flip attempts")
@click.option("--kt", type=float, default=2.3, help="Temperature")
@click.option("--density", type=float, default=0.5, help="Initial Density of white pixels")
@click.option("--verbose", type=bool, default=False)
def main(n: int, total_flips: int, kt: float, density: float, verbose: bool):
    


    np.random.seed(0)
    init_data = np.random.choice([1.0, -1.0], size=(n, n), p=[density, 1 - density])
    
    J = 1.0
    beta = 1.0 / kt
    
    start_time = perf_counter()
    metropolis_phase(n, init_data, J, beta, total_flips)
    end_time = perf_counter()

    magnetizations = []
    for step in range(100_000 + 1):
        i, j = np.random.randint(0, n), np.random.randint(0, n)
        
        spin = init_data[i, j]
        neighbor_sum = (
            init_data[(i + 1) % n, j] +
            init_data[(i - 1) % n, j] +
            init_data[i, (j + 1) % n] +
            init_data[i, (j - 1) % n]
        )

        dE = 2 * J * spin * neighbor_sum
        if dE <= 0:
            init_data[i, j] *= -1
        elif np.random.random() < np.exp(-dE * beta):
            init_data[i, j] *= -1

        avg_mag = np.abs(np.sum(init_data)) / (n * n)
        if step % 1000 == 0:
            magnetizations.append(avg_mag)

    print(f"{(end_time - start_time):.4f}")
        
    if verbose:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.imshow(init_data, cmap='gray', interpolation='nearest')
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
            

if __name__ == '__main__':
    main()