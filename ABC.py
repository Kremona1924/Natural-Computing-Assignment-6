import numpy as np
from matplotlib import pyplot as plt
from boids_new import boids_sim
import time

class ABC:
    def __init__(self, pop_size, target_values) -> None:
        self.pop_size = pop_size
        self.target = target_values

        # ABC Hyperparameters
        self.tolerance = 0.8
        self.tolerance_decay = 0.9
        self.max_param_value = [0.1, 0.5, 0.1]
        self.sigma = [0.01, 0.05, 0.01]

        # Boids hyperparameters
        self.N_boids = 15
        self.N_steps = 300

        self.rej = 0 # Number of rejected samples
        self.pop = np.array([self.generate_individual() for i in range(self.pop_size)])

        # Open savefiles to remove any prior text
        open("population_over_time.txt", "w")
        open("fitness_over_time.txt", "w")

    def generate_individual(self):
        # Each individual consists of 3 numbers: [cohesion, alignment, separation]
        return np.random.uniform(0, self.max_param_value, 3)
    
    def mutate(self, x):
        return np.clip(x + np.random.normal(0, self.sigma, 3), [0,0,0], self.max_param_value)
    
    def fitness(self, params):
        # Initialise a boid simulator, run it, and return the last order value
        sim = boids_sim(self.N_boids, params)
        order = sim.run(self.N_steps)
        return np.mean(order[-30:])
    
    def save_pop(self):
        # Save the parameter values of the population. At each step,
        # all values of a single individual are printed to one line.
        # After printing the full population, an additional newline is added
        with open("population_over_time.txt", "a") as f:
            np.savetxt(f, self.pop)
            f.write("\n")
    
    def save_fitness(self, pop_fitness):
        # Save fitness values of the population. At each step, all fitness values are printed to one line
        with open("fitness_over_time.txt", "a") as f:
            np.savetxt(f,pop_fitness, newline=' ')
            f.write("\n")

    def step(self):
        k = 0
        self.rej = 0
        new_pop = np.empty((self.pop_size, 3))
        pop_fitness = np.empty((self.pop_size, 1))
        while k < self.pop_size:
            # Get a sample from the population, mutate it, and add it to the new pop if it is within the tolerance
            random_individual = self.pop[np.random.randint(0, self.pop_size), :]
            mutated_individual = self.mutate(random_individual)
            fitness = self.fitness(mutated_individual)
            if abs(fitness - self.target) < self.tolerance:
                new_pop[k] = mutated_individual
                pop_fitness[k] = fitness
                k += 1
            else:
                self.rej += 1
        print("rejections in step", self.current_step, " = ", self.rej)
        self.current_step += 1
        
        self.pop = new_pop
        self.save_pop()
        self.save_fitness(pop_fitness)

        self.tolerance *= self.tolerance_decay # Tolerance is reduced at each step to enforce convergence

    def run(self, num_steps):
        self.current_step = 0
        for _ in range(num_steps):
            self.step()
        print("Total rejections = ", self.rej)
        return self.pop

target = 0.6
N = 20
abc = ABC(pop_size=N, target_values=target)
final_pop = abc.run(300)

fig, axs = plt.subplots(1,3)
for i, ax in enumerate(axs):
    ax.hist(final_pop[:,i], bins=5)

plt.show()