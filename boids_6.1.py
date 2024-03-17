from matplotlib import pyplot as plt
import pygame
import random
import math
import numpy as np
import time

# Define constants
WIDTH = 450
HEIGHT = 450
WALL_MARGIN = 50
NUM_AGENTS = 15
AGENT_SIZE = 5
BG_COLOR = (255, 255, 255)
AGENT_COLOR = (0, 0, 0)
AGENT_SPEED = 2  # Adjust speed as needed

# Define Agent class
class Agent:
    def __init__(self, x, y, params):
        self.xpos = x
        self.ypos = y
        # Generate random angle
        angle = random.uniform(0, 2*math.pi)
        # Calculate velocity components based on angle and speed
        self.xvel = AGENT_SPEED * math.cos(angle)
        self.yvel = AGENT_SPEED * math.sin(angle)

        # Boid hyperparameters
        self.cohesion = params[0]
        self.alignment = params[1]
        self.separation = params[2]

        self.neighbor_dist = 100  # Adjust neighbor distance as needed
        self.avoid_dist = 20  # Adjust avoid distance as needed
        self.fov_angle = 100 # How far back it can look up to 180 degrees
        self.turnfactor = 0.2

    def set_agents(self, agents):
        self.agents = agents

    def angle_between_agents(self, agent_pos):
        # Calculate vectors between the agents
        vec_agent1 = np.array([self.xvel, self.yvel])  # Velocity vector of agent 1
        vec_agent2 = np.array(agent_pos) - np.array([self.xpos, self.ypos])  # Vector from agent 1 to agent 2
        
        # If the agents are on top of each other, return 0
        if np.linalg.norm(vec_agent2) < AGENT_SIZE: 
            return 0

        # Calculate the angle between the two vectors
        dot_product = np.dot(vec_agent1, vec_agent2)
        norms = np.linalg.norm(vec_agent1) * np.linalg.norm(vec_agent2)

        angle_radians = np.arccos(np.clip(dot_product / norms, -1, 1)) # Clip between -1 and 1. Needed due to numerical inaccuracies

        # Convert angle from radians to degrees
        angle_degrees = np.degrees(angle_radians)
        
        return angle_degrees

    def get_neighbors(self, agents):
        neighbors = []
        for agent in agents:
            distance = math.sqrt((self.xpos - agent.xpos)**2 + (self.ypos - agent.ypos)**2)
            if distance < self.neighbor_dist:
                angle = self.angle_between_agents([agent.xpos, agent.ypos])
                if angle < self.fov_angle:
                    neighbors.append(agent)
        return neighbors 

    def move(self):
        # Based on the neighbors and cohesion, alignment, and separation values, compute where the agent moves
        neighbors = self.get_neighbors(self.agents)
        num_neighbors = len(neighbors)

        if num_neighbors > 0:
            # Calculate cohesion vector
            avg_x = 0
            avg_y = 0
            for neighbor in neighbors:
                avg_x += neighbor.xpos
                avg_y += neighbor.ypos
            avg_x /= num_neighbors
            avg_y /= num_neighbors
            cohesion_x = avg_x - self.xpos
            cohesion_y = avg_y - self.ypos

            # Calculate alignment vector
            alignment_x = 0
            alignment_y = 0
            for neighbor in neighbors:
                alignment_x += neighbor.xvel
                alignment_y += neighbor.yvel
            alignment_x /= num_neighbors
            alignment_y /= num_neighbors

            # Calculate separation vector
            separation_x = 0
            separation_y = 0
            for neighbor in neighbors:
                distance = math.sqrt((self.xpos - neighbor.xpos)**2 + (self.ypos - neighbor.ypos)**2)
                if distance < self.avoid_dist:
                    separation_x += self.xpos - neighbor.xpos
                    separation_y += self.ypos - neighbor.ypos

            # Calculate new velocity
            self.xvel += self.cohesion * cohesion_x + self.alignment * alignment_x + self.separation * separation_x
            self.yvel += self.cohesion * cohesion_y + self.alignment * alignment_y + self.separation * separation_y
        
        # Avoid going outside of the grid
        if self.xpos < WALL_MARGIN:
            self.xvel += self.turnfactor
        if self.xpos > WIDTH - WALL_MARGIN:
            self.xvel += -self.turnfactor
        if self.ypos < WALL_MARGIN:
            self.yvel += self.turnfactor
        if self.ypos > HEIGHT - WALL_MARGIN:
            self.yvel += - self.turnfactor
            
        # Normalize velocity
        magnitude = math.sqrt(self.xvel**2 + self.yvel**2)
        self.xvel *= AGENT_SPEED/magnitude
        self.yvel *= AGENT_SPEED/magnitude
        
        # Update position
        self.xpos += self.xvel
        self.ypos += self.yvel

    def draw(self, screen):
        direction_angle = math.atan2(self.yvel, self.xvel)
        
        # Punt van de driehoek in de richting van beweging
        front_point = (self.xpos + AGENT_SIZE * 2 * math.cos(direction_angle),
                    self.ypos + AGENT_SIZE * 2 * math.sin(direction_angle))
        
        # Achterpunten van de driehoek
        back_left = (self.xpos + AGENT_SIZE * math.cos(direction_angle + math.pi * 3/4),
                    self.ypos + AGENT_SIZE * math.sin(direction_angle + math.pi * 3/4))
        back_right = (self.xpos + AGENT_SIZE * math.cos(direction_angle - math.pi * 3/4),
                    self.ypos + AGENT_SIZE * math.sin(direction_angle-math.pi*3/4))
        
        pygame.draw.polygon(screen, AGENT_COLOR, [front_point, back_left, back_right])

class boids_sim:
    def __init__(self, pop_size, params) -> None:
        random.seed() # Ensure each sim starts the same
        self.pop_size = pop_size
        self.params = params
        self.agents = np.array([Agent(random.randint(0, WIDTH), random.randint(0, HEIGHT), self.params) for _ in range(self.pop_size)])
        for i, agent in enumerate(self.agents):
            agent.set_agents(self.agents[np.arange(pop_size) != i])
       
    def run(self, steps):
        order = []
        for _ in range(steps):
            for agent in self.agents:
                agent.move()
            order.append(self.compute_order(self.agents))
        return order

    def compute_order(self, agents):
        vx = 0
        vy = 0
        for agent in agents:
            vel_magnitude = math.sqrt(agent.xvel**2 + agent.yvel**2)
            vx += agent.xvel/vel_magnitude
            vy += agent.yvel/vel_magnitude
        return math.sqrt(vx**2 + vy**2)/len(agents)
    
    def compute_nearest_neighbor_distances(self):
        distances = []
        for agent in self.agents:
            min_dist = min([math.sqrt((agent.xpos - other.xpos) ** 2 + (agent.ypos - other.ypos) ** 2) for other in self.agents if other is not agent], default=0)
            distances.append(min_dist)
        return distances
    
    def draw_margin_box(self, screen):
        """Tekent een blauw vierkant van stippellijnen op de aangegeven wall margin locatie."""
        color = (0, 0, 255)  # Blauw
        dash_length = 10
        # Loop over de randen van het vierkant om stippellijnen te tekenen
        for x in range(WALL_MARGIN, WIDTH - WALL_MARGIN + 1, dash_length * 2):
            pygame.draw.line(screen, color, (x, WALL_MARGIN), (min(x + dash_length, WIDTH - WALL_MARGIN), WALL_MARGIN), 1)
            pygame.draw.line(screen, color, (x, HEIGHT - WALL_MARGIN), (min(x + dash_length, WIDTH - WALL_MARGIN), HEIGHT - WALL_MARGIN), 1)
        for y in range(WALL_MARGIN, HEIGHT - WALL_MARGIN + 1, dash_length * 2):
            pygame.draw.line(screen, color, (WALL_MARGIN, y), (WALL_MARGIN, min(y + dash_length, HEIGHT - WALL_MARGIN)), 1)
            pygame.draw.line(screen, color, (WIDTH - WALL_MARGIN, y), (WIDTH - WALL_MARGIN, min(y + dash_length, HEIGHT - WALL_MARGIN)), 1)
    
    def reinitialize_agents(self):
        for agent in self.agents:
            agent.xpos = random.randint(0, WIDTH)
            agent.ypos = random.randint(0, HEIGHT)
            angle = random.uniform(0, 2 * math.pi)
            agent.xvel = AGENT_SPEED * math.cos(angle)
            agent.yvel = AGENT_SPEED * math.sin(angle)

    def run_with_screen(self, steps):
        with open("boids_log.txt", "w") as log_file:
            start = time.time()
            # Initialize pygame
            pygame.init()
            screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Simple Agent Simulation")
            clock = pygame.time.Clock()
            
            order = []
            all_distances = [] 
            avg_distances = []

            for _ in range(steps):
                screen.fill(BG_COLOR)
                # Draw the margin box
                self.draw_margin_box(screen)

                distances = []
                for agent in self.agents:
                    agent.move()
                    agent.draw(screen)
                    # if its placed inside this loop, it will print the position of each agent at each step
                    log_file.write(f"Position: ({agent.xpos:.2f}, {agent.ypos:.2f}), Velocity: ({agent.xvel:.2f}, {agent.yvel:.2f})\n")
                    distances.append(self.compute_nearest_neighbor_distances())
                    
                order.append(self.compute_order(self.agents))
                all_distances.append(distances)
                avg_distances.append(np.mean(distances))

                # Event handling
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                pygame.display.flip()
                clock.tick(60)

            self.reinitialize_agents()
            pygame.quit()
            end = time.time()
            print("Elapsed time: ", end - start)

            return order, all_distances, avg_distances

            

# Cohesion, alignment, separation
sim_params = [
    [0.0005,0.1,0.05],
    [0.001,0.1,0.05],
    [0.0001,0.1,0.05],
    [0.0005,0.2,0.05],
    [0.0005,0.05,0.05],
    [0.0005,0.1,0.1],
    [0.0005,0.1,0.01]
]

number_of_runs = 10

for params in sim_params:
    sim = boids_sim(15, params)
    total_order = []
    total_all_distances = []
    total_avg_distances = []
    for i in range(number_of_runs):
        order, all_distances, avg_distances = sim.run_with_screen(300)
        total_order.append(order)
        total_all_distances.append(all_distances)
        total_avg_distances.append(avg_distances)

    # calculate the average for each of the 300 steps for every run 
    # so for every step, calculate the average of all the runs
    avg_order = np.mean(total_order, axis=0)
    avg_all_distances = np.mean(total_all_distances, axis=0)
    avg_avg_distances = np.mean(total_avg_distances, axis=0)

    
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Order', color=color)
    ax1.plot(avg_order, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Avg Nearest Neighbor Distance', color=color)  # we already handled the x-label with ax1
    ax2.plot(avg_avg_distances, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()





