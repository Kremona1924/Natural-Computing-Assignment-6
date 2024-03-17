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
        #random.seed(1) # Ensure each sim starts the same
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


    def run_with_screen(self, steps):
        # Initialize pygame
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Simple Agent Simulation")
        clock = pygame.time.Clock()
        
        order = []
        for _ in range(steps):
            screen.fill(BG_COLOR)
            for agent in self.agents:
                agent.move()
                agent.draw(screen)

            order.append(self.compute_order(self.agents))

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            pygame.display.flip()
            clock.tick(60)

        plt.plot(order)
        plt.show()
        pygame.quit()

# # Uncomment to run with screen
# sim = boids_sim(15, [0.01,0.3,0.1])
# sim.run_with_screen(300)