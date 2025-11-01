import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

class Particle:

    """This class defines functions for the simulation"""


    def __init__(self, mass, positionx, positiony, velocityx, velocityy, accelerationx, accelerationy):
        self.mass = mass
        self.positionx = positionx
        self.positiony = positiony
        self.velocityx = velocityx
        self.velocityy = velocityy
        self.accelerationx = accelerationx
        self.accelerationy = accelerationy

    def move(self, time):
        # Cap the velocities to prevent overflow in calculations
        max_speed = 50 
        speed = (self.velocityx**2 + self.velocityy**2)**0.5
        if speed > max_speed:
            ratio = max_speed / speed
            self.velocityx *= ratio
            self.velocityy *= ratio
        
        self.positionx += self.velocityx * time + 0.5 * self.accelerationx * time**2
        self.positiony += self.velocityy * time + 0.5 * self.accelerationy * time**2
        self.velocityx += self.accelerationx * time
        self.velocityy += self.accelerationy * time

    def colision(self):
        particle_radius = 5
        coefficient_of_restitution = 0.99  # A value between 0 and 1
        
        # Collision with left wall
        if self.positionx <= particle_radius:
            self.velocityx = -self.velocityx * coefficient_of_restitution
            self.positionx = particle_radius
        # Collision with right wall
        if self.positionx >= 200 - particle_radius:
            self.velocityx = -self.velocityx * coefficient_of_restitution
            self.positionx = 200 - particle_radius
        # Collision with bottom wall
        if self.positiony <= particle_radius:
            self.velocityy = -self.velocityy * coefficient_of_restitution
            self.positiony = particle_radius
        # Collision with top wall
        if self.positiony >= 200 - particle_radius:
            self.velocityy = -self.velocityy * coefficient_of_restitution
            self.positiony = 200 - particle_radius

    def kinetic_energy(self):
        return 0.5 * self.mass * (self.velocityx**2 + self.velocityy**2)
    
    def colisionparticle(self, other):
        # The collision distance is the sum of the radii.
        # Assuming both particles have a radius of 5, the collision distance is 10.
        collision_distance = 10.0
        coefficient_of_restitution = 0.99  # A value between 0 and 1

        dx = other.positionx - self.positionx
        dy = other.positiony - self.positiony
        distance = (dx**2 + dy**2)**0.5

        if distance <= collision_distance:
            # 1. Separate the particles to prevent sticking
            overlap = collision_distance - distance
            
            # If the distance is zero, prevent division by zero
            if distance == 0:
                self.positionx -= overlap / 2
                other.positionx += overlap / 2
            else:
                separation_x = (dx / distance) * overlap / 2
                separation_y = (dy / distance) * overlap / 2
                self.positionx -= separation_x
                self.positiony -= separation_y
                other.positionx += separation_x
                other.positiony += separation_y

            # 2. Calculate and apply the impulse
            
            # Unit normal vector (direction of impulse)
            nx = dx / distance
            ny = dy / distance

            # Relative velocity
            dvx = other.velocityx - self.velocityx
            dvy = other.velocityy - self.velocityy
            
            # Relative velocity along the normal vector
            relative_velocity_on_normal = dvx * nx + dvy * ny
            
            # Impulse magnitude calculation
            j = -(1 + coefficient_of_restitution) * relative_velocity_on_normal / (1/self.mass + 1/other.mass)
            
            # Apply impulse to update velocities
            self.velocityx -= j * nx / self.mass
            self.velocityy -= j * ny / self.mass
            other.velocityx += j * nx / other.mass
            other.velocityy += j * ny / other.mass

# Number of particles
N = 20
particles = []

# Create N particles with random initial conditions
for _ in range(N):
    p = Particle(
        1,
        random.uniform(10, 190),    # random x
        random.uniform(10, 190),    # random y
        random.uniform(-10, 10),   # Reduced velocity range to prevent overflow
        random.uniform(-10, 10),   # Reduced velocity range to prevent overflow
        0, -240
    )
    particles.append(p)

# Set up figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
plt.tight_layout(pad=3.0)  # Add padding between subplots

# Subplot 1: Particle Simulation
ax1.set_xlim(0, 200)
ax1.set_ylim(0, 200)
ax1.set_title("Particle Simulation")
ax1.set_xlabel("X-position")
ax1.set_ylabel("Y-position")

# Points to animate (different colors for each particle)
colors = plt.cm.tab10.colors  # 10 distinct colors
points = [
    ax1.plot([], [], 'o', color=colors[i % len(colors)], markersize=15)[0]
    for i in range(N)
]
ax1.legend(loc="upper right", fontsize=8)

# Add dynamic text for total kinetic energy
energy_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=10, color="black")

# Subplot 2: Kinetic Energy Plot
time_data = []
ke_data = []
line, = ax2.plot([], [], 'b-')
ax2.grid(True)
ax2.set_title("Total Kinetic Energy Over Time")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Total Kinetic Energy")


# Update function
def update(frame):
    dt = 0.01
    total_energy = 0
    
    # Nested loops for all-particle collision checks
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            particles[i].colisionparticle(particles[j])

    # After checking for all collisions, move all particles
    for i, p in enumerate(particles):
        p.move(dt)
        p.colision() # Boundary collision check
        points[i].set_data([p.positionx], [p.positiony])
        total_energy += p.kinetic_energy()
    
    # Update KE text
    energy_text.set_text(f"Total KE: {total_energy:.2f}")

    # Update kinetic energy plot data
    time_data.append(frame)
    ke_data.append(total_energy)
    line.set_data(time_data, ke_data)
    
    # Dynamically adjust the limits of the kinetic energy plot
    ax2.set_xlim(0, 10000)
    ax2.set_ylim(min(ke_data)*0.85, max(ke_data) * 1.15)

    return points + [energy_text, line]

# Animation
ani = animation.FuncAnimation(fig, update, frames=100000, interval=1, blit=True)

plt.show()
