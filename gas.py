import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math
import numpy as np
from matplotlib.widgets import Button

# --- GLOBAL CONSTANTS / VARIABLES ---
BOX_SIZE = 200.0
PARTICLE_RADIUS = 5.0
COEFFICIENT_OF_RESTITUTION = 1.0
N = 30 
GLOBAL_TIME = 0.0
TINY_TIME = 1e-9
NUM_BINS = 40 
TINY_SEPARATION = 1e-7 
WINDOW_SIZE = 1000 
KE_STOP_THRESHOLD = 0.001 

# --- NEW CONSTANT FOR VISUAL SPEED ---
UPDATES_PER_FRAME = 120 
DT_COOLING = 0.01 

# --- GLOBAL DYNAMIC VARIABLES ---
SPEEDS_HISTORY = [] 
MAX_SPEED_GLOBAL = 3.0 
T_INITIAL_FIXED = 0.0
SIMULATION_ACTIVE = True 
COOLING_STEPS_REMAINING = 0 
# ------------------------------------

# --- Maxwell-Boltzmann Function (2D) ---
def maxwell_boltzmann_2d(v, T):
    """
    2D Maxwell-Boltzmann PDF for speed v.
    """
    if T <= 1e-9: return 0.0
    return (v / T) * np.exp(-v**2 / (2 * T))

class Particle:
    """This class defines functions for the event-driven simulation"""

    def __init__(self, mass, positionx, positiony, velocityx, velocityy):
        self.mass = mass
        self.positionx = positionx
        self.positiony = positiony
        self.velocityx = velocityx
        self.velocityy = velocityy
        
        # EDMD properties
        self.next_wall_time = float('inf') 
        self.next_particle_time = float('inf')
        self.next_partner = None

    def move(self, dt):
        """Moves the particle based on its constant velocity for time dt."""
        self.positionx += self.velocityx * dt
        self.positiony += self.velocityy * dt

    def time_to_wall_collision(self):
        """Calculates the time until the particle hits any boundary."""
        
        if self.velocityx > 0:
            tx = (BOX_SIZE - PARTICLE_RADIUS - self.positionx) / self.velocityx
        elif self.velocityx < 0:
            tx = (PARTICLE_RADIUS - self.positionx) / self.velocityx
        else:
            tx = float('inf')

        if self.velocityy > 0:
            ty = (BOX_SIZE - PARTICLE_RADIUS - self.positiony) / self.velocityy
        elif self.velocityy < 0:
            ty = (PARTICLE_RADIUS - self.positiony) / self.velocityy
        else:
            ty = float('inf')
            
        return min(t for t in [tx, ty] if t > TINY_TIME)

    def resolve_wall_collision(self):
        """Resolves collision with the wall."""
        global COEFFICIENT_OF_RESTITUTION
        e = COEFFICIENT_OF_RESTITUTION
        
        if self.positionx <= PARTICLE_RADIUS + TINY_TIME or self.positionx >= BOX_SIZE - PARTICLE_RADIUS - TINY_TIME:
            self.velocityx = -self.velocityx * e 
            if self.positionx < PARTICLE_RADIUS: self.positionx = PARTICLE_RADIUS
            if self.positionx > BOX_SIZE - PARTICLE_RADIUS: self.positionx = BOX_SIZE - PARTICLE_RADIUS
            
        if self.positiony <= PARTICLE_RADIUS + TINY_TIME or self.positiony >= BOX_SIZE - PARTICLE_RADIUS - TINY_TIME:
            self.velocityy = -self.velocityy * e 
            if self.positiony < PARTICLE_RADIUS: self.positiony = PARTICLE_RADIUS
            if self.positiony > BOX_SIZE - PARTICLE_RADIUS: self.positiony = BOX_SIZE - PARTICLE_RADIUS
            
    def get_speed(self):
        """Returns the absolute value of the velocity (the speed)."""
        return math.sqrt(self.velocityx**2 + self.velocityy**2)

    def time_to_particle_collision(self, other):
        """Calculates the time until collision between this particle and 'other' using the EDMD formalism."""
        
        dx = other.positionx - self.positionx
        dy = other.positiony - self.positiony
        dvx = other.velocityx - self.velocityx
        dvy = other.velocityy - self.velocityy
        
        A = dvx**2 + dvy**2
        B = 2 * (dx * dvx + dy * dvy)
        C = dx**2 + dy**2 - (2 * PARTICLE_RADIUS)**2

        if A < TINY_TIME or B > 0:
            return float('inf')
        
        discriminant = B**2 - 4 * A * C
        
        if discriminant < 0:
            return float('inf')
        
        t = (-B - math.sqrt(discriminant)) / (2 * A)
        
        if t > TINY_TIME:
            return t
        
        return float('inf')

    def resolve_particle_collision(self, other):
        """Resolves the collision between this particle and 'other' using impulse."""
        global COEFFICIENT_OF_RESTITUTION
        e = COEFFICIENT_OF_RESTITUTION

        distance_sq = (other.positionx - self.positionx)**2 + (other.positiony - self.positiony)**2
        distance = math.sqrt(distance_sq)
        
        if distance < TINY_TIME or distance > 2 * PARTICLE_RADIUS + TINY_TIME:
             return

        nx = (other.positionx - self.positionx) / distance
        ny = (other.positiony - self.positiony) / distance

        v_dot_n = (other.velocityx - self.velocityx) * nx + (other.velocityy - self.velocityy) * ny
        
        j = -(1 + e) * v_dot_n / (1/self.mass + 1/other.mass)

        self.velocityx -= j * nx / self.mass
        self.velocityy -= j * ny / self.mass
        other.velocityx += j * nx / other.mass
        other.velocityy += j * ny / other.mass

        self.positionx -= TINY_SEPARATION * nx
        self.positiony -= TINY_SEPARATION * ny
        other.positionx += TINY_SEPARATION * nx
        other.positiony += TINY_SEPARATION * ny


# --- EDMD INITIALIZATION ---
particles = []

for _ in range(N):
    p = Particle(
        1.0,
        random.uniform(PARTICLE_RADIUS * 2, BOX_SIZE - PARTICLE_RADIUS * 2),
        random.uniform(PARTICLE_RADIUS * 2, BOX_SIZE - PARTICLE_RADIUS * 2),
        random.uniform(-1, 1),
        random.uniform(-1, 1)
    )
    particles.append(p)

def recalculate_events(p_index):
    """Calculates all future event times for a given particle and updates the queue."""
    p = particles[p_index]
    
    p.next_wall_time = p.time_to_wall_collision()

    min_time = float('inf')
    min_partner = None

    for j in range(len(particles)):
        if p_index == j: continue
        
        p2 = particles[j]
        t = p.time_to_particle_collision(p2)
        
        if t < min_time:
            min_time = t
            min_partner = j
    
    p.next_particle_time = min_time
    p.next_partner = min_partner


# Initialize all events at t=0
for i in range(N):
    recalculate_events(i)

# --- Calculate Fixed Initial Temperature ---
initial_ke_total = sum(0.5 * p.mass * (p.velocityx**2 + p.velocityy**2) for p in particles)
T_INITIAL_FIXED = initial_ke_total / N 
KE_Y_LIMIT = initial_ke_total * 1.1 # Calculate the fixed KE Y-limit
# ------------------------------------------

# --- NEW: Button Callback Function ---
def set_restitution(event):
    """Callback to change COEFFICIENT_OF_RESTITUTION to 0.9 and initiate fixed-step cooling."""
    global COEFFICIENT_OF_RESTITUTION, SPEEDS_HISTORY, SIMULATION_ACTIVE, COOLING_STEPS_REMAINING
    
    COEFFICIENT_OF_RESTITUTION = 0.9
    SIMULATION_ACTIVE = True
    
    # Set the total number of physics steps for the cooling period
    COOLING_STEPS_REMAINING = 5000 * PHYSICS_STEPS_PER_FRAME
    
    print(f"\n--- COEFFICIENT OF RESTITUTION changed to {COEFFICIENT_OF_RESTITUTION} (Inelastic) ---")
    print(f"--- Entering HYBRID fixed-step mode for {COOLING_STEPS_REMAINING} total steps ---")
    
    SPEEDS_HISTORY.clear()
    
    for i in range(len(particles)):
        recalculate_events(i) 
    
    fig.canvas.draw_idle()


# --- VISUALIZATION SETUP ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))
plt.tight_layout(pad=3.0)

try:
    plt.get_current_fig_manager().window.showMaximized()
except Exception:
    pass

# --- Button Setup ---
button_ax = fig.add_axes([0.75, 0.01, 0.20, 0.03])
button = Button(button_ax, 'Set e = 0.9 (Cooling)', color='salmon', hovercolor='red')
button.on_clicked(set_restitution)
# -------------------------

# Subplot 1: Particle Simulation
ax1.set_xlim(0, BOX_SIZE)
ax1.set_ylim(0, BOX_SIZE)
ax1.set_title("Particle Simulation (EDMD)")
ax1.set_xlabel("X-position")
ax1.set_ylabel("Y-position")

colors = plt.cm.tab10.colors
points = [
    ax1.plot([], [], 'o', color=colors[i % len(colors)], markersize=15)[0]
    for i in range(N)
]
energy_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=10, color="black")

# Subplot 2: Kinetic Energy Plot (FIXED Y-AXIS APPLIED)
time_data = [0.0]
ke_data = [initial_ke_total]
line, = ax2.plot(time_data, ke_data, 'b-')
ax2.grid(True)
ax2.set_title("Total Kinetic Energy Over Time")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Total Kinetic Energy")
ax2.set_ylim(0, KE_Y_LIMIT) # <-- FIXED Y-LIMIT

# --- NEW SUBPLOT: Velocity Histogram (Frequency PDF) ---
initial_speeds = [p.get_speed() for p in particles]
MAX_SPEED_GLOBAL = 3.0 
bins = np.linspace(0, MAX_SPEED_GLOBAL, NUM_BINS)

counts, bins, hist_patches = ax3.hist(
    initial_speeds,
    bins=bins,
    color='teal',
    edgecolor='black',
    cumulative=False,
    histtype='bar'
)
hist_bars = hist_patches

# Calculate Theoretical Curve ONCE using T_0
v_plot = np.linspace(0, MAX_SPEED_GLOBAL, 200) 
theoretical_pdf = maxwell_boltzmann_2d(v_plot, T_INITIAL_FIXED)
total_samples_in_window = WINDOW_SIZE * N
bin_width = MAX_SPEED_GLOBAL / NUM_BINS
normalized_curve_fixed = theoretical_pdf * total_samples_in_window * bin_width

# Find the maximum height of the theoretical curve for Y-axis scaling
theoretical_max_height = np.max(normalized_curve_fixed)

# Initialize the theoretical Maxwell-Boltzmann line (red dashed)
maxwell_line, = ax3.plot(v_plot, normalized_curve_fixed, 'r--', linewidth=1.5, label=f"Theory (T={T_INITIAL_FIXED:.3f})")

ax3.set_title("Speed Distribution (Frequency PDF)")
ax3.set_xlabel("Speed (|v|)")
ax3.set_ylabel("") 
ax3.set_xlim(0, MAX_SPEED_GLOBAL)
ax3.grid(True, alpha=0.3) 
ax3.set_yticklabels([]) 
ax3.set_ylim(0, theoretical_max_height * 1.1) 

# --- Matplotlib Resize Event Handler ---
def on_resize(event):
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('resize_event', on_resize)
# ------------------------------------------


# --- EDMD Update Function ---
def update(frame):
    global GLOBAL_TIME, time_data, ke_data, MAX_SPEED_GLOBAL, SPEEDS_HISTORY, SIMULATION_ACTIVE, COOLING_STEPS_REMAINING, DT_COOLING, PHYSICS_STEPS_PER_FRAME
    
    if not SIMULATION_ACTIVE:
        return points + [energy_text, line] + list(hist_bars) + [maxwell_line]

    # --- START OF HYBRID MODE LOGIC ---
    if COOLING_STEPS_REMAINING > 0:
        # FIXED TIME STEP: Run physics loop multiple times before one redraw
        
        dt_physics = DT_COOLING
        total_dt_advanced = 0
        
        # Run physics UPDATES_PER_FRAME times
        for _ in range(PHYSICS_STEPS_PER_FRAME):
            if COOLING_STEPS_REMAINING <= 0: break
            
            # Collision Check and Resolution (Fixed-step logic)
            for i in range(N):
                for j in range(i + 1, N):
                    particles[i].resolve_particle_collision(particles[j]) 
            
            # Movement and Wall Collision (Fixed-step logic)
            for i, p in enumerate(particles):
                p.move(dt_physics) 
                p.resolve_wall_collision() 
                
            COOLING_STEPS_REMAINING -= 1
            total_dt_advanced += dt_physics
        
        min_time = total_dt_advanced # This is the time advanced since the last redraw
        
        if COOLING_STEPS_REMAINING <= 0:
            print("--- Exiting FIXED-TIME STEP mode. Re-entering EDMD. ---")
            for i in range(N):
                recalculate_events(i)
                
    else:
        # --- EDMD MODE (for physical accuracy) ---
        
        # 1. Find the Next Event
        min_time = float('inf')
        event_type = None
        p_index = None
        p2_index = None

        for i in range(N):
            p = particles[i]

            if p.next_wall_time < min_time:
                min_time = p.next_wall_time
                event_type = 'wall'
                p_index = i
                p2_index = None

            if p.next_particle_time < min_time:
                min_time = p.next_particle_time
                event_type = 'particle'
                p_index = i
                p2_index = p.next_partner
        
        # Check for stall
        if min_time == float('inf') or min_time < TINY_TIME:
            SIMULATION_ACTIVE = False
            print("Simulation stopped: Numerical instability or total energy depletion.")
            for p in points: p.set_data([], [])
            return points + [energy_text, line] + list(hist_bars) + [maxwell_line]
        
        # 2. Advance Time and Move All Particles (EDMD Step)
        for p in particles:
            p.move(min_time)
            p.next_wall_time -= min_time
            p.next_particle_time -= min_time
            
        # 3. Resolve Event (EDMD Step)
        p1 = particles[p_index]
        if event_type == 'wall':
            p1.resolve_wall_collision()
            recalculate_events(p_index) 
        elif event_type == 'particle':
            p2 = particles[p2_index]
            p1.resolve_particle_collision(p2)
            recalculate_events(p_index)
            recalculate_events(p2_index)
        
    # --- END OF HYBRID MODE LOGIC ---
    
    GLOBAL_TIME += min_time # Total simulation time always advances
    
    # --- 4. Visualization Update ---
    total_energy = sum(0.5 * p.mass * (p.velocityx**2 + p.velocityy**2) for p in particles)
    
    time_data.append(GLOBAL_TIME)
    ke_data.append(total_energy)
    
    # Update particle positions
    for i, p in enumerate(particles):
        points[i].set_data([p.positionx], [p.positiony])
    
    energy_text.set_text(f"Total KE: {total_energy:.2f}")
    
    # Update KE plot
    line.set_data(time_data, ke_data)
    ax2.set_xlim(0, GLOBAL_TIME * 1.05 if GLOBAL_TIME > 1 else 1)
    ax2.set_ylim(0, KE_Y_LIMIT) # <-- FIXED Y-LIMIT
    
    # --- HISTOGRAM UPDATE ---
    
    current_speeds = [p.get_speed() for p in particles]
    SPEEDS_HISTORY.extend(current_speeds)
    
    if len(SPEEDS_HISTORY) > WINDOW_SIZE * N:
        SPEEDS_HISTORY = SPEEDS_HISTORY[-WINDOW_SIZE * N:]
    
    bins = np.linspace(0, MAX_SPEED_GLOBAL, NUM_BINS)
    counts, _ = np.histogram(SPEEDS_HISTORY, bins=bins)
    
    for bar, count in zip(hist_bars, counts):
        bar.set_height(count)
        
    ax3.set_ylim(0, theoretical_max_height * 1.1) 

    # Return all artists for blitting
    return points + [energy_text, line] + list(hist_bars) + [maxwell_line]


# Animation
ani = animation.FuncAnimation(fig, update, frames=50000, interval=1, blit=True)

plt.show()