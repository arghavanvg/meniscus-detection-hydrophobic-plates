import mdtraj as md
import numpy as np
# from scipy.spatial.distance import pdist, squareform
# import parmed as pmd
from warnings import filterwarnings
filterwarnings('ignore')
# import logging
import os
import sys

t = int(sys.argv[1])
d = sys.argv[2]

input_path = f'/Users/arghavan/Graduate Center Dropbox/Arghavan Vedadi Gargari/MyFiles/{t}K/{d}/'
output_path = f'/Users/arghavan/lab/hydrophobic_plates/results/{t}K/{d}/'
mean_nbr_path = f'{input_path}all_frame_mean_neighbours.pkl'
Findx_path = f'{input_path}Findx_N_indices_neighbours_hbonds.pkl'

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# input_path = f'/gibbs/arghavan/counting_hbonds_between_graphite_walls/hbond_inputs/{t}K/{d}/'
# output_path = f'/gibbs/arghavan/pair_energy/results_pair_energy/{t}K_{d}/'
# mean_nbr_path = f'/gibbs/arghavan/counting_hbonds_between_graphite_walls/hbond_outputs/{t}K/{d}/all_frame_mean_neighbours.pkl'

traj_path = f'{input_path}{t}K_{d}.nc'
top_path = f'{input_path}topol_edited.prmtop'

traj = md.load(traj_path, top=top_path)
tpl = traj.top
all_ox_tpl_indices = tpl.select("name O")
all_wat_atm_tpl_indices = tpl.select("water")
walls = tpl.select("resname =~ 'WALL'")

n_frames = traj.n_frames
# n_frames = 50

wall_coords = traj.xyz[0][walls]

float_d = float(d)
x_center = 2.50
x_min, x_max = np.float32(x_center - float_d/2), np.float32(x_center + float_d/2)
y_min, y_max = np.min(wall_coords[:,1]), np.max(wall_coords[:,1])
z_min, z_max = np.min(wall_coords[:,2]), np.max(wall_coords[:,2])



slice_thickness = 0.010

n_slices = int((z_max - z_min) / slice_thickness)
z_starts = np.array([round(z_min + i * slice_thickness, 3) for i in range(n_slices)])
z_ends = z_starts + slice_thickness
z_distances = [round((i + 1) * slice_thickness * 10, 1) for i in range(len(z_starts))]

z_counts = [[] for _ in range(len(z_starts))]

# Loop through frames
for frame_num in range(n_frames):
    coords = traj.xyz[frame_num][all_ox_tpl_indices]

    for i, (z_start, z_end) in enumerate(zip(z_starts, z_ends)):
        mask = (
            (coords[:, 0] >= x_min) & (coords[:, 0] < x_max) &
            (coords[:, 1] >= y_min) & (coords[:, 1] < y_max) &
            (coords[:, 2] >= z_start) & (coords[:, 2] < z_end)
        )
        z_counts[i].append(np.sum(mask))

z_data = [[float(z_dist), float(np.mean(counts))] for z_dist, counts in zip(z_distances, z_counts)]
np.savetxt(f"{output_path}counts.dat", z_data, fmt="%.1f\t%.8f")

v = float_d * 10 * slice_thickness * 10 * 17.1 #A^3
numdens_data = [[float(z_dist), float(np.mean(counts))/v] for z_dist, counts in zip(z_distances, z_counts)]
np.savetxt(f"{output_path}numdens.dat", numdens_data, fmt="%.1f\t%.5f")


x_axis = [float(z_dist) for z_dist, _ in z_data]
no_wats = [float(count) for _, count in z_data]
numdens = [float(dens) for _, dens in numdens_data]


import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.figure(figsize=(8, 6))
plt.plot(x_axis, no_wats, color='blue', linewidth=2, label=f"Plates distance = {float_d*10:.1f} Å")
# plt.plot(bin_centers, rho_i_plate, color='green', linewidth=2, label=f"{float(d)*10:.1f} Å")
plt.title("Water Distribution between the edges")
plt.xlabel('Distance from the edge in Å', fontsize=14)
plt.ylabel('Average Count', fontsize=14)
plt.legend()
plt.xlim(0, 18)
plt.ylim(bottom=0)
# plt.xticks([-4.0, -2.0, 0.0, 2.0])
plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
plt.tick_params(axis='x', which='minor', length=2)
plt.tight_layout()
plt.savefig(f"{output_path}/counts.png", dpi=300)
plt.close()


plt.figure(figsize=(8, 6))
plt.plot(x_axis, numdens, color='blue', linewidth=2, label=f"Plates distance = {float_d*10:.1f} Å")
plt.title("Number Density Distribution between the edges")
plt.xlabel('Distance from the edge in Å', fontsize=14)
plt.ylabel('Number Density N/(Å^3)', fontsize=14)
plt.legend()
plt.xlim(0, 18)
plt.ylim(bottom=0)
# plt.xticks([-4.0, -2.0, 0.0, 2.0])
plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
plt.tick_params(axis='x', which='minor', length=2)
plt.tight_layout()
plt.savefig(f"{output_path}/numDens.png", dpi=300)
plt.close()