import mdtraj as md
import numpy as np
import os
import sys
from warnings import filterwarnings
filterwarnings("ignore")

def load_trajectory(input_path: str, temp: int, dist: str):
    """
    Load the molecular dynamics trajectory.
    Args:
        input_path (str): Path to the directory containing trajectory and topology files.
        temp (int): Temperature in Kelvin.
        dist (str): Distance between plates (in nanometers).

    Returns:
        md.Trajectory: trajectory object
    """

    traj_path = f'{input_path}{temp}K_{dist}.nc'
    top_path = f'{input_path}topol_edited.prmtop'
    traj = md.load(traj_path, top=top_path)
    return traj

def get_plate_boundaries(traj, plate_atom_indices, plate_distance):
    """
    Get the boundaries of the plates from the first frame. plates are fixed. so the first frame is sufficient.
    plates are centered at x=2.50 nm, and the distance between them is given by plate_distance.

    Args:
        traj (md.Trajectory): Molecular dynamics trajectory object.
        plate_atoms (list or np.ndarray): Indices of atoms that make up the plate surfaces.
        plate_distance (float): Distance between the plates in nanometers
    Returns:
        tuple: (x_min, x_max, y_min, y_max, z_min, z_max) boundaries of the region between plates.
    """

    wall_coords = traj.xyz[0][plate_atom_indices]
    x_center = 2.50
    x_min = np.float32(x_center - plate_distance / 2)
    x_max = np.float32(x_center + plate_distance / 2)
    y_min = np.min(wall_coords[:, 1])
    y_max = np.max(wall_coords[:, 1])
    z_min = np.min(wall_coords[:, 2])
    z_max = np.max(wall_coords[:, 2])
    return x_min, x_max, y_min, y_max, z_min, z_max

def count_waters_per_frame(traj, ox_indices, bounds):
    """
    Count the number of water molecules between the plates for each frame.

    Args:
        traj (md.Trajectory): Molecular dynamics trajectory object.
        ox_indices (list or np.ndarray): Indices of water oxygen atoms.
        bounds (tuple): (x_min, x_max, y_min, y_max, z_min, z_max) boundaries of the region between plates.

    Returns:
        list: Number of water molecules between the plates for each frame.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    num_waters_per_frame = []
    for frame_num in range(traj.n_frames):
        coords = traj.xyz[frame_num][ox_indices]
        in_box = (
            (coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) &
            (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max) &
            (coords[:, 2] >= z_min) & (coords[:, 2] <= z_max)
        )
        num_waters_per_frame.append(np.count_nonzero(in_box))
    return num_waters_per_frame

def save_dx_file(output_path, no_of_waters_between_plates):
    """
    Save the water count data to a .dx file.
    Args:
        output_path (str): Path to the output directory.
        no_of_waters_between_plates (list): Number of water molecules between the plates for each frame.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    average_count = np.mean(no_of_waters_between_plates)

    with open(f"{output_path}num_waters_between_plates.dx", "w") as f:
        f.write(f"#Average number of water molecules between plates: {average_count:.2f} \n")
        f.write("# Frame_index  Number_of_waters\n")
        for i, val in enumerate(no_of_waters_between_plates):
            f.write(f"{i} {val}\n")

def main():

    """
    Main function to analyze the number of water molecules confined between hydrophobic plates.
    """
    
    temp = int(sys.argv[1])
    dist = sys.argv[2]
    if len(sys.argv) != 3:
        print("Usage: python confined_water_counter.py <temperature> <plate_distance>")
        sys.exit(1)

    plate_distance = float(dist)

    input_path = f'/Users/arghavan/Graduate Center Dropbox/Arghavan Vedadi Gargari/MyFiles/{temp}K/{dist}/'
    output_path = f'/Users/arghavan/lab/hydrophobic_plates/results/{temp}K/{dist}/'


    traj = load_trajectory(input_path, temp, dist)
    tpl = traj.topology
    all_ox_indices = tpl.select("name O")
    plate_indices = tpl.select("resname =~ 'WALL'")

    bounds = get_plate_boundaries(traj, plate_indices, plate_distance)
    no_of_waters_between_plates = count_waters_per_frame(traj, all_ox_indices, bounds)

    save_dx_file(output_path, no_of_waters_between_plates)


if __name__ == "__main__":
    main()