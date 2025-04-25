import numpy as np
import os
import sys
import pyroomacoustics as pra
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

sys.path.append("/scratch/yw5759/HARP")
sys.path.append("/scratch/yw5759/spatialscaper/SpatialScaper")

from Generate import (
    generate_hoa_array,
    SphericalHarmonicDirectivity,
    get_material,
    get_random_dimensions,
)

from spatialscaper.sofa_utils import create_srir_sofa

OUTPUT_DIR = "/scratch/yw5759/spatialscaper/spatialscaper_RIRs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROOM_NAMES = [
    "arni_foa", "bomb_shelter_foa", "gym_foa",
    "motus_foa", "pb132_foa", "pc226_foa", "sa203_foa",
    "sc203_foa", "se203_foa", "tb103_foa", "tc352_foa"
]

ROOM_DIMENSIONS = {
    "gym_foa": [50, 50, 12],
    "pc226_foa": [26, 16, 9],
    "sa203_foa": [9, 7, 4],
    "sc203_foa": [5, 4, 3],
    "tb103_foa": [20, 15, 6],
    "pb132_foa": [9, 7, 4],
    "se203_foa": [15, 10, 4],
    "tc352_foa": [20, 15, 6],
    "arni_foa": [5, 4, 3],
    "bomb_shelter_foa": [20, 15, 6],
    "motus_foa": [9, 7, 4],
}

ORDER = 1
SAMPLE_RATE = 24000
DURATION = 0.3
MAX_ORDER = 20
NUM_SOURCES = 6480
MAX_SAMPLES = int(DURATION * SAMPLE_RATE)
NUM_CHANNELS = (ORDER + 1) ** 2

# Generate FOA mic array
mic_array_template, orientations, degrees = generate_hoa_array(
    num_microphones=NUM_CHANNELS,
    radius=0.0001,
    ambisonic_order=ORDER,
)
mic_array_template = mic_array_template.T * 0.001
mic_directivities = [
    SphericalHarmonicDirectivity(orientations[i], n=degrees[i][0], m=degrees[i][1])
    for i in range(len(degrees))
]

# Source position generation with cylindrical sampling
def generate_structured_sources(room_dims, mic_center, n_ring=360):
    """Generates cylindrical shell pattern of source positions"""
    dims = np.array(room_dims)
    center = mic_center

    max_rad = (min(dims[0], dims[1]) / 2) * 0.9
    radii = [max_rad, max_rad / 2]
    heights = np.linspace(-dims[2] / 2.5, dims[2] / 2.5, 9)

    source_positions = []
    for rad in radii:
        for h in heights:
            for i in range(n_ring):
                theta = i * (2 * np.pi) / n_ring
                x = center[0] + rad * np.cos(theta)
                y = center[1] + rad * np.sin(theta)
                z = center[2] + h
                pos = [x, y, z]

                # Ensure inside room bounds
                if all(0.0 <= pos[d] <= dims[d] for d in range(3)):
                    source_positions.append(pos)

    return np.array(source_positions)

# Room RIR generation
def generate_room(room_name):
    print(f"Generating: {room_name}")
    room_dims = ROOM_DIMENSIONS[room_name]
    materials = get_material()

    # Place mic at room center in X/Y and fixed height 1.2m
    mic_center = np.array([room_dims[0] / 2, room_dims[1] / 2, 1.2])
    mic_array = mic_array_template + mic_center.reshape(3, 1)

    rir_list = []
    rir_src_pos = []
    rir_mic_pos = []

    # Generate source positions
    source_positions = generate_structured_sources(room_dims, mic_center)

    for src_pos in tqdm(source_positions, desc=f"{room_name} Sources", leave=False):
        room = pra.ShoeBox(
            room_dims,
            fs=SAMPLE_RATE,
            materials=pra.make_materials(**materials),
            max_order=MAX_ORDER,
        )

        room.add_microphone_array(
            pra.MicrophoneArray(mic_array, SAMPLE_RATE, directivity=mic_directivities)
        )

        room.add_source(src_pos.tolist())
        room.image_source_model()
        room.compute_rir()

        rir_stack = np.stack([
            np.pad(r[0], (0, max(0, MAX_SAMPLES - len(r[0]))))[:MAX_SAMPLES]
            for r in room.rir
        ])
        rir_list.append(rir_stack)
        rir_src_pos.append(src_pos)
        rir_mic_pos.append(mic_center)

    # Stack and normalize
    rirs = np.stack(rir_list)
    rirs = rirs / np.max(np.abs(rirs))
    rirs = rirs.astype(np.float32)

    rir_src_pos = np.array(rir_src_pos)
    rir_mic_pos = np.array(rir_mic_pos)

    # Adjust positions for SOFA file
    # Translate to listener origin at [0, 0, 1.2]
    translation_vector = mic_center
    sofa_mic_pos = np.array([[0.0, 0.0, 1.2]])
    sofa_src_pos = rir_src_pos - translation_vector[np.newaxis, :]

    # Save to SOFA
    sofa_path = os.path.join(OUTPUT_DIR, f"{room_name}.sofa")
    create_srir_sofa(
        filepath=sofa_path,
        rirs=rirs,
        source_pos=sofa_src_pos,
        mic_pos=sofa_mic_pos,
        db_name=f"HARP_{room_name}",
        room_name=room_name,
        listener_name="foa",
        sr=SAMPLE_RATE
    )

    print(f"Finished: {room_name} | RIR shape: {rirs.shape}")

# Multiprocessing Launcher
if __name__ == '__main__':
    num_workers = min(cpu_count(), len(ROOM_NAMES))
    print(f"Using {num_workers} parallel workers")

    with Pool(processes=num_workers) as pool:
        pool.map(generate_room, ROOM_NAMES)

    print("All rooms done.")
