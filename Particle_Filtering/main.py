import numpy as np

radio_stations = {
    "Copernicus": {
        "coordinates": [
            99,
            198,
            -4
        ],
        "noise_std": 15
    },
    "Montes": {
        "coordinates": [
            320,
            545,
            5
        ],
        "noise_std": 10
    },
    "Ptolem": {
        "coordinates": [
            463,
            -181,
            -11
        ],
        "noise_std": 80
    },
    "Albat": {
        "coordinates": [
            581,
            -233,
            0
        ],
        "noise_std": 80
    },
    "Lacus": {
        "coordinates": [
            680,
            279,
            0
        ],
        "noise_std": 66
    },
    "Theophilus": {
        "coordinates": [
            1020,
            -230,
            -1
        ],
        "noise_std": 15
    },
    "Luna21": {
        "coordinates": [
            1117,
            509,
            7
        ],
        "noise_std": 40
    },
    "Amoris": {
        "coordinates": [
            1316,
            388,
            2
        ],
        "noise_std": 35
    }
}

radio_station_keys = list(radio_stations.keys())


initial_coord = np.array([float(x) for x in input().strip().split(", ")])

step_magnitude_distrib_param = float(input().strip())
cov_matrix_param = float(input().strip())
count_steps_agent = int(input().strip())

estimated_distance = {}
for _ in range(len(radio_stations)):
    station_name = input().strip()
    for __ in range(count_steps_agent):
        dist = float(input().strip())
        if station_name not in estimated_distance:
            estimated_distance[station_name] = []
        estimated_distance[station_name].append(dist)



def get_movement_vector():
    magnitude = np.random.exponential(step_magnitude_distrib_param)
    angle_vertical = np.random.uniform(-np.pi/8, 0)
    angle_from_x_ax = np.random.uniform(-np.pi/8, np.pi/4)

    delta_x = magnitude * np.cos(angle_vertical) * np.cos(angle_from_x_ax)
    delta_y = magnitude * np.cos(angle_vertical) * np.sin(angle_from_x_ax)
    delta_z = magnitude * np.sin(angle_vertical)
    return np.array([delta_x, delta_y, delta_z])

def get_cov_matrix():
    return np.identity(3) * cov_matrix_param


def assign_weight_to_particles(particles, step, estimated_distances):
    weights = np.ones(particles.shape[0])
    for station_name in radio_station_keys:
        # No data of current step movement!
        if estimated_distances[station_name][step] == "None":
            continue
        # Assign weight (probability) to each particle
        for i in range(particles.shape[0]):
            estimated_distance = estimated_distances[station_name][step]
            particle_distance = np.linalg.norm(particles[i] - radio_stations[station_name]['coordinates'])
            weights[i] *= np.exp(-np.sqrt(((particle_distance - estimated_distance)**2) / radio_stations[station_name]['noise_std']**2))
    return weights



def particle_filtering():
    particle_count = 1000
    particles = np.random.multivariate_normal(initial_coord, get_cov_matrix(), particle_count)
    for i in range(count_steps_agent):
        # move particles
        for j in range(particle_count):
            particles[j] += get_movement_vector()
        # resample based on evidence
        weights = assign_weight_to_particles(particles, i, estimated_distance)
        weights /= weights.sum()
        chosen_particle_indices = np.random.choice(np.arange(particle_count), size=particle_count, p=weights)
        particles = particles[chosen_particle_indices]
    return particles

def main():
    particles = particle_filtering()
    mean = particles.mean(axis=0)
    x = int(np.ceil(mean[0]/100)*100)
    y = int(np.ceil(mean[1]/100)*100)
    z = int(np.ceil(mean[2]/100)*100)
    print(x)
    print(y)
    print(z)
    


if __name__ == "__main__":
    main()