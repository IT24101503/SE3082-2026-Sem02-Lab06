#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define PARTICLES 256
#define STEPS 10
#define BOX_LENGTH 12.0
#define CUTOFF_SQUARED 9.0
#define TIME_STEP 0.001

typedef struct {
    double x;
    double y;
    double z;
} Vector;

static void apply_periodic(Vector *position) {
    position->x -= BOX_LENGTH * floor(position->x / BOX_LENGTH);
    position->y -= BOX_LENGTH * floor(position->y / BOX_LENGTH);
    position->z -= BOX_LENGTH * floor(position->z / BOX_LENGTH);
}

static void forces(const Vector *position, Vector *force,
                  double *potential, double *virial) {
    double potential_sum = 0.0;
    double virial_sum = 0.0;
#pragma omp parallel for schedule(static) reduction(+:potential_sum, virial_sum)
    for (int first = 0; first < PARTICLES; ++first) {
        for (int second = first + 1; second < PARTICLES; ++second) {
            double dx = position[first].x - position[second].x;
            double dy = position[first].y - position[second].y;
            double dz = position[first].z - position[second].z;
            dx -= BOX_LENGTH * nearbyint(dx / BOX_LENGTH);
            dy -= BOX_LENGTH * nearbyint(dy / BOX_LENGTH);
            dz -= BOX_LENGTH * nearbyint(dz / BOX_LENGTH);
            double distance_squared = dx * dx + dy * dy + dz * dz;
            if (distance_squared >= CUTOFF_SQUARED || distance_squared < 1e-12) {
                continue;
            }

            double inverse_squared = 1.0 / distance_squared;
            double inverse_sixth = inverse_squared * inverse_squared * inverse_squared;
            double force_factor = 24.0 * inverse_squared * inverse_sixth *
                                  (2.0 * inverse_sixth - 1.0);
            Vector pair_force = {
                force_factor * dx,
                force_factor * dy,
                force_factor * dz
            };
#pragma omp critical(force_update)
            {
                force[first].x += pair_force.x;
                force[first].y += pair_force.y;
                force[first].z += pair_force.z;
                force[second].x -= pair_force.x;
                force[second].y -= pair_force.y;
                force[second].z -= pair_force.z;
            }
            potential_sum += 4.0 * inverse_sixth * (inverse_sixth - 1.0);
            virial_sum += force_factor * distance_squared;
        }
    }
    *potential = potential_sum;
    *virial = virial_sum;
}

int main(void) {
    Vector position[PARTICLES];
    Vector velocity[PARTICLES] = {0};
    Vector force[PARTICLES] = {0};
    const int lattice_x = 8;
    const int lattice_y = 8;
    const double spacing = 1.5;
    for (int index = 0; index < PARTICLES; ++index) {
        int x = index % lattice_x;
        int y = (index / lattice_x) % lattice_y;
        int z = index / (lattice_x * lattice_y);
        position[index] = (Vector){
            (x + 0.5 * (y % 2)) * spacing,
            (y + 0.5 * (z % 2)) * spacing,
            z * spacing
        };
    }

    double start = omp_get_wtime();
    double potential = 0.0;
    double virial = 0.0;
    for (int step = 0; step < STEPS; ++step) {
        for (int index = 0; index < PARTICLES; ++index) {
            velocity[index].x += 0.5 * TIME_STEP * force[index].x;
            velocity[index].y += 0.5 * TIME_STEP * force[index].y;
            velocity[index].z += 0.5 * TIME_STEP * force[index].z;
            position[index].x += TIME_STEP * velocity[index].x;
            position[index].y += TIME_STEP * velocity[index].y;
            position[index].z += TIME_STEP * velocity[index].z;
            apply_periodic(&position[index]);
            force[index] = (Vector){0.0, 0.0, 0.0};
        }
        forces(position, force, &potential, &virial);
        for (int index = 0; index < PARTICLES; ++index) {
            velocity[index].x += 0.5 * TIME_STEP * force[index].x;
            velocity[index].y += 0.5 * TIME_STEP * force[index].y;
            velocity[index].z += 0.5 * TIME_STEP * force[index].z;
        }
    }
    double kinetic = 0.0;
    for (int index = 0; index < PARTICLES; ++index) {
        kinetic += velocity[index].x * velocity[index].x +
                   velocity[index].y * velocity[index].y +
                   velocity[index].z * velocity[index].z;
    }
    printf("particles = %d, steps = %d, potential = %.6f, kinetic = %.6f\n",
           PARTICLES, STEPS, potential, 0.5 * kinetic);
    printf("threads = %d, time = %.6f seconds\n", omp_get_max_threads(),
           omp_get_wtime() - start);
    return EXIT_SUCCESS;
}
