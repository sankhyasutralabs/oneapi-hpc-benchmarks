#include <omp.h>

#include <iostream>
#include <fstream>
#include <iomanip>      // std::setprecision
#include <sys/time.h>
#include <random>
#include <cassert>
#include <chrono>

#ifndef VARTYPE
  #define VARTYPE double
#endif

struct particle{
  VARTYPE pos[3];
  VARTYPE vel[3];
};

inline
VARTYPE distance2(VARTYPE dx, VARTYPE dy, VARTYPE dz) {
  return (dx*dx + dy*dy + dz*dz);
}

void
print_data(int t_iter, particle *particles, const std::size_t num_particles) {
  std::ofstream file("particles_at_iter_"+ std::to_string(t_iter) +".csv");

  file << "#X,Y,Z" << std::endl;
  for (std::size_t i = 0; i < num_particles; i++) {
    file << particles[i].pos[0] << ", " << particles[i].pos[1] << ", " << particles[i].pos[2] << std::endl;
  }
}

void
init(particle* particles, const std::size_t num)
{
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  // std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uint32_t seed = 777;

  std::uniform_real_distribution<> dist(-0.5, 0.5);

  #pragma omp parallel
  {
    std::size_t num_threads = omp_get_num_threads();
    std::size_t particles_per_thread = num/num_threads;
    assert(0 == (num%num_threads));
    // std::cout << " particles_per_thread " << particles_per_thread << std::endl; 
    std::size_t tid = omp_get_thread_num();

    for (std::size_t i = tid*particles_per_thread; i < (tid+1)*particles_per_thread; i++)
    {
      std::minstd_rand gen(seed + i); //Standard mersenne_twister_engine seeded with rd()
      for (std::size_t j = 0; j < 3; j++) {
       particles[i].pos[j] = dist(gen);
       particles[i].vel[j] = 0.0;
      } 
    }
  }
}

void
update_velocities(particle* particles, const std::size_t num, const VARTYPE dt)
{
  #pragma omp parallel
  {
    std::size_t num_threads = omp_get_num_threads();
    std::size_t particles_per_thread = num/num_threads;
    assert(0 == (num%num_threads));
    // std::cout << " particles_per_thread " << particles_per_thread << std::endl; 
    std::size_t tid = omp_get_thread_num();

    for (std::size_t k = tid*particles_per_thread; k < (tid+1)*particles_per_thread; k++)
    {
      // Compute all to all interactions
      VARTYPE force[3] = {0.0, 0.0, 0.0};

      for (std::size_t i = 0; i < num; i++)
      {
        VARTYPE r[3] = {0.0, 0.0, 0.0};
        VARTYPE f = 0.0;

        for (std::size_t j = 0; j < 3; j++) {
          r[j] = particles[i].pos[j]-particles[k].pos[j];
        }
        VARTYPE d2  = distance2(r[0], r[1], r[2]); // 6 Flops

        f = 1. / (std::sqrt(d2 * d2 * d2) + 0.00001);

        for (std::size_t j = 0; j < 3; j++) {
          force[j] += r[j]*f; // 2 Flops per axis
        }
      }

      for (std::size_t j = 0; j < 3; j++) {
        particles[k].vel[j] += force[j]*dt;
      }
    }
  }
}

void
update_positions(particle* particles, const std::size_t num, const VARTYPE dt)
{
  #pragma omp parallel
  {
    std::size_t num_threads = omp_get_num_threads();
    std::size_t particles_per_thread = num/num_threads;
    assert(0 == (num%num_threads));
    std::size_t tid = omp_get_thread_num();

    for (std::size_t k = tid*particles_per_thread; k < (tid+1)*particles_per_thread; k++)
    {
      for (std::size_t j = 0; j < 3; j++) {
        particles[k].pos[j] += dt*particles[k].vel[j]; // 2Flops per axis, 2 memops per axis
      }
    }
  }
}

int 
main(int argc, const char** argv)
{
  std::size_t num = std::atoi(argv[1]);
  std::size_t nt = std::atoi(argv[2]);
  VARTYPE dt = 0.001;

  const size_t alloc_bytes = (num) * sizeof(particle);
  particle* particles = (particle*)malloc(alloc_bytes);

  init(particles, num);

  double total_time = 0.0;
  auto start_time = std::chrono::high_resolution_clock::now();
  for (std::size_t t = 0; t < nt; t++) {
    update_velocities(particles, num, dt);
    update_positions(particles, num, dt);
  }
  total_time += std::chrono::duration<double, std::nano>(
                 std::chrono::high_resolution_clock::now() - start_time).count();
  total_time *= 1E-9; //nano to seconds

  std::cout << " ++ sample position " << particles[num-1].pos[0] << std::endl;
  std::cout << " ++ Total Number of particles " << num << std::endl;  
  std::cout << " ++ Time step " << dt << std::endl;  
  std::cout << " ++ Time iterations " << nt << std::endl;  
  std::cout << " ++ summary: " << std::endl;  
  std::cout << std::setprecision(9) << " ++ Elapsed time for nt iters " << total_time << " seconds" << std::endl;
  std::cout << std::setprecision(9) << " ++ Elapsed time per iter " << total_time/nt << " seconds" << std::endl;

  // print_data(0, particles, num);
  return 0;
}
