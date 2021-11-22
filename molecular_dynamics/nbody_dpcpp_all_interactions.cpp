#include <iostream>
#include <iomanip>      // std::setprecision
#include <sys/time.h>
#include <random>
#include <chrono>

#include <CL/sycl.hpp>
#include <dpc_common.hpp>
#include <oneapi/dpl/random>

#ifndef VARTYPE
  #define VARTYPE double
#endif

struct particle{
  VARTYPE pos[3];
  VARTYPE vel[3];
};

VARTYPE distance2(VARTYPE dx, VARTYPE dy, VARTYPE dz) {
  return (dx*dx + dy*dy + dz*dz);
}

inline
uint32_t get_random_num(uint32_t x) 
{
  return x*1664525 + 1013904223;
}

void
print_data(int t_iter, particle *particles, const std::size_t num_particles) {
  std::ofstream file("particles_at_iter_"+ std::to_string(t_iter) +".csv");

  file << "#X,Y,Z" << std::endl;
  for (std::size_t i = 0; i < num_particles; i++) {
    file << particles[i].pos[0] << ", " << particles[i].pos[1] << ", " << particles[i].pos[2] << std::endl;
  }
}

float Q_rsqrt( float number )
{
  long i;
  float x2, y;
  const float threehalfs = 1.5F;

  x2 = number * 0.5F;
  y  = number;
  i  = * ( long * ) &y;                       // evil floating point bit level hacking
  i  = 0x5f3759df - ( i >> 1 ); 
  y  = * ( float * ) &i;
  y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
//  y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

  return y;
}

std::pair<double, VARTYPE>
run(const size_t num_particles, const VARTYPE dt, const size_t time_iters, sycl::device& d) {

  sycl::property_list properties{ sycl::property::queue::in_order() };
  sycl::queue q(d, dpc_common::exception_handler, properties);

  // prepare random particle positions
  const size_t alloc_bytes = (num_particles) * sizeof(particle);

  // particle *particles_host = (particle*)sycl::malloc_host(alloc_bytes, q);

  // uint32_t x = 123;

  // for(size_t i = 0; i < num_particles; i++) {
  //   for (std::size_t j = 0; j < 3; j++) {
  //     x = get_random_num(x);
  //     const auto rval = (x/4294967295.0) - 0.5;
  //     particles_host[i].pos[j] = rval;
  //     particles_host[i].vel[j] = 0.0;
  //   }
  // }

  // std::random_device rd;  //Will be used to obtain a seed for the random number engine
  // std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  // std::uniform_real_distribution<> dist(-0.5, 0.5);

  // for(size_t i = 0; i < num_particles; i++) {
  //   for (std::size_t j = 0; j < 3; j++) {
  //     particles_host[i].pos[j] = dist(gen);
  //     particles_host[i].vel[j] = 0.0;
  //   }
  // }

  // q.memcpy(particles, particles_host, alloc_bytes);
  // q.wait();

  particle *particles = (particle*)sycl::malloc_device(alloc_bytes, q);

  sycl::range<1> threads = {num_particles};
  //sycl::range<1> wg = {64};
  sycl::range<1> wg = {8000};

  // Init
  auto initialise = [&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(threads, wg), [=](sycl::nd_item<1> it) {
      const size_t particle_id = it.get_global_id();

      std::uint32_t seed = 777;
      // std::uint64_t offset = 0;
      std::uint64_t offset = particle_id;

      // Create minstd_rand engine
      oneapi::dpl::minstd_rand engine(seed, offset);

      // Create uniform_real_distribution distribution
      oneapi::dpl::uniform_real_distribution<VARTYPE> distr(-0.5,0.5);

      for (std::size_t j = 0; j < 3; j++) {
        particles[particle_id].pos[j] = distr(engine);
        particles[particle_id].vel[j] = 0.0;
      }
    });
  };

  auto update_velocity = [&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(threads, wg), [=](sycl::nd_item<1> it) {
      const size_t particle_id = it.get_global_id();

      VARTYPE force[3] = {0.0, 0.0, 0.0};

      for(std::size_t i = 0; i < num_particles; i++) { // w.r.t to all particles
        VARTYPE r[3] = {0.0, 0.0, 0.0};
        VARTYPE f = 0.0;

        for (std::size_t j = 0; j < 3; j++) {
          r[j] = particles[i].pos[j]-particles[particle_id].pos[j]; // 2 Flops per axis , 1 memops per axis + 3*N 
        }
        auto d2  = distance2(r[0], r[1], r[2]); // 6 Flops

        // f  = f*f*f;
        // f  = f*f*f;
        // f  = 1./f;
        // f = Q_rsqrt(d2*d2*d2); // 3 Flops + 7 Flops (Q_sqrt)
        f = 1./ (std::sqrt(d2*d2*d2) + 0.00001);
        for (std::size_t j = 0; j < 3; j++) {
          force[j] += r[j]*f; // 2 Flops per axis
        }
      }

      for (std::size_t j = 0; j < 3; j++) {
        particles[particle_id].vel[j] += force[j]*dt; // 2 Flops per axis, 1 memops per axis
      }
    });
  };

  auto update_position = [&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(threads, wg), [=](sycl::nd_item<1> it) {
      const size_t particle_id = it.get_global_id();

      for (std::size_t j = 0; j < 3; j++) {
        particles[particle_id].pos[j] += dt*particles[particle_id].vel[j]; // 2Flops per axis, 2 memops per axis
      }
    });
  };

  double total_time = 0.0;
  // auto start_time = std::chrono::high_resolution_clock::now();
  try {

    // over T iterations
    for (std::size_t t = 0; t < time_iters; t++) {

      q.submit(initialise);
      q.wait();

      auto tic1 = std::chrono::high_resolution_clock::now();
      q.submit(update_velocity);
      q.wait();

      q.submit(update_position);
      q.wait();
      auto tic2 = std::chrono::high_resolution_clock::now();
      auto elapsed_time = (std::chrono::duration<double, std::nano>(
                         tic2 - tic1).count())*1E-9;
      total_time += elapsed_time;
    }
  } catch (sycl::exception const& ex) {
    std::cerr << "dpcpp error: " << ex.what() << std::endl;
  }
  // q.wait();
  // total_time += std::chrono::duration<double, std::nano>(
  //                std::chrono::high_resolution_clock::now() - start_time).count();
  // total_time *= 1E-9; //nano to seconds

  // check correctness
  // q.memcpy(&particles_host[0].pos[0], &particles[0].pos[0], alloc_bytes);
  // q.wait();

  // VARTYPE sample_val;
  // q.memcpy(&sample_val, &particles_host[num_particles-1].pos[0], sizeof(VARTYPE));

  // print_data(0, particles_host, num_particles);

  VARTYPE sample_val;
  q.memcpy(&sample_val, &particles[num_particles-1].pos[0], sizeof(VARTYPE));

  // sycl::free(particles_host, q);
  sycl::free(particles, q);

  return std::make_pair(total_time, sample_val);
}

int main(int argc, const char** argv)
{
  if(argc != 3) {
    std::cerr << "Expected 3 args : Usage = " << "./a.out 1024 20" << std::endl;
    exit(EXIT_FAILURE);
  }

  const size_t num_particles = atoi(argv[1]);
  const size_t time_iters    = atoi(argv[2]);
  const VARTYPE dt           = 0.001;

  sycl::default_selector d_selector;
  sycl::device d = sycl::device(d_selector);
  std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++"
            << std::endl;
  std::cout << "++ Using " << d.get_info<sycl::info::device::name>()
            << std::endl;

  auto result = run(num_particles, dt, time_iters, d);

  std::cout << " ++ Number of particles " << num_particles << std::endl;  
  std::cout << " ++ Time step " << dt << std::endl;  
  std::cout << " ++ Time iterations " << time_iters << std::endl;  
  std::cout << " ++ summary: " << std::endl;  
  std::cout << std::setprecision(9) << " ++ Elapsed time " << result.first << " seconds" << std::endl;
  std::cout << std::setprecision(9) << " ++ Elapsed time per iter " << (result.first)/time_iters << " seconds" << std::endl;
  std::cout << " ++ sample_val : " << result.second << std::endl;  
  return 0;
}
