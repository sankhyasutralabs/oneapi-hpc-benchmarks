#include <CL/sycl.hpp>
#include <dpc_common.hpp>
#include <utility>
#include <cstdlib>
#include <iostream>
#include <chrono>

#ifndef VARTYPE
  #define VARTYPE double
#endif

int
main(int argc, char* argv[])
{
  sycl::default_selector d_selector;
  sycl::device d = sycl::device(d_selector);

  sycl::property_list properties{ sycl::property::queue::in_order() };
  sycl::queue q(d, dpc_common::exception_handler, properties);

  const size_t ARRAY_SIZE = 9E9;
  const size_t alloc_bytes = ARRAY_SIZE * sizeof(VARTYPE);
  VARTYPE* T = (VARTYPE*)sycl::malloc_device(alloc_bytes, q);
  // VARTYPE* T = (VARTYPE*)sycl::malloc_shared(alloc_bytes, q);

  sycl::range<1> threads = ARRAY_SIZE;
  sycl::range<1> wg = 6000;

  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(threads, wg),
                       [=](sycl::nd_item<1> it) {
    const size_t bidx = it.get_group(0);
    VARTYPE* b = &T[6000 * bidx];

    const int i = it.get_local_id(0);

      b[i] = 10.0;
    });
  });

  q.wait();

  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(threads, wg),
                       [=](sycl::nd_item<1> it) {
      const size_t bidx = it.get_group(0);
      VARTYPE* b = &T[6000 * bidx];
      const int i = it.get_local_id(0);

      b[i] *= 3.0;
    });
  });

  q.wait();
  VARTYPE sample_val;
  q.memcpy(&sample_val, &T[ARRAY_SIZE-1], sizeof(VARTYPE));

  std::cout << " ARRAY_SIZE = " << ARRAY_SIZE << std::endl;
  std::cout << " sample_val = " << sample_val << std::endl;
  return 0;
}
