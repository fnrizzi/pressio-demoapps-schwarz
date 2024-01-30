
option(SCHWARZ_ENABLE_OMP "" ON)
option(SCHWARZ_ENABLE_THREADPOOL "" ON)

if(SCHWARZ_ENABLE_OMP)
  find_package(OpenMP)
endif()

add_subdirectory(eigen_2d_swe_slip_wall_implicit_large)
add_subdirectory(eigen_2d_swe_slip_wall_implicit_schwarz_parallel)
add_subdirectory(eigen_2d_swe_slip_wall_implicit_hproms_schwarz_parallel)
