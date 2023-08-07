
set(MESHSRC ${PDA_SOURCE}/meshing_scripts)
set(DECOMPSRC ${CMAKE_CURRENT_SOURCE_DIR}/../meshing_scripts)

# include demoapps headers and Schwarz routines
include_directories(
  ${PDA_SOURCE}/tpls/eigen3
  ${PDA_SOURCE}/include
  ${PDA_SOURCE}/tests_cpp/pressio/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# ---------------------------------------------------------
# 2d problems
# ---------------------------------------------------------

add_subdirectory(eigen_2d_euler_riemann_implicit)
#add_subdirectory(eigen_2d_euler_riemann_implicit_schwarz)
add_subdirectory(eigen_2d_swe_slip_wall_implicit)
#add_subdirectory(eigen_2d_swe_slip_wall_implicit_schwarz)
