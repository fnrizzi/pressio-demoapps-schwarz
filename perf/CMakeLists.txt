
include(FindUnixCommands)

#
# first set executables
#
# include demoapps headers and Schwarz routines
include_directories(
  ${PDA_SOURCE}/tpls/eigen3
  ${PDA_SOURCE}/include
  ${PRESSIO_SOURCE}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# find_package(OpenMP)
# link_libraries(OpenMP::OpenMP_CXX)
#link_libraries(stdc++fs pthread)

# case a: we just test performance of calling demoapps repeteadly
set(exename perfexe_a)
add_executable(${exename} ${CMAKE_CURRENT_SOURCE_DIR}/main.cc)
target_compile_definitions(${exename} PRIVATE PRESSIO_ENABLE_TPL_EIGEN SCHWARZ_PERF_A)

# case b: this runs the actual real thing, but use max nonliniter = 7
set(exename perfexe_b)
add_executable(${exename} ${CMAKE_CURRENT_SOURCE_DIR}/main.cc)
target_compile_definitions(${exename} PRIVATE PRESSIO_ENABLE_TPL_EIGEN SCHWARZ_PERF_B)

# case c: similar to b except that
# - the linear solver has fixed 20 iterations
set(exename perfexe_c)
add_executable(${exename} ${CMAKE_CURRENT_SOURCE_DIR}/main.cc)
target_compile_definitions(${exename} PRIVATE PRESSIO_ENABLE_TPL_EIGEN SCHWARZ_PERF_C)

# case d: similar to c except that
# - the linear solver has fixed 1 iterations
set(exename perfexe_d)
add_executable(${exename} ${CMAKE_CURRENT_SOURCE_DIR}/main.cc)
target_compile_definitions(${exename} PRIVATE PRESSIO_ENABLE_TPL_EIGEN SCHWARZ_PERF_D)

# case e:
# - we replace the linear solver with a trivial thing that sets solution in 1 iter
set(exename perfexe_e)
add_executable(${exename} ${CMAKE_CURRENT_SOURCE_DIR}/main.cc)
target_compile_definitions(${exename} PRIVATE PRESSIO_ENABLE_TPL_EIGEN SCHWARZ_PERF_E)

# case f:
# - we replace the nonlinear solver with a trivial thing
set(exename perfexe_f)
add_executable(${exename} ${CMAKE_CURRENT_SOURCE_DIR}/main.cc)
target_compile_definitions(${exename} PRIVATE PRESSIO_ENABLE_TPL_EIGEN SCHWARZ_PERF_F)


#
# deal with all the various mesh cases to create
#
set(MESHSRC ${PDA_SOURCE}/meshing_scripts)
set(DECOMPSRC ${CMAKE_CURRENT_SOURCE_DIR}/../meshing_scripts)
set(MESHDRIVER ${MESHSRC}/create_full_mesh.py)
set(DECOMPDRIVER ${DECOMPSRC}/create_decomp_meshes.py)

# mesh case A
set(N 200)
set(ns 2)
set(OUTDIR_A ${CMAKE_CURRENT_BINARY_DIR}/mesh_a/mesh)
file(MAKE_DIRECTORY ${OUTDIR_A})
set(CMD "python3 ${DECOMPDRIVER} --meshScript ${MESHDRIVER} -n ${N} ${N} --outDir ${OUTDIR_A} -s 5 --bounds -5.0 5.0 -5.0 5.0 --numDoms ${ns} ${ns} --overlap 10")
message(NOTICE ${CMD})
execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
if(RES)
  message(FATAL_ERROR "Mesh generation failed")
else()
  message("Mesh generation succeeded!")
endif()


# #(${exename} PUBLIC -DUSE_WENO3)
# add_test(NAME ${testname}
# COMMAND ${CMAKE_COMMAND}
# -DMESHDRIVER=${MESHSRC}/create_full_mesh.py
# -DDECOMPDRIVER=${DECOMPSRC}/create_decomp_meshes.py
# -DOUTDIR=${CMAKE_CURRENT_BINARY_DIR}
# -DEXENAME=$<TARGET_FILE:${exename}>
# -DSTENCILVAL=5
# -P ${CMAKE_CURRENT_SOURCE_DIR}/../test.cmake
# )
