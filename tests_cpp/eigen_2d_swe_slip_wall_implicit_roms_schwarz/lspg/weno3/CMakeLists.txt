
set(testname eigen_2d_swe_slip_wall_weno3_implicit_lspg_schwarz)
set(exename  ${testname}_exe)

configure_file(../plot.py plot.py COPYONLY)
configure_file(../../gen_trial_space.py gen_trial_space.py COPYONLY)
configure_file(../compare.py compare.py COPYONLY)
foreach(DOM RANGE 3)
  configure_file(h_gold_${DOM}.txt h_gold_${DOM}.txt COPYONLY)
endforeach()

add_executable(${exename} ${CMAKE_CURRENT_SOURCE_DIR}/../main.cc)
target_compile_definitions(${exename} PUBLIC -DUSE_WENO3)

add_test(NAME ${testname}
COMMAND ${CMAKE_COMMAND}
-DMESHDRIVER=${MESHSRC}/create_full_mesh.py
-DDECOMPDRIVER=${DECOMPSRC}/create_decomp_meshes.py
-DOUTDIR=${CMAKE_CURRENT_BINARY_DIR}
-DEXENAME=$<TARGET_FILE:${exename}>
-DSTENCILVAL=5
-P ${CMAKE_CURRENT_SOURCE_DIR}/../test.cmake
)

