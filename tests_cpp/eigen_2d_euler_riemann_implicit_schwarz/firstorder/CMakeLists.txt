
set(testname eigen_2d_riemann_firstorder_implicit_schwarz)
set(exename  ${testname}_exe)

configure_file(../plot.py plot.py COPYONLY)
configure_file(../compare.py compare.py COPYONLY)
foreach(DOM RANGE 3)
  configure_file(p_gold_${DOM}.txt p_gold_${DOM}.txt COPYONLY)
  configure_file(rho_gold_${DOM}.txt rho_gold_${DOM}.txt COPYONLY)
endforeach()

add_executable(${exename} ${CMAKE_CURRENT_SOURCE_DIR}/../main.cc)

add_test(NAME ${testname}
COMMAND ${CMAKE_COMMAND}
-DMESHDRIVER=${MESHSRC}/create_full_mesh.py
-DDECOMPDRIVER=${DECOMPSRC}/create_decomp_meshes.py
-DOUTDIR=${CMAKE_CURRENT_BINARY_DIR}
-DEXENAME=$<TARGET_FILE:${exename}>
-DSTENCILVAL=3
-P ${CMAKE_CURRENT_SOURCE_DIR}/../test.cmake
)

