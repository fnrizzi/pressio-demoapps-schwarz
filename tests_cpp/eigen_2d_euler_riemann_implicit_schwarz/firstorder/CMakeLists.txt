
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
-DOUTDIR=${CMAKE_CURRENT_BINARY_DIR}/mesh
-DEXENAME=$<TARGET_FILE:${exename}>
-DSTENCILVAL=3
-P ${CMAKE_CURRENT_SOURCE_DIR}/../test.cmake
)

#
# FRIZZI
#
set(testname_fr eigen_2d_riemann_firstorder_implicit_schwarz_frizzi)
set(exename_fr ${testname}_frizzi_exe)
add_executable(${exename_fr} ${CMAKE_CURRENT_SOURCE_DIR}/../main_frizzi.cc)
add_test(NAME ${testname_fr}
COMMAND ${CMAKE_COMMAND}
-DMESHDRIVER=${MESHSRC}/create_full_mesh.py
-DDECOMPDRIVER=${DECOMPSRC}/create_decomp_meshes.py
-DOUTDIR=${CMAKE_CURRENT_BINARY_DIR}/mesh
-DEXENAME=$<TARGET_FILE:${exename_fr}>
-DSTENCILVAL=3
-P ${CMAKE_CURRENT_SOURCE_DIR}/../test.cmake
)
