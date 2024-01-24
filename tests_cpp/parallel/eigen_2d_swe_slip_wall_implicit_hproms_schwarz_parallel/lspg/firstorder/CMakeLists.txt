configure_file(../plot.py plot.py COPYONLY)
configure_file(../../gen_trial_space.py gen_trial_space.py COPYONLY)
configure_file(../../gen_sample_mesh.py gen_sample_mesh.py COPYONLY)
configure_file(../compare.py compare.py COPYONLY)
foreach(DOM RANGE 11)
  configure_file(h_gold_${DOM}.txt h_gold_${DOM}.txt COPYONLY)
endforeach()

set(basetestname eigen_2d_swe_slip_wall_firstorder_implicit_lspg_hyper_schwarz_parallel)

if(SCHWARZ_ENABLE_THREADPOOL)
  set(testname ${basetestname}_tp)
  set(exename  ${testname}_exe_tp)

  add_executable(${exename} ${CMAKE_CURRENT_SOURCE_DIR}/../main.cc)
  target_compile_definitions(${exename} PRIVATE SCHWARZ_ENABLE_THREADPOOL)
  target_link_libraries(${exename} PRIVATE pthread)
  target_compile_options(${exename} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)

  add_test(NAME ${testname}
  COMMAND ${CMAKE_COMMAND}
  -DMESHDRIVER=${MESHSRC}/create_full_mesh.py
  -DDECOMPDRIVER=${DECOMPSRC}/create_decomp_meshes.py
  -DOUTDIR=${CMAKE_CURRENT_BINARY_DIR}
  -DEXENAME=$<TARGET_FILE:${exename}>
  -DSTENCILVAL=3
  -P ${CMAKE_CURRENT_SOURCE_DIR}/../test.cmake
  )
endif()

if(SCHWARZ_ENABLE_OMP)
  set(testname ${basetestname}_omp)
  set(exename  ${testname}_exe_omp)

  add_executable(${exename} ${CMAKE_CURRENT_SOURCE_DIR}/../main.cc)
  target_compile_definitions(${exename} PRIVATE SCHWARZ_ENABLE_OMP)
  target_link_libraries(${exename} PRIVATE OpenMP::OpenMP_CXX pthread)
  target_compile_options(${exename} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)

  add_test(NAME ${testname}
  COMMAND ${CMAKE_COMMAND}
  -DMESHDRIVER=${MESHSRC}/create_full_mesh.py
  -DDECOMPDRIVER=${DECOMPSRC}/create_decomp_meshes.py
  -DOUTDIR=${CMAKE_CURRENT_BINARY_DIR}
  -DEXENAME=$<TARGET_FILE:${exename}>
  -DSTENCILVAL=3
  -P ${CMAKE_CURRENT_SOURCE_DIR}/../test.cmake
  )
endif()

# add_executable(${exename} ${CMAKE_CURRENT_SOURCE_DIR}/../main.cc)
