configure_file(../gen_trial_space.py gen_trial_space.py COPYONLY)
configure_file(../gen_sample_mesh.py gen_sample_mesh.py COPYONLY)

list(APPEND CASES firstorder)
list(APPEND SSIZES 3)
if(${TESTWENO3})
  list(APPEND CASES weno3)
  list(APPEND SSIZES 5)
endif()

foreach(case ss IN ZIP_LISTS CASES SSIZES)

  set(TESTDIR ${CMAKE_CURRENT_BINARY_DIR}/${case})

  message(STATUS ${case})
  set(testname eigen_2d_swe_slip_wall_${case}_implicit_lspg_hyper_schwarz_parallel)

  configure_file(compare.py compare.py COPYONLY)

  set(EXTRADEF "")
  if(${case} STREQUAL "weno3")
    set(EXTRADEF USE_WENO3)
  endif()

  if(SCHWARZ_ENABLE_OMP)
    file(MAKE_DIRECTORY ${TESTDIR}_omp)
    configure_file(plot.py ${TESTDIR}_omp/plot.py COPYONLY)
    foreach(DOM RANGE 11)
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${case}/h_gold_${DOM}.txt ${TESTDIR}_omp/h_gold_${DOM}.txt COPYONLY)
    endforeach()

    set(exename ${testname}_exe_omp)
    add_executable(${exename} ${CMAKE_CURRENT_SOURCE_DIR}/main.cc)
    target_compile_definitions(${exename} PRIVATE SCHWARZ_ENABLE_OMP ${EXTRADEF})
    target_link_libraries(${exename} PRIVATE OpenMP::OpenMP_CXX pthread)
    target_compile_options(${exename} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)

    add_test(NAME ${testname}_omp
      COMMAND ${CMAKE_COMMAND}
      -DMESHDRIVER=${MESHSRC}/create_full_mesh.py
      -DDECOMPDRIVER=${DECOMPSRC}/create_decomp_meshes.py
      -DOUTDIR=${TESTDIR}_omp
      -DEXENAME=$<TARGET_FILE:${exename}>
      -DSTENCILVAL=${ss}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/test.cmake
    )
  endif()

  if(SCHWARZ_ENABLE_THREADPOOL)
    file(MAKE_DIRECTORY ${TESTDIR}_tp)
    configure_file(plot.py ${TESTDIR}_tp/plot.py COPYONLY)
    foreach(DOM RANGE 11)
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${case}/h_gold_${DOM}.txt ${TESTDIR}_tp/h_gold_${DOM}.txt COPYONLY)
    endforeach()

    set(exename ${testname}_exe_tp)
    add_executable(${exename} ${CMAKE_CURRENT_SOURCE_DIR}/main.cc)
    target_compile_definitions(${exename} PRIVATE SCHWARZ_ENABLE_THREADPOOL ${EXTRADEF})
    target_link_libraries(${exename} PRIVATE pthread)
    target_compile_options(${exename} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)

    add_test(NAME ${testname}_tp
      COMMAND ${CMAKE_COMMAND}
      -DMESHDRIVER=${MESHSRC}/create_full_mesh.py
      -DDECOMPDRIVER=${DECOMPSRC}/create_decomp_meshes.py
      -DOUTDIR=${TESTDIR}_tp
      -DEXENAME=$<TARGET_FILE:${exename}>
      -DSTENCILVAL=${ss}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/test.cmake
    )
  endif()

endforeach()
