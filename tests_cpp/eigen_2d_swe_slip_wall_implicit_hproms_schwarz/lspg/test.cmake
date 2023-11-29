include(FindUnixCommands)

set(CMD "python3 ${DECOMPDRIVER} --meshScript ${MESHDRIVER} -n 50 50 --outDir ${OUTDIR}/full_mesh -s ${STENCILVAL} --bounds -5.0 5.0 -5.0 5.0 --numDoms 2 2 --overlap 10")
execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
if(RES)
  message(FATAL_ERROR "Full mesh generation failed")
else()
  message("Full mesh generation succeeded!")
endif()

set(CMD "python3 ./gen_sample_mesh.py")
execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
if(RES)
  message(FATAL_ERROR "Global sample meshes generation failed")
else()
  message("Global sample mesh generation succeeded!")
endif()

set(CMD "python3 ${SAMPMESHDRIVER} --decompMeshDir ./full_mesh --sampleMeshDir ./sample_mesh --sampleGIDFile sample_mesh_gids.txt")
execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
if(RES)
  message(FATAL_ERROR "Local sample mesh generation failed")
else()
  message("Local sample mesh generation succeeded!")
endif()

set(CMD "python3 ./gen_trial_space.py")
execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
if(RES)
  message(FATAL_ERROR "Basis generation failed")
else()
  message("Basis generation succeeded!")
endif()

execute_process(COMMAND ${EXENAME} RESULT_VARIABLE CMD_RESULT)
if(RES)
  message(FATAL_ERROR "run failed")
else()
  message("run succeeded!")
endif()

#set(CMD "python3 compare.py")
#execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
#if(RES)
#  message(FATAL_ERROR "comparison failed")
#else()
#  message("comparison succeeded!")
#endif()
