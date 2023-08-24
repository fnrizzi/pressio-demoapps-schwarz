
configure_file(./gen_trial_space.py gen_trial_space.py COPYONLY)

add_subdirectory(galerkin/firstorder)
add_subdirectory(lspg/firstorder)
