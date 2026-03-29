open_project pf_find_index_ping_pong_prj -reset
set_top find_index_kernel

add_files pf_find_index.cpp -cflags "-I."
add_files pf_find_index.h
add_files -tb pf_find_index_tb.cpp -cflags "-I."

open_solution "solution1" -reset
set_part {xcu50-fsvh2104-2-e}
create_clock -period 3.33

csim_design
csynth_design

exit
