# Commands to compile your matrix multiply design
# and package it as an AXI IP for Vivado
set src_dir "../hls/mmult_float"
open_project accel
set_top mmult_hw
add_files $src_dir/mmult_float.cpp
add_files -tb $src_dir/mmult_test.cpp
open_solution "solution0"
set_part {xc7z020clg484-1}
create_clock -period 10 -name default
set_clock_uncertainty 12.5%

config_compile -pipeline_loops 0
config_export -vivado_phys_opt place
config_export -vivado_optimization_level 2
csim_design -clean
csynth_design
export_design -evaluate verilog -format ip_catalog 
close_project
exit