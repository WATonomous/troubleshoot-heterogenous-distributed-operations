// https://github.com/openucx/ucx/wiki/Build-and-run-ROCM-UCX-OpenMPI#sanity-check-for-large-bar-setting
// $ /opt/rocm/bin/hipcc $(/opt/rocm/bin/hipconfig --cpp_config) -L/opt/rocm/lib/ -lamdhip64 check_large_bar.c -o check_large_bar 
#include <stdio.h>
#include "hip/hip_runtime.h"

int main(int argc, char ** argv) {
  int * buf;
  hipMalloc((void**)& buf, 100);
  printf("address buf %p \n", buf);
  printf("Buf[0] = %d\n", buf[0]);
  buf[0] = 1;
  printf("Buf[0] = %d\n", buf[0]);
  return 0;
}
