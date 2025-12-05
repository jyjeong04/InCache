__kernel void mot(__global const uint *arr) {
  uint gid = get_global_id(0);
  volatile uint val = arr[gid];
}