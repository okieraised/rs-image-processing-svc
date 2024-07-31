#ifndef GPU_NMS_HPP
#define GPU_NMS_HPP

#include <cstdint>

extern "C" {
  void _nms(int32_t* keep, int* num_out, float* boxes, int boxes_num, int boxes_dim, float thresh, int device_id);
}

#endif
