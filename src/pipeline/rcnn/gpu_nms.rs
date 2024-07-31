// // the gpu_nms function is calling an external function defined in a C/C++ header file.
// // The extern "C" block is used to declare the C linkage.
// // The unsafe block is used to call the unsafe function _nms.
// // The rest of the logic remains similar to the original Python code.
//
// use ndarray::{Array1, Array2, Axis, s};
// use std::ffi::c_void;
//
// extern "C" {
//     fn _nms(
//         keep: *mut i32,
//         num_out: *mut i32,
//         boxes: *const f32,
//         boxes_num: i32,
//         boxes_dim: i32,
//         thresh: f32,
//         device_id: i32,
//     );
// }
//
// fn gpu_nms(dets: Array2<f32>, thresh: f32, device_id: i32) -> Vec<usize> {
//     let boxes_num = dets.shape()[0] as i32;
//     let boxes_dim = dets.shape()[1] as i32;
//
//     let mut keep: Vec<i32> = vec![0; boxes_num as usize];
//     let mut num_out: i32 = 0;
//
//     let scores: Array1<f32> = dets.slice(s![.., 4]).to_owned();
//     let mut order: Vec<usize> = (0..scores.len()).collect();
//     order.sort_unstable_by(|&i, &j| scores[j].partial_cmp(&scores[i]).unwrap());
//
//     let sorted_dets: Array2<f32> = dets.select(Axis(0), &order);
//
//     unsafe {
//         _nms(
//             keep.as_mut_ptr(),
//             &mut num_out,
//             sorted_dets.as_ptr(),
//             boxes_num,
//             boxes_dim,
//             thresh,
//             device_id,
//         );
//     }
//
//     keep.truncate(num_out as usize);
//
//     keep.iter().map(|&i| order[i as usize]).collect()
// }
//
// #[cfg(test)]
// mod tests {
//     use ndarray::{array, Array, Array2, Array4, ArrayBase, Ix0, Ix1};
//     use crate::rcnn::gpu_nms::gpu_nms;
//
//     #[test]
//     fn test_cpu_nms() {
//         // Example usage
//         let dets: Array2::<f32> = array![
//             [100.0, 100.0, 210.0, 210.0, 0.72],
//             [250.0, 250.0, 420.0, 420.0, 0.8],
//             [220.0, 220.0, 320.0, 330.0, 0.92],
//             [100.0, 100.0, 210.0, 210.0, 0.6]
//         ];
//         // let dets = Array2::<f32>::from([100.0, 100.0, 210.0, 210.0, 0.72]).to_owned();
//
//         let thresh = 0.3;
//         let device_id = 0;
//         let keep = gpu_nms(dets, thresh, device_id);
//
//         println!("Kept indices: {:?}", keep);
//     }
// }