use ndarray::{Array1, ArrayView2, s};
use std::cmp::Ordering;


/// Extracts x1, y1, x2, y2, and scores from the detections array.
/// Calculate Areas: Calculates the area for each bounding box.
/// Sort Scores: Orders the boxes by their score in descending order.
/// Suppress Overlapping Boxes: Iterates through each box,
/// checks for overlap, and suppresses boxes that have an IoU greater than the threshold.
fn cpu_nms(dets: ArrayView2<f32>, thresh: f32) -> Vec<usize> {
    let x1 = dets.slice(s![.., 0]);
    let y1 = dets.slice(s![.., 1]);
    let x2 = dets.slice(s![.., 2]);
    let y2 = dets.slice(s![.., 3]);
    let scores = dets.slice(s![.., 4]);

    let areas: Array1<f32> = (&x2 - &x1 + 1.0) * (&y2 - &y1 + 1.0);
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_unstable_by(|&i, &j| scores[j].partial_cmp(&scores[i]).unwrap_or(Ordering::Equal));

    let ndets = dets.shape()[0];
    let mut suppressed = vec![0; ndets];

    let mut keep = Vec::new();

    for &_i in order.iter() {
        if suppressed[_i] == 1 {
            continue;
        }
        keep.push(_i);
        let ix1 = x1[_i];
        let iy1 = y1[_i];
        let ix2 = x2[_i];
        let iy2 = y2[_i];
        let iarea = areas[_i];
        for &_j in order.iter() {
            if suppressed[_j] == 1 {
                continue;
            }
            let xx1 = ix1.max(x1[_j]);
            let yy1 = iy1.max(y1[_j]);
            let xx2 = ix2.min(x2[_j]);
            let yy2 = iy2.min(y2[_j]);
            let w = (xx2 - xx1 + 1.0).max(0.0);
            let h = (yy2 - yy1 + 1.0).max(0.0);
            let inter = w * h;
            let ovr = inter / (iarea + areas[_j] - inter);
            if ovr >= thresh {
                suppressed[_j] = 1;
            }
        }
    }

    keep
}


#[cfg(test)]
mod tests {
    use ndarray::array;
    use crate::rcnn::cpu_nms::cpu_nms;

    #[test]
    fn test_cpu_nms() {
        let dets = array![
        [100.0, 100.0, 210.0, 210.0, 0.72],
        [250.0, 250.0, 420.0, 420.0, 0.8],
        [220.0, 220.0, 320.0, 330.0, 0.92],
        [100.0, 100.0, 210.0, 210.0, 0.6]
    ];

        let thresh = 0.3;
        let keep = cpu_nms(dets.view(), thresh);

        println!("Kept indices: {:?}", keep);
    }
}