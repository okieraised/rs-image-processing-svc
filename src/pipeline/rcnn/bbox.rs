use ndarray::{Array2};


pub(crate) fn bbox_overlaps(boxes: &Array2<f32>, query_boxes: &Array2<f32>) -> Array2<f32> {
    let n = boxes.nrows();
    let k = query_boxes.nrows();
    let mut overlaps = Array2::<f32>::zeros((n, k));

    for k in 0..k {
        let box_area = (query_boxes[[k, 2]] - query_boxes[[k, 0]] + 1.0) *
            (query_boxes[[k, 3]] - query_boxes[[k, 1]] + 1.0);

        for n in 0..n {
            let iw = f32::min(boxes[[n, 2]], query_boxes[[k, 2]]) -
                f32::max(boxes[[n, 0]], query_boxes[[k, 0]]) + 1.0;
            if iw > 0.0 {
                let ih = f32::min(boxes[[n, 3]], query_boxes[[k, 3]]) -
                    f32::max(boxes[[n, 1]], query_boxes[[k, 1]]) + 1.0;
                if ih > 0.0 {
                    let ua = (boxes[[n, 2]] - boxes[[n, 0]] + 1.0) *
                        (boxes[[n, 3]] - boxes[[n, 1]] + 1.0) +
                        box_area - iw * ih;
                    overlaps[[n, k]] = iw * ih / ua;
                }
            }
        }
    }

    overlaps
}


#[cfg(test)]
mod tests {
    use ndarray::array;
    use crate::rcnn::bbox::bbox_overlaps;

    #[test]
    fn test_bbox_overlaps() {
        let boxes = array![
            [10.0, 20.0, 50.0, 60.0],
            [15.0, 25.0, 55.0, 65.0]
        ];

        let query_boxes = array![
            [12.0, 22.0, 52.0, 62.0],
            [18.0, 28.0, 58.0, 68.0]
        ];

        let overlaps = bbox_overlaps(&boxes, &query_boxes);
        println!("Overlaps:\n{:?}", overlaps);
    }
}