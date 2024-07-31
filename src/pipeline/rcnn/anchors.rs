use ndarray::{Array2, Array4};

pub fn anchors(height: usize, width: usize, stride: usize, base_anchors: &Array2<f32>) -> Array4<f32> {
    let a = base_anchors.nrows();
    let mut all_anchors = Array4::<f32>::zeros((height, width, a, 4));

    for iw in 0..width {
        let sw = (iw * stride) as f32;
        for ih in 0..height {
            let sh = (ih * stride) as f32;
            for k in 0..a {
                all_anchors[[ih, iw, k, 0]] = base_anchors[[k, 0]] + sw;
                all_anchors[[ih, iw, k, 1]] = base_anchors[[k, 1]] + sh;
                all_anchors[[ih, iw, k, 2]] = base_anchors[[k, 2]] + sw;
                all_anchors[[ih, iw, k, 3]] = base_anchors[[k, 3]] + sh;
            }
        }
    }

    all_anchors
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use crate::rcnn::anchors::anchors;

    #[test]
    fn test_anchors() {
        let base_anchors = array![
            [0.0, 0.0, 15.0, 15.0],
            [0.0, 0.0, 31.0, 31.0]
        ];

        let height = 2;
        let width = 2;
        let stride = 16;

        let all_anchors = anchors(height, width, stride, &base_anchors);
        println!("All anchors:\n{:?}", all_anchors);
    }
}