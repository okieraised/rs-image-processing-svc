use ndarray::{Array2, stack, Axis, s};
pub fn bbox_overlaps_py(boxes: &Array2<f32>, query_boxes: &Array2<f32>) -> Array2<f32> {
    let n_ = boxes.nrows();
    let k_ = query_boxes.nrows();
    let mut overlaps = Array2::<f32>::zeros((n_, k_));

    for k in 0..k_ {
        let query_box_area = (query_boxes[[k, 2]] - query_boxes[[k, 0]] + 1.0) *
            (query_boxes[[k, 3]] - query_boxes[[k, 1]] + 1.0);
        for n in 0..n_ {
            let iw = (boxes[[n, 2]].min(query_boxes[[k, 2]]) - boxes[[n, 0]].max(query_boxes[[k, 0]]) + 1.0).max(0.0);
            if iw > 0.0 {
                let ih = (boxes[[n, 3]].min(query_boxes[[k, 3]]) - boxes[[n, 1]].max(query_boxes[[k, 1]]) + 1.0).max(0.0);
                if ih > 0.0 {
                    let box_area = (boxes[[n, 2]] - boxes[[n, 0]] + 1.0) * (boxes[[n, 3]] - boxes[[n, 1]] + 1.0);
                    let all_area = box_area + query_box_area - iw * ih;
                    overlaps[[n, k]] = iw * ih / all_area;
                }
            }
        }
    }

    overlaps
}


pub fn clip_boxes(boxes: &mut Array2<f32>, im_shape: (usize, usize)) {
    let rows = boxes.nrows();
    let cols = boxes.ncols();
    let width = im_shape.1 as f32 - 1.0;
    let height = im_shape.0 as f32 - 1.0;

    for i in 0..rows {
        for j in (0..cols).step_by(4) {
            // Clip x1 to [0, width]
            boxes[(i, j)] = boxes[(i, j)].min(width).max(0.0);
            // Clip y1 to [0, height]
            boxes[(i, j + 1)] = boxes[(i, j + 1)].min(height).max(0.0);
            // Clip x2 to [0, width]
            boxes[(i, j + 2)] = boxes[(i, j + 2)].min(width).max(0.0);
            // Clip y2 to [0, height]
            boxes[(i, j + 3)] = boxes[(i, j + 3)].min(height).max(0.0);
        }
    }
}

pub fn clip_points(points: &mut Array2<f32>, im_shape: (usize, usize)) {
    let rows = points.nrows();
    let cols = points.ncols();
    let width = im_shape.1 as f32 - 1.0;
    let height = im_shape.0 as f32 - 1.0;

    for i in 0..rows {
        for j in (0..cols).step_by(10) {
            // Clip x1, x2, ..., x9 to [0, width]
            for k in (0..10).step_by(2) {
                points[(i, j + k)] = points[(i, j + k)].min(width).max(0.0);
            }
            // Clip y1, y2, ..., y9 to [0, height]
            for k in (1..10).step_by(2) {
                points[(i, j + k)] = points[(i, j + k)].min(height).max(0.0);
            }
        }
    }
}

pub fn nonlinear_transform(ex_rois: &Array2<f32>, gt_rois: &Array2<f32>) -> Array2<f32> {
    assert_eq!(ex_rois.nrows(), gt_rois.nrows(), "inconsistent rois number");

    let ex_widths = &ex_rois.slice(s![.., 2]) - &ex_rois.slice(s![.., 0]) + 1.0;
    let ex_heights = &ex_rois.slice(s![.., 3]) - &ex_rois.slice(s![.., 1]) + 1.0;
    let ex_ctr_x = &ex_rois.slice(s![.., 0]) + 0.5 * (&ex_widths - 1.0);
    let ex_ctr_y = &ex_rois.slice(s![.., 1]) + 0.5 * (&ex_heights - 1.0);

    let gt_widths = &gt_rois.slice(s![.., 2]) - &gt_rois.slice(s![.., 0]) + 1.0;
    let gt_heights = &gt_rois.slice(s![.., 3]) - &gt_rois.slice(s![.., 1]) + 1.0;
    let gt_ctr_x = &gt_rois.slice(s![.., 0]) + 0.5 * (&gt_widths - 1.0);
    let gt_ctr_y = &gt_rois.slice(s![.., 1]) + 0.5 * (&gt_heights - 1.0);

    let targets_dx = (&gt_ctr_x - &ex_ctr_x) / (&ex_widths + 1e-14);
    let targets_dy = (&gt_ctr_y - &ex_ctr_y) / (&ex_heights + 1e-14);
    let targets_dw = (&gt_widths / &ex_widths).mapv(|x| x.ln());
    let targets_dh = (&gt_heights / &ex_heights).mapv(|x| x.ln());

    let targets = stack![Axis(1), targets_dx, targets_dy, targets_dw, targets_dh];

    targets
}

pub fn nonlinear_pred(boxes: &Array2<f32>, box_deltas: &Array2<f32>) -> Array2<f32> {
    if boxes.nrows() == 0 {
        return Array2::zeros((0, box_deltas.ncols()));
    }

    let widths = &boxes.slice(s![.., 2]) - &boxes.slice(s![.., 0]) + 1.0;
    let heights = &boxes.slice(s![.., 3]) - &boxes.slice(s![.., 1]) + 1.0;
    let ctr_x = &boxes.slice(s![.., 0]) + 0.5 * (&widths - 1.0);
    let ctr_y = &boxes.slice(s![.., 1]) + 0.5 * (&heights - 1.0);

    let dx = box_deltas.slice(s![.., ..;4]);
    let dy = box_deltas.slice(s![.., 1..;4]);
    let dw = box_deltas.slice(s![.., 2..;4]);
    let dh = box_deltas.slice(s![.., 3..;4]);

    let n_width = widths.insert_axis(Axis(1));
    let n_height = heights.insert_axis(Axis(1));

    let pred_ctr_x = &dx.into_dyn() * &n_width + &ctr_x.insert_axis(Axis(1));
    let pred_ctr_y = &dy.into_dyn() * &n_height + &ctr_y.insert_axis(Axis(1));
    let pred_w = dw.mapv(f32::exp) * &n_width;
    let pred_h = dh.mapv(f32::exp) * &n_height;

    let mut pred_boxes = Array2::<f32>::zeros(box_deltas.raw_dim());
    pred_boxes.slice_mut(s![.., ..;4]).assign(&(&pred_ctr_x - 0.5 * (&pred_w - 1.0)));
    pred_boxes.slice_mut(s![.., 1..;4]).assign(&(&pred_ctr_y - 0.5 * (&pred_h - 1.0)));
    pred_boxes.slice_mut(s![.., 2..;4]).assign(&(&pred_ctr_x + 0.5 * (&pred_w - 1.0)));
    pred_boxes.slice_mut(s![.., 3..;4]).assign(&(&pred_ctr_y + 0.5 * (&pred_h - 1.0)));

    pred_boxes
}


pub fn landmark_pred(boxes: &Array2<f32>, point_deltas: &Array2<f32>) -> Array2<f32> {
    if boxes.nrows() == 0 {
        return Array2::zeros((0, point_deltas.ncols()));
    }

    let boxes = boxes.mapv(|x| x as f32);

    let widths = &boxes.slice(s![.., 2]) - &boxes.slice(s![.., 0]) + 1.0;
    let heights = &boxes.slice(s![.., 3]) - &boxes.slice(s![.., 1]) + 1.0;
    let ctr_x = &boxes.slice(s![.., 0]) + 0.5 * (&widths - 1.0);
    let ctr_y = &boxes.slice(s![.., 1]) + 0.5 * (&heights - 1.0);

    let d1x = point_deltas.slice(s![.., 0]);
    let d1y = point_deltas.slice(s![.., 1]);
    let d2x = point_deltas.slice(s![.., 2]);
    let d2y = point_deltas.slice(s![.., 3]);
    let d3x = point_deltas.slice(s![.., 4]);
    let d3y = point_deltas.slice(s![.., 5]);
    let d4x = point_deltas.slice(s![.., 6]);
    let d4y = point_deltas.slice(s![.., 7]);
    let d5x = point_deltas.slice(s![.., 8]);
    let d5y = point_deltas.slice(s![.., 9]);

    let mut pred_points = Array2::<f32>::zeros(point_deltas.raw_dim());

    pred_points.slice_mut(s![.., 0]).assign(&(&d1x * &widths + &ctr_x));
    pred_points.slice_mut(s![.., 1]).assign(&(&d1y * &heights + &ctr_y));
    pred_points.slice_mut(s![.., 2]).assign(&(&d2x * &widths + &ctr_x));
    pred_points.slice_mut(s![.., 3]).assign(&(&d2y * &heights + &ctr_y));
    pred_points.slice_mut(s![.., 4]).assign(&(&d3x * &widths + &ctr_x));
    pred_points.slice_mut(s![.., 5]).assign(&(&d3y * &heights + &ctr_y));
    pred_points.slice_mut(s![.., 6]).assign(&(&d4x * &widths + &ctr_x));
    pred_points.slice_mut(s![.., 7]).assign(&(&d4y * &heights + &ctr_y));
    pred_points.slice_mut(s![.., 8]).assign(&(&d5x * &widths + &ctr_x));
    pred_points.slice_mut(s![.., 9]).assign(&(&d5y * &heights + &ctr_y));

    pred_points
}

pub fn iou_pred(boxes: &Array2<f32>, box_deltas: &Array2<f32>, num_classes: usize) -> Array2<f32> {
    let num_boxes = boxes.shape()[0];
    if num_boxes == 0 {
        return Array2::<f32>::zeros((0, box_deltas.shape()[1]));
    }

    let x1 = boxes.slice(s![.., 0]);
    let y1 = boxes.slice(s![.., 1]);
    let x2 = boxes.slice(s![.., 2]);
    let y2 = boxes.slice(s![.., 3]);

    let dx1 = box_deltas.slice(s![.., 0..num_classes*4; 4]);
    let dy1 = box_deltas.slice(s![.., 1..num_classes*4; 4]);
    let dx2 = box_deltas.slice(s![.., 2..num_classes*4; 4]);
    let dy2 = box_deltas.slice(s![.., 3..num_classes*4; 4]);

    let mut pred_boxes = Array2::<f32>::zeros(box_deltas.raw_dim());

    pred_boxes.slice_mut(s![.., 0..num_classes*4; 4]).assign(&(&dx1.into_dyn() + &x1.insert_axis(Axis(1))));
    pred_boxes.slice_mut(s![.., 1..num_classes*4; 4]).assign(&(&dy1.into_dyn() + &y1.insert_axis(Axis(1))));
    pred_boxes.slice_mut(s![.., 2..num_classes*4; 4]).assign(&(&dx2.into_dyn() + &x2.insert_axis(Axis(1))));
    pred_boxes.slice_mut(s![.., 3..num_classes*4; 4]).assign(&(&dy2.into_dyn() + &y2.insert_axis(Axis(1))));

    pred_boxes
}



#[cfg(test)]
mod tests {
    use ndarray::{array};
    use crate::pipeline::processing::bbox_transform::{clip_boxes, clip_points, landmark_pred, nonlinear_transform, nonlinear_pred, iou_pred};

    #[test]
    fn test_iou_pred() {
        // Example input
        let boxes = array![
            [50.0, 50.0, 100.0, 100.0],
            [30.0, 30.0, 70.0, 70.0]
        ];
        let box_deltas = array![
            [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
        ];
        let num_classes = 2;

        let pred_boxes = iou_pred(&boxes, &box_deltas, num_classes);
        println!("{:?}", pred_boxes);
    }
    #[test]
    fn test_clip_boxes() {
        let mut boxes = array![
            [50.0, 50.0, 150.0, 150.0, 60.0, 60.0, 160.0, 160.0],
            [30.0, 30.0, 200.0, 200.0, 40.0, 40.0, 220.0, 220.0]
        ];
        let im_shape = (100, 100);

        clip_boxes(&mut boxes, im_shape);
        println!("Clipped Boxes: {:?}", boxes);
    }

    #[test]
    fn test_clip_points() {
        let mut points = array![
            [50.0, 50.0, 150.0, 150.0, 60.0, 60.0, 160.0, 160.0, 70.0, 70.0],
            [30.0, 30.0, 200.0, 200.0, 40.0, 40.0, 220.0, 220.0, 80.0, 80.0]
        ];

        let im_shape = (100, 100);

        clip_points(&mut points, im_shape);
        println!("Clipped Points: {:?}", points);
    }

    #[test]
    fn test_nonlinear_transform() {
        let ex_rois = array![
            [50.0, 50.0, 150.0, 150.0],
            [30.0, 30.0, 200.0, 200.0]
        ];
        let gt_rois = array![
            [60.0, 60.0, 170.0, 170.0],
            [35.0, 35.0, 210.0, 210.0]
        ];

        let targets = nonlinear_transform(&ex_rois, &gt_rois);
        println!("Targets: {:?}", targets);
    }

    #[test]
    fn test_nonlinear_pred() {
        let boxes = array![
        [50.0, 50.0, 150.0, 150.0],
        [30.0, 30.0, 200.0, 200.0]
    ];
        let box_deltas = array![
        [0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2, 0.1],
        [0.2, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2]
    ];

        let pred_boxes = nonlinear_pred(&boxes, &box_deltas);
        println!("Predicted Boxes: {:?}", pred_boxes);
    }

    #[test]
    fn test_landmark_pred() {
        let boxes = array![
            [50.0, 50.0, 150.0, 150.0],
            [30.0, 30.0, 200.0, 200.0]
        ];
        let point_deltas = array![
            [0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2, 0.1, 0.3, 0.3],
            [0.2, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2, 0.3, 0.3]
        ];

        let pred_points = landmark_pred(&boxes, &point_deltas);
        println!("Predicted Points: {:?}", pred_points);
    }
}