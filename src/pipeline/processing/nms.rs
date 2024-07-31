use ndarray::{Array1, Array2};

pub fn nms(dets: &Array2<f32>, thresh: f32) -> Vec<usize> {
    let scores = dets.column(4);
    let mut order: Vec<usize> = (0..dets.nrows()).collect();
    order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut keep: Vec<usize> = Vec::new();

    while !order.is_empty() {
        let i = order[0];
        keep.push(i);

        let xx1 = Array1::from(
            order[1..]
                .iter()
                .map(|&j| f32::max(dets[[i, 0]], dets[[j, 0]]))
                .collect::<Vec<f32>>(),
        );
        let yy1 = Array1::from(
            order[1..]
                .iter()
                .map(|&j| f32::max(dets[[i, 1]], dets[[j, 1]]))
                .collect::<Vec<f32>>(),
        );
        let xx2 = Array1::from(
            order[1..]
                .iter()
                .map(|&j| f32::min(dets[[i, 2]], dets[[j, 2]]))
                .collect::<Vec<f32>>(),
        );
        let yy2 = Array1::from(
            order[1..]
                .iter()
                .map(|&j| f32::min(dets[[i, 3]], dets[[j, 3]]))
                .collect::<Vec<f32>>(),
        );

        let w = &xx2 - &xx1 + 1.0;
        let h = &yy2 - &yy1 + 1.0;

        let w = w.mapv(|val| f32::max(0.0, val));
        let h = h.mapv(|val| f32::max(0.0, val));

        let inter = &w * &h;
        let area_i = (dets[[i, 2]] - dets[[i, 0]] + 1.0) * (dets[[i, 3]] - dets[[i, 1]] + 1.0);
        let area_order = Array1::from(
            order[1..]
                .iter()
                .map(|&j| (dets[[j, 2]] - dets[[j, 0]] + 1.0) * (dets[[j, 3]] - dets[[j, 1]] + 1.0))
                .collect::<Vec<f32>>(),
        );

        let ovr = inter.clone() / (area_i + &area_order - inter);

        let inds: Vec<usize> = ovr
            .indexed_iter()
            .filter_map(|(idx, &val)| if val <= thresh { Some(idx) } else { None })
            .collect();

        order = inds.iter().map(|&idx| order[idx + 1]).collect();
    }

    keep
}



#[cfg(test)]
mod tests {
    use ndarray::array;
    use crate::processing::nms::nms;

    #[test]
    fn test_nms() {
        let dets = array![
            [100.0, 100.0, 210.0, 210.0, 0.72],
            [250.0, 250.0, 420.0, 420.0, 0.8],
            [220.0, 220.0, 320.0, 330.0, 0.92],
            [100.0, 100.0, 210.0, 210.0, 0.6]
        ];

        let thresh = 0.4;
        let keep = nms(&dets, thresh);

        println!("Kept indices: {:?}", keep);
    }

}