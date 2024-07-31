use ndarray::{array, Array, Array1, Array2, Axis, s};
use std::ops::{AddAssign};
use std::collections::HashMap;



#[derive(Debug, Clone)]
pub struct Config {
    pub rpn_anchor_cfg: HashMap<String, AnchorConfig>,
}

#[derive(Debug, Clone)]
pub struct AnchorConfig {
    pub base_size: i32,
    pub ratios: Vec<f32>,
    pub scales: Vec<f32>,
    pub allowed_border: i32,
}

pub fn _whctrs(anchor: &Array1<f32>) -> (f32, f32, f32, f32) {
    let w = anchor[2] - anchor[0] + 1.0;
    let h = anchor[3] - anchor[1] + 1.0;
    let x_ctr = anchor[0] + 0.5 * (w - 1.0);
    let y_ctr = anchor[1] + 0.5 * (h - 1.0);
    (w, h, x_ctr, y_ctr)
}

pub fn _mkanchors(ws: Array1<f32>, hs: Array1<f32>, x_ctr: f32, y_ctr: f32) -> Array2<f32> {
    let anchors = Array2::from_shape_fn((ws.len(), 4), |(i, j)| {
        match j {
            0 => x_ctr - 0.5 * (ws[i] - 1.0),
            1 => y_ctr - 0.5 * (hs[i] - 1.0),
            2 => x_ctr + 0.5 * (ws[i] - 1.0),
            3 => y_ctr + 0.5 * (hs[i] - 1.0),
            _ => unreachable!(),
        }
    });
    anchors
}

pub fn generate_anchors(
    base_size: usize,
    ratios: Array1<f32>,
    scales: Array1<f32>,
) -> Array2<f32> {
    let base_anchor = array![1.0, 1.0, base_size as f32, base_size as f32] - 1.0;
    let ratio_anchors = _ratio_enum(&base_anchor, &ratios);

    let anchors = ratio_anchors.axis_iter(Axis(0))
        .map(|anchor| _scale_enum(&anchor.to_owned(), &scales))
        .fold(Array2::<f32>::zeros((0, 4)), |acc, x| {
            let mut result = Array2::<f32>::zeros((acc.shape()[0] + x.shape()[0], 4));
            result.slice_mut(s![0..acc.shape()[0], ..]).assign(&acc);
            result.slice_mut(s![acc.shape()[0].., ..]).assign(&x);
            result
        });

    anchors
}

pub fn generate_anchors2(
    base_size: usize,
    ratios: Array1<f32>,
    scales: Array1<f32>,
    stride: usize,
    dense_anchor: bool,
) -> Array2<f32> {
    let base_anchor = array![1.0, 1.0, base_size as f32, base_size as f32] - 1.0;
    let ratio_anchors = _ratio_enum(&base_anchor, &ratios);

    let anchors = ratio_anchors.axis_iter(Axis(0))
        .map(|anchor| _scale_enum(&anchor.to_owned(), &scales))
        .fold(Array2::<f32>::zeros((0, 4)), |acc, x| {
            let mut result = Array2::<f32>::zeros((acc.shape()[0] + x.shape()[0], 4));
            result.slice_mut(s![0..acc.shape()[0], ..]).assign(&acc);
            result.slice_mut(s![acc.shape()[0].., ..]).assign(&x);
            result
        });

    let anchors = if dense_anchor {
        assert_eq!(stride % 2, 0);
        let mut anchors2 = anchors.clone();
        anchors2.add_assign(stride as f32 / 2.0);
        let mut result = Array2::<f32>::zeros((anchors.shape()[0] * 2, 4));
        result.slice_mut(s![0..anchors.shape()[0], ..]).assign(&anchors);
        result.slice_mut(s![anchors.shape()[0].., ..]).assign(&anchors2);
        result
    } else {
        anchors
    };

    anchors
}

pub fn generate_anchors_fpn(
    base_size: Vec<i32>,
    ratios: Vec<f32>,
    scales: Vec<f32>,
) -> Vec<Array2<f32>> {
    let mut anchors = Vec::new();
    let num_levels = base_size.len();

    let _ratios = Array::from_shape_vec((num_levels, 1), ratios).unwrap();
    let _scales = Array::from_shape_vec((num_levels, 1), scales).unwrap();

    for (i, &bs) in base_size.iter().enumerate() {
        let __ratios = _ratios.slice(s![i, ..]).to_owned();
        let __scales = _scales.slice(s![i, ..]).to_owned();
        let r = generate_anchors(bs as usize, __ratios, __scales);
        anchors.push(r);
    }

    anchors
}

pub fn generate_anchors_fpn2(dense_anchor: bool, cfg: Option<&Config>) -> Vec<Array2<f32>> {
    let config = if let Some(cfg) = cfg {
        cfg
    } else {
        unimplemented!("Config loading not implemented");
    };

    let mut rpn_feat_stride: Vec<i32> = config.rpn_anchor_cfg.keys().map(|k| k.parse::<i32>().unwrap()).collect();
    rpn_feat_stride.sort_unstable_by(|a, b| b.cmp(a)); // Sort in reverse order

    let mut anchors = Vec::new();
    for k in rpn_feat_stride {
        let v = &config.rpn_anchor_cfg[&k.to_string()];
        let bs = v.base_size;
        let ratios = Array::from(v.ratios.clone());
        let scales = Array::from(v.scales.clone());
        let stride = k;
        let r = generate_anchors2(bs as usize, ratios, scales, stride as usize, dense_anchor);
        anchors.push(r);
    }

    anchors
}


pub fn _ratio_enum(anchor: &Array1<f32>, ratios: &Array1<f32>) -> Array2<f32> {
    let (w, h, x_ctr, y_ctr) = _whctrs(anchor);
    let size = w * h;
    let size_ratios = size / ratios;
    let ws = size_ratios.mapv(|sr| sr.sqrt().round());
    let hs = ws.clone() * ratios;
    _mkanchors(ws, hs, x_ctr, y_ctr)
}


pub fn _scale_enum(anchor: &Array1<f32>, scales: &Array1<f32>) -> Array2<f32> {
    let (w, h, x_ctr, y_ctr) = _whctrs(anchor);
    let ws = scales.mapv(|scale| w * scale);
    let hs = scales.mapv(|scale| h * scale);
    let anchors = _mkanchors(ws, hs, x_ctr, y_ctr);
    anchors
}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use ndarray::array;
    use crate::pipeline::processing::generate_anchors::{_ratio_enum, _scale_enum, AnchorConfig, Config, generate_anchors, generate_anchors2, generate_anchors_fpn, generate_anchors_fpn2};

    #[test]
    fn test_ratio_enum() {
        let anchor = array![0.0, 0.0, 15.0, 15.0];
        let ratios = array![0.5, 1.0, 2.0];
        let anchors = _ratio_enum(&anchor, &ratios);
        println!("{:?}", anchors);
    }

    #[test]
    fn test_scale_enum() {
        let anchor = array![0.0, 0.0, 15.0, 15.0];
        let scales = array![0.5, 1.0, 2.0];
        let anchors = _scale_enum(&anchor, &scales);
        println!("{:?}", anchors);
    }

    #[test]
    fn test_generate_anchors2() {
        let base_size = 16;
        let ratios = array![0.5, 1.0, 2.0];
        let scales = array![8.0, 16.0, 32.0];
        let stride = 16;
        let dense_anchor = false;

        let anchors = generate_anchors2(base_size, ratios, scales, stride, dense_anchor);
        println!("{:?}", anchors);
    }

    #[test]
    fn test_generate_anchors() {
        let base_size = 16;
        let ratios = array![0.5, 1.0, 2.0];
        let scales = array![8.0, 16.0, 32.0];

        let anchors = generate_anchors(base_size, ratios, scales);
        println!("{:?}", anchors);
    }

    #[test]
    fn test_generate_anchors_fpn() {
        let base_size = vec![64, 32, 16, 8, 4];
        let ratios = vec![0.5, 1.0, 2.0, 1.0, 1.0];
        let scales = vec![8.0, 8.0, 8.0, 8.0, 8.0];

        let anchors = generate_anchors_fpn(base_size, ratios, scales);
        for anchor in anchors {
            println!("{:?}", anchor);
        }
    }

    #[test]
    fn test_generate_anchors_fpn2() {
        let mut cfg = HashMap::new();
        cfg.insert(
            "32".to_string(),
            AnchorConfig {
                base_size: 16,
                ratios: vec![1.0, ],
                scales: vec![32.0, 16.0],
                allowed_border: 9999,
            },
        );
        cfg.insert(
            "16".to_string(),
            AnchorConfig {
                base_size: 16,
                ratios: vec![1.0,],
                scales: vec![8.0, 4.0],
                allowed_border: 9999,
            },
        );
        cfg.insert(
            "8".to_string(),
            AnchorConfig {
                base_size: 16,
                ratios: vec![1.0, ],
                scales: vec![2.0, 1.0],
                allowed_border: 9999,
            },
        );

        let config = Config {
            rpn_anchor_cfg: cfg,
        };

        let anchors = generate_anchors_fpn2(false, Some(&config));
        for anchor in anchors {
            println!("{:?}", anchor);
        }
    }
}