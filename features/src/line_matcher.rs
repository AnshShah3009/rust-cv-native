//! Line Matching functionality

use crate::hough::LineSegment;

#[derive(Debug, Clone)]
pub struct LineDescriptor {
    pub data: Vec<u8>,
    pub segment: LineSegment,
}

#[derive(Debug, Clone, Copy)]
pub struct LineMatch {
    pub query_idx: usize,
    pub train_idx: usize,
    pub distance: f32,
}

pub struct LineMatcher {
    pub threshold: f32,
}

impl LineMatcher {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    pub fn match_lines(
        &self,
        query: &[LineDescriptor],
        train: &[LineDescriptor],
    ) -> Vec<LineMatch> {
        let mut matches = Vec::new();

        for (qi, q) in query.iter().enumerate() {
            let mut best_dist = u32::MAX;
            let mut best_idx = None;

            for (ti, t) in train.iter().enumerate() {
                let dist = hamming_dist(&q.data, &t.data);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = Some(ti);
                }
            }

            if let Some(ti) = best_idx {
                if (best_dist as f32) < self.threshold {
                    matches.push(LineMatch {
                        query_idx: qi,
                        train_idx: ti,
                        distance: best_dist as f32,
                    });
                }
            }
        }

        matches
    }
}

fn hamming_dist(a: &[u8], b: &[u8]) -> u32 {
    let mut dist = 0;
    for i in 0..a.len().min(b.len()) {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}
