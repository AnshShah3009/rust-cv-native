use crate::descriptor::Descriptors;
use cv_core::{FeatureMatch, Matches};
use std::cmp::Ordering;

pub enum MatchType {
    BruteForce,
    BruteForceHamming,
}

pub struct Matcher {
    match_type: MatchType,
    cross_check: bool,
    ratio_threshold: Option<f32>,
}

impl Matcher {
    pub fn new(match_type: MatchType) -> Self {
        Self {
            match_type,
            cross_check: false,
            ratio_threshold: None,
        }
    }

    pub fn with_cross_check(mut self) -> Self {
        self.cross_check = true;
        self
    }

    pub fn with_ratio_test(mut self, threshold: f32) -> Self {
        self.ratio_threshold = Some(threshold);
        self
    }

    pub fn match_descriptors(&self, query: &Descriptors, train: &Descriptors) -> Matches {
        match self.match_type {
            MatchType::BruteForce | MatchType::BruteForceHamming => {
                self.brute_force_match(query, train)
            }
        }
    }

    fn brute_force_match(&self, query: &Descriptors, train: &Descriptors) -> Matches {
        let mut matches = Matches::with_capacity(query.len());

        for (query_idx, q_desc) in query.iter().enumerate() {
            let mut best_match: Option<(usize, u32)> = None;
            let mut second_best: Option<u32> = None;

            for (train_idx, t_desc) in train.iter().enumerate() {
                let distance = q_desc.hamming_distance(t_desc);

                match best_match {
                    None => {
                        best_match = Some((train_idx, distance));
                    }
                    Some((_, best_dist)) => {
                        if distance < best_dist {
                            second_best = Some(best_dist);
                            best_match = Some((train_idx, distance));
                        } else if second_best.is_none() || distance < second_best.unwrap() {
                            second_best = Some(distance);
                        }
                    }
                }
            }

            if let Some((train_idx, distance)) = best_match {
                let mut keep_match = true;

                if let Some(threshold) = self.ratio_threshold {
                    if let Some(second) = second_best {
                        let ratio = distance as f32 / second as f32;
                        if ratio > threshold {
                            keep_match = false;
                        }
                    }
                }

                if self.cross_check {
                    let reverse_match = self.find_best_match(train, query, train_idx);
                    if reverse_match != Some(query_idx) {
                        keep_match = false;
                    }
                }

                if keep_match {
                    let m = FeatureMatch::new(query_idx as i32, train_idx as i32, distance as f32);
                    matches.push(m);
                }
            }
        }

        matches
    }

    fn find_best_match(
        &self,
        query: &Descriptors,
        train: &Descriptors,
        query_idx: usize,
    ) -> Option<usize> {
        let q_desc = query.descriptors.get(query_idx)?;

        let mut best_idx = 0;
        let mut best_dist = q_desc.hamming_distance(&train.descriptors[0]);

        for (idx, t_desc) in train.iter().enumerate().skip(1) {
            let distance = q_desc.hamming_distance(t_desc);
            if distance < best_dist {
                best_dist = distance;
                best_idx = idx;
            }
        }

        Some(best_idx)
    }
}

pub fn match_descriptors(
    query: &Descriptors,
    train: &Descriptors,
    ratio_threshold: Option<f32>,
) -> Matches {
    let mut matcher = Matcher::new(MatchType::BruteForceHamming);

    if let Some(threshold) = ratio_threshold {
        matcher = matcher.with_ratio_test(threshold);
    }

    matcher.match_descriptors(query, train)
}

pub fn knn_match(query: &Descriptors, train: &Descriptors, k: usize) -> Vec<Vec<FeatureMatch>> {
    let mut all_matches: Vec<Vec<FeatureMatch>> = Vec::with_capacity(query.len());

    for (query_idx, q_desc) in query.iter().enumerate() {
        let mut distances: Vec<(usize, u32)> = train
            .iter()
            .enumerate()
            .map(|(idx, t_desc)| (idx, q_desc.hamming_distance(t_desc)))
            .collect();

        distances.sort_by(|a, b| a.1.cmp(&b.1));

        let knn: Vec<FeatureMatch> = distances
            .into_iter()
            .take(k)
            .map(|(train_idx, distance)| {
                FeatureMatch::new(query_idx as i32, train_idx as i32, distance as f32)
            })
            .collect();

        all_matches.push(knn);
    }

    all_matches
}

pub fn filter_matches_by_distance(matches: &mut Matches, max_distance: f32) {
    matches.filter_by_distance(max_distance);
}

pub fn filter_matches_by_ratio_test(
    matches: &[Vec<FeatureMatch>],
    ratio: f32,
) -> Vec<FeatureMatch> {
    let mut good_matches = Vec::new();

    for knn in matches {
        if knn.len() >= 2 {
            let best = &knn[0];
            let second = &knn[1];

            if best.distance < ratio * second.distance {
                good_matches.push(*best);
            }
        }
    }

    good_matches
}
