use cv_features::*;
use cv_core::KeyPoint;
use image::{GrayImage, Luma};

#[test]
fn test_harris_detect_finds_corners() {
    let mut img = GrayImage::new(50, 50);
    // Draw a white square
    for y in 10..40 {
        for x in 10..40 {
            img.put_pixel(x, y, Luma([255]));
        }
    }
    
    let kps = harris_detect(&img, 3, 3, 0.04, 1000.0);
    assert!(!kps.keypoints.is_empty());
    
    // Check if some keypoints are near the corners (10,10), (39,10), etc.
    let has_top_left = kps.keypoints.iter().any(|kp| (kp.x - 10.0).abs() < 2.0 && (kp.y - 10.0).abs() < 2.0);
    assert!(has_top_left);
}

#[test]
fn test_brute_force_match_basic() {
    use cv_features::descriptor::Descriptor;
    
    let mut q_desc = Descriptors::new();
    let mut t_desc = Descriptors::new();
    
    let kp = KeyPoint::new(0.0, 0.0);
    
    // Create identical descriptors
    let d1 = Descriptor { data: vec![0xAA; 32], keypoint: kp.clone() };
    let d2 = Descriptor { data: vec![0x55; 32], keypoint: kp };
    
    q_desc.descriptors.push(d1.clone());
    q_desc.descriptors.push(d2.clone());
    
    t_desc.descriptors.push(d1);
    t_desc.descriptors.push(d2);
    
    let matcher = Matcher::new(MatchType::BruteForceHamming);
    let matches = matcher.match_descriptors(&q_desc, &t_desc);
    
    assert_eq!(matches.len(), 2);
    assert_eq!(matches.matches[0].query_idx, 0);
    assert_eq!(matches.matches[0].train_idx, 0);
    assert_eq!(matches.matches[1].query_idx, 1);
    assert_eq!(matches.matches[1].train_idx, 1);
}

#[test]
fn test_compute_hog_dimensions() {
    let img = GrayImage::new(64, 128);
    let params = HogParams::default();
    let descriptor = compute_hog(&img, &params);
    
    // Default: 8x8 cell, 2x2 block, 9 bins
    // 64/8 = 8 cells x, 128/8 = 16 cells y
    // blocks: (8-2+1) * (16-2+1) = 7 * 15 = 105 blocks
    // each block: 2*2*9 = 36 values
    // total: 105 * 36 = 3780
    assert_eq!(descriptor.len(), 3780);
}
