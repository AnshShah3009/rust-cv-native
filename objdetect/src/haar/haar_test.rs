use super::*;
use image::Luma;

#[test]
fn test_haar_integral_image() {
    let mut img = GrayImage::new(4, 4);
    for y in 0..4 {
        for x in 0..4 {
            img.put_pixel(x, y, Luma([1]));
        }
    }
    
    let integral = compute_integral_image(&img);
    // 4x4 image -> 5x5 integral image
    assert_eq!(integral.shape.width, 5);
    assert_eq!(integral.shape.height, 5);
    
    // Sum of 2x2 area at (1, 1)
    let sum = get_rect_sum(&integral, 1, 1, 2, 2);
    assert_eq!(sum, 4);
}

#[test]
fn test_haar_detection_simple() {
    // Create a 100x100 black image with a 20x20 white square at (40, 40)
    let mut img = GrayImage::new(100, 100);
    for y in 40..60 {
        for x in 40..60 {
            img.put_pixel(x, y, Luma([255]));
        }
    }
    
    // Create a simple cascade that detects a 20x20 white square
    // A single stage with one feature: a 20x20 rectangle with weight 1.0
    let feature = HaarFeature {
        rects: vec![(Rect::new(0.0, 0.0, 20.0, 20.0), 1.0)],
        threshold: 100.0, // Should be much higher for a white square
        left_val: -1.0,
        right_val: 1.0,
    };
    
    let stage = CascadeStage {
        threshold: 0.5,
        features: vec![feature],
    };
    
    let cascade = HaarCascade {
        stages: vec![stage],
        size: (20, 20),
    };
    
    let detections = cascade.detect(&img, 1.1, 0);
    println!("Detections: {:?}", detections);
    
    assert!(!detections.is_empty());
    // Check if at least one detection is near (40, 40)
    let found = detections.iter().any(|r| (r.x - 40.0).abs() < 5.0 && (r.y - 40.0).abs() < 5.0);
    assert!(found);
}
