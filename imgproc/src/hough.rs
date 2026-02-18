use image::GrayImage;
use rayon::prelude::*;
use crate::{canny, sobel};
use cv_runtime::orchestrator::{ResourceGroup, scheduler};
use std::sync::atomic::{AtomicU32, Ordering};

/// Hough Circle result
#[derive(Debug, Clone, Copy)]
pub struct Circle {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub score: f32,
}

/// Hough Line result (rho, theta)
#[derive(Debug, Clone, Copy)]
pub struct Line {
    pub rho: f32,
    pub theta: f32,
    pub score: u32,
}

pub fn hough_circles(
    src: &GrayImage,
    min_radius: f32,
    max_radius: f32,
    threshold: u32,
) -> Vec<Circle> {
    let group = scheduler().get_default_group();
    hough_circles_ctx(src, min_radius, max_radius, threshold, &group)
}

/// Optimized Hough Circle Transform using gradient direction
pub fn hough_circles_ctx(
    src: &GrayImage,
    min_radius: f32,
    max_radius: f32,
    threshold: u32,
    group: &ResourceGroup,
) -> Vec<Circle> {
    let edges = canny(src, 50, 150);
    let (gx, gy) = sobel(src); // Sobel for gradients
    
    let width = src.width() as usize;
    let height = src.height() as usize;
    
    let mut all_circles = Vec::new();
    
    // Accumulator for centers (2D)
    // We iterate through radius range to keep memory usage sane
    for r in (min_radius as i32)..=(max_radius as i32) {
        let r_f = r as f32;
        let mut acc = vec![0u32; width * height];
        // Wrap in atomics for parallel accumulation
        let acc_atomic: &[AtomicU32] = unsafe { std::mem::transmute(&acc[..]) };

        group.run(|| {
            edges.as_raw().par_iter().enumerate().for_each(|(i, &e)| {
                if e > 0 {
                    let ex = (i % width) as i32;
                    let ey = (i / width) as i32;
                    
                    // Gradient direction
                    let dx = gx.as_raw()[i] as f32 - 128.0;
                    let dy = gy.as_raw()[i] as f32 - 128.0;
                    let angle = dy.atan2(dx);
                    
                    // Possible centers along the gradient line
                    for sign in [-1.0, 1.0] {
                        let cx = ex as f32 + sign * r_f * angle.cos();
                        let cy = ey as f32 + sign * r_f * angle.sin();
                        
                        if cx >= 0.0 && cx < width as f32 && cy >= 0.0 && cy < height as f32 {
                            let idx = cy as usize * width + cx as usize;
                            acc_atomic[idx].fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            });
        });

        // Find peaks in this radius slice
        for (i, count) in acc.iter().enumerate() {
            if *count >= threshold {
                all_circles.push(Circle {
                    x: (i % width) as f32,
                    y: (i / width) as f32,
                    radius: r_f,
                    score: *count as f32,
                });
            }
        }
    }
    
    // NMS for circles (TODO)
    all_circles
}

pub fn hough_lines(
    src: &GrayImage,
    rho_res: f32,
    theta_res: f32,
    threshold: u32,
) -> Vec<Line> {
    let group = scheduler().get_default_group();
    hough_lines_ctx(src, rho_res, theta_res, threshold, &group)
}

pub fn hough_lines_ctx(
    src: &GrayImage,
    rho_res: f32,
    theta_res: f32,
    threshold: u32,
    group: &ResourceGroup,
) -> Vec<Line> {
    let edges = canny(src, 50, 150);
    let width = src.width();
    let height = src.height();
    
    let max_rho = (width as f32 * width as f32 + height as f32 * height as f32).sqrt();
    let num_rho = (max_rho / rho_res) as usize * 2;
    let num_theta = (std::f32::consts::PI / theta_res) as usize;
    
    let mut acc = vec![0u32; num_rho * num_theta];
    let acc_atomic: &[AtomicU32] = unsafe { std::mem::transmute(&acc[..]) };

    group.run(|| {
        edges.as_raw().par_iter().enumerate().for_each(|(i, &e)| {
            if e > 0 {
                let x = (i % width as usize) as f32;
                let y = (i / width as usize) as f32;
                
                for t_idx in 0..num_theta {
                    let theta = t_idx as f32 * theta_res;
                    let rho = x * theta.cos() + y * theta.sin();
                    let rho_idx = ((rho + max_rho) / rho_res) as usize;
                    
                    if rho_idx < num_rho {
                        acc_atomic[rho_idx * num_theta + t_idx].fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        });
    });

    let mut lines = Vec::new();
    for r_idx in 0..num_rho {
        for t_idx in 0..num_theta {
            let score = acc[r_idx * num_theta + t_idx];
            if score >= threshold {
                lines.push(Line {
                    rho: r_idx as f32 * rho_res - max_rho,
                    theta: t_idx as f32 * theta_res,
                    score,
                });
            }
        }
    }
    
    lines
}
