//! Video processing and optical flow
//!
//! This module provides algorithms for motion estimation and tracking
//! in video sequences.

use image::GrayImage;

pub mod optical_flow;
pub mod tracking;

pub use optical_flow::*;
pub use tracking::*;

pub type Result<T> = std::result::Result<T, VideoError>;

#[derive(Debug, thiserror::Error)]
pub enum VideoError {
    #[error("Image size mismatch: {0}")]
    SizeMismatch(String),

    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),

    #[error("Tracking error: {0}")]
    TrackingError(String),
}

/// Video frame representation
#[derive(Debug, Clone)]
pub struct VideoFrame {
    pub image: GrayImage,
    pub timestamp: f64,
    pub frame_number: usize,
}

impl VideoFrame {
    pub fn new(image: GrayImage, timestamp: f64, frame_number: usize) -> Self {
        Self {
            image,
            timestamp,
            frame_number,
        }
    }
}

/// Motion vector field
#[derive(Debug, Clone)]
pub struct MotionField {
    pub u: Vec<f32>,  // Horizontal motion
    pub v: Vec<f32>,  // Vertical motion
    pub width: u32,
    pub height: u32,
}

impl MotionField {
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width * height) as usize;
        Self {
            u: vec![0.0; size],
            v: vec![0.0; size],
            width,
            height,
        }
    }

    pub fn get_motion(&self, x: u32, y: u32) -> (f32, f32) {
        let idx = (y * self.width + x) as usize;
        (self.u[idx], self.v[idx])
    }

    pub fn set_motion(&mut self, x: u32, y: u32, u: f32, v: f32) {
        let idx = (y * self.width + x) as usize;
        self.u[idx] = u;
        self.v[idx] = v;
    }

    /// Compute magnitude of motion at each pixel
    pub fn magnitude(&self) -> Vec<f32> {
        self.u.iter()
            .zip(self.v.iter())
            .map(|(u, v)| (u * u + v * v).sqrt())
            .collect()
    }

    /// Visualize motion field as HSV image (converted to RGB)
    pub fn visualize(&self) -> image::RgbImage {
        use image::Rgb;
        
        let mut img = image::RgbImage::new(self.width, self.height);
        
        // Find max magnitude for normalization
        let max_mag = self.magnitude()
            .into_iter()
            .fold(0.0f32, f32::max)
            .max(1.0);
        
        for y in 0..self.height {
            for x in 0..self.width {
                let (u, v) = self.get_motion(x, y);
                
                // Compute angle and magnitude
                let angle = v.atan2(u);  // -PI to PI
                let magnitude = (u * u + v * v).sqrt();
                
                // Convert to HSV: Hue = angle, Saturation = 1.0, Value = magnitude
                let hue = ((angle + std::f32::consts::PI) / (2.0 * std::f32::consts::PI) * 255.0) as u8;
                let saturation = 255u8;
                let value = ((magnitude / max_mag) * 255.0) as u8;
                
                // Simple HSV to RGB conversion
                let rgb = hsv_to_rgb(hue, saturation, value);
                img.put_pixel(x, y, Rgb(rgb));
            }
        }
        
        img
    }
}

/// Simple HSV to RGB conversion
fn hsv_to_rgb(h: u8, s: u8, v: u8) -> [u8; 3] {
    let h = h as f32 / 255.0 * 360.0;
    let s = s as f32 / 255.0;
    let v = v as f32 / 255.0;
    
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    
    let (r, g, b) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    
    [
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_field() {
        let mut field = MotionField::new(10, 10);
        
        field.set_motion(5, 5, 3.0, 4.0);
        let (u, v) = field.get_motion(5, 5);
        
        assert_eq!(u, 3.0);
        assert_eq!(v, 4.0);
        
        let mag = field.magnitude();
        assert_eq!(mag[55], 5.0);  // sqrt(3^2 + 4^2) = 5
    }

    #[test]
    fn test_video_frame() {
        let img = GrayImage::new(100, 100);
        let frame = VideoFrame::new(img, 0.0, 0);
        
        assert_eq!(frame.frame_number, 0);
        assert_eq!(frame.timestamp, 0.0);
    }
}
