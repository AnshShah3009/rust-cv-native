use image::{GrayImage, Luma, Rgb, RgbImage, RgbaImage};

pub fn convert_gray_to_rgb(gray: &GrayImage) -> RgbImage {
    let (w, h) = gray.dimensions();
    let mut rgb = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let val = gray.get_pixel(x, y)[0];
            rgb.put_pixel(x, y, Rgb([val, val, val]));
        }
    }
    rgb
}

pub fn convert_rgb_to_gray(rgb: &RgbImage) -> GrayImage {
    let (w, h) = rgb.dimensions();
    let mut gray = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let p = rgb.get_pixel(x, y);
            let gray_val = (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) as u8;
            gray.put_pixel(x, y, Luma([gray_val]));
        }
    }
    gray
}

pub fn to_gray(img: &RgbImage) -> GrayImage {
    convert_rgb_to_gray(img)
}

pub fn to_rgb(img: &GrayImage) -> RgbImage {
    convert_gray_to_rgb(img)
}

pub fn adjust_brightness(src: &GrayImage, factor: f32) -> GrayImage {
    let mut output = src.clone();
    for pixel in output.pixels_mut() {
        let val = (pixel[0] as f32 * factor).clamp(0.0, 255.0) as u8;
        pixel[0] = val;
    }
    output
}

pub fn adjust_contrast(src: &GrayImage, factor: f32) -> GrayImage {
    let mut output = src.clone();
    let mean = compute_mean_intensity(src);
    for pixel in output.pixels_mut() {
        let val = ((pixel[0] as f32 - mean) * factor + mean).clamp(0.0, 255.0) as u8;
        pixel[0] = val;
    }
    output
}

pub fn compute_mean_intensity(src: &GrayImage) -> f32 {
    let mut sum = 0u64;
    let mut count = 0u64;
    for pixel in src.pixels() {
        sum += pixel[0] as u64;
        count += 1;
    }
    (sum / count) as f32
}

pub fn invert(src: &GrayImage) -> GrayImage {
    let mut output = src.clone();
    for pixel in output.pixels_mut() {
        pixel[0] = 255 - pixel[0];
    }
    output
}

pub fn equalize_histogram(src: &GrayImage) -> GrayImage {
    let mut hist = [0u32; 256];
    for pixel in src.pixels() {
        hist[pixel[0] as usize] += 1;
    }

    let mut cdf = [0u32; 256];
    cdf[0] = hist[0];
    for i in 1..256 {
        cdf[i] = cdf[i - 1] + hist[i];
    }

    let cdf_min = cdf.iter().find(|&&x| x > 0).copied().unwrap_or(0);
    let total = src.width() * src.height() as u32;

    let mut lut = [0u8; 256];
    for i in 0..256 {
        let val = ((cdf[i] - cdf_min) as f32 / (total - cdf_min) as f32 * 255.0).round() as u8;
        lut[i] = val;
    }

    let mut output = src.clone();
    for pixel in output.pixels_mut() {
        pixel[0] = lut[pixel[0] as usize];
    }
    output
}
