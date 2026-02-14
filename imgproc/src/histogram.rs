use image::GrayImage;

pub fn compute_histogram(image: &GrayImage) -> [u32; 256] {
    let mut hist = [0u32; 256];
    for pixel in image.pixels() {
        hist[pixel[0] as usize] += 1;
    }
    hist
}

pub fn compute_histogram_normalized(image: &GrayImage) -> [f32; 256] {
    let hist = compute_histogram(image);
    let total = image.width() * image.height();
    hist.map(|h| h as f32 / total as f32)
}

pub fn compute_cdf(hist: &[u32; 256]) -> [u32; 256] {
    let mut cdf = [0u32; 256];
    cdf[0] = hist[0];
    for i in 1..256 {
        cdf[i] = cdf[i - 1] + hist[i];
    }
    cdf
}

pub fn histogram_equalization(image: &GrayImage) -> GrayImage {
    let hist = compute_histogram(image);
    let cdf = compute_cdf(&hist);

    let cdf_min = cdf.iter().find(|&&x| x > 0).copied().unwrap_or(0);
    let total = image.width() * image.height() as u32;

    let mut lut = [0u8; 256];
    for i in 0..256 {
        let val = ((cdf[i] - cdf_min) as f32 / (total - cdf_min) as f32 * 255.0).round() as u8;
        lut[i] = val;
    }

    let mut output = GrayImage::new(image.width(), image.height());

    for (i, pixel) in output.pixels_mut().enumerate() {
        let src_pixel = image.as_raw()[i];
        pixel[0] = lut[src_pixel as usize];
    }

    output
}
