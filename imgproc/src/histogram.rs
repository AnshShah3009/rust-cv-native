use cv_core::GrayImage;

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
    let cdf = compute_cdf(hist);

    let cdf_min = cdf.iter().find(|&&x| x > 0).copied().unwrap_or(0);
    let total = image.width() * image.height();

    let mut lut = [0u8; 256];
    for i in 0..256 {
        let val = ((cdf[i] - cdf_min) as f32 / (total - cdf_min) * 255.0).round() as u8;
        lut[i] = val;
    }

    let mut output = GrayImage::new(image.width(), image.height());

    for (i, pixel) in output.pixels_mut().enumerate() {
        let src_pixel = image.as_raw()[i];
        pixel[0] = lut[src_pixel as usize];
    }

    output
}

pub fn clahe(image: &GrayImage, clip_limit: f32, tile_size: u32) -> GrayImage {
    histogram_equalization(image)
}

pub fn compute_2d_histogram(img1: &GrayImage, img2: &GrayImage, bins: u32) -> Vec<Vec<u32>> {
    let mut hist = vec![vec![0u32; bins as usize]; bins as usize];

    let factor = (bins as f32) / 256.0;

    for (p1, p2) in img1.pixels().zip(img2.pixels()) {
        let b1 = ((p1[0] as f32) * factor) as usize;
        let b2 = ((p2[0] as f32) * factor) as usize;

        let b1 = b1.min(bins as usize - 1);
        let b2 = b2.min(bins as usize - 1);

        hist[b1][b2] += 1;
    }

    hist
}

pub fn back_projection(image: &GrayImage, sample: &GrayImage, bins: u32) -> GrayImage {
    let sample_hist = compute_histogram_normalized(sample);
    let cdf = compute_cdf(sample_hist.map(|h| (h * 10000.0) as u32));

    let factor = (bins as f32) / 256.0;
    let mut lut = [0u8; 256];

    let cdf_min = cdf.iter().find(|&&x| x > 0).copied().unwrap_or(0);
    let total = sample.width() * sample.height();

    for i in 0..256 {
        let val = ((cdf[i] - cdf_min) as f32 / (total - cdf_min) * 255.0).round() as u8;
        lut[i] = val;
    }

    let mut output = GrayImage::new(image.width(), image.height());

    for (i, pixel) in output.pixels_mut().enumerate() {
        let src_pixel = image.as_raw()[i];
        pixel[0] = lut[src_pixel as usize];
    }

    output
}
