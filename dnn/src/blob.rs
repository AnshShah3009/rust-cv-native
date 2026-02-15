use image::GrayImage;

pub fn image_to_blob(image: &GrayImage) -> Vec<f32> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let mut blob = Vec::with_capacity(width * height);
    
    for pixel in image.as_raw() {
        blob.push(*pixel as f32 / 255.0);
    }
    
    blob
}

pub fn blob_to_image(blob: &[f32], width: u32, height: u32) -> GrayImage {
    let mut raw = Vec::with_capacity((width * height) as usize);
    for val in blob {
        raw.push((val * 255.0).clamp(0.0, 255.0) as u8);
    }
    GrayImage::from_raw(width, height, raw).unwrap_or_else(|| GrayImage::new(width, height))
}
