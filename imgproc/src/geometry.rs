use image::GrayImage;
use nalgebra::{Matrix3, Point2};

pub fn get_pixel_bilinear(img: &GrayImage, x: f32, y: f32) -> f32 {
    let width = img.width() as f32;
    let height = img.height() as f32;

    if x < 0.0 || x >= width - 1.0 || y < 0.0 || y >= height - 1.0 {
        return 0.0;
    }

    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let v00 = img.get_pixel(x0, y0)[0] as f32;
    let v10 = img.get_pixel(x1, y0)[0] as f32;
    let v01 = img.get_pixel(x0, y1)[0] as f32;
    let v11 = img.get_pixel(x1, y1)[0] as f32;

    let v0 = v00 * (1.0 - fx) + v10 * fx;
    let v1 = v01 * (1.0 - fx) + v11 * fx;

    v0 * (1.0 - fy) + v1 * fy
}

pub fn warp_perspective(
    src: &GrayImage,
    matrix: &Matrix3<f32>,
    width: u32,
    height: u32,
) -> GrayImage {
    let mut dst = GrayImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let pt = Point2::new(x as f32, y as f32);
            let src_pt = transform_point(matrix, &pt);

            if src_pt.x >= 0.0
                && src_pt.x < src.width() as f32
                && src_pt.y >= 0.0
                && src_pt.y < src.height() as f32
            {
                let val = get_pixel_bilinear(src, src_pt.x, src_pt.y);
                dst.put_pixel(x, y, image::Luma([val as u8]));
            }
        }
    }

    dst
}

fn transform_point(matrix: &Matrix3<f32>, pt: &Point2<f32>) -> Point2<f32> {
    let x = pt.x;
    let y = pt.y;

    let w = matrix[(2, 0)] * x + matrix[(2, 1)] * y + matrix[(2, 2)];

    if w.abs() > 1e-10 {
        Point2::new(
            (matrix[(0, 0)] * x + matrix[(0, 1)] * y + matrix[(0, 2)]) / w,
            (matrix[(1, 0)] * x + matrix[(1, 1)] * y + matrix[(1, 2)]) / w,
        )
    } else {
        Point2::new(
            matrix[(0, 0)] * x + matrix[(0, 1)] * y + matrix[(0, 2)],
            matrix[(1, 0)] * x + matrix[(1, 1)] * y + matrix[(1, 2)],
        )
    }
}

pub fn get_rotation_matrix(center: Point2<f32>, angle: f32, scale: f32) -> Matrix3<f32> {
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    let tx = center.x * (1.0 - scale * cos_a) + center.y * scale * sin_a;
    let ty = center.y * (1.0 - scale * cos_a) - center.x * scale * sin_a;

    Matrix3::new(
        scale * cos_a,
        scale * sin_a,
        tx,
        -scale * sin_a,
        scale * cos_a,
        ty,
        0.0,
        0.0,
        1.0,
    )
}

pub fn get_translation_matrix(dx: f32, dy: f32) -> Matrix3<f32> {
    Matrix3::new(1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0)
}

pub fn get_scaling_matrix(sx: f32, sy: f32) -> Matrix3<f32> {
    Matrix3::new(sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 1.0)
}
