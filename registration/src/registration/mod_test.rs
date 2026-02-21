
#[cfg(test)]
mod tests {
    use crate::registration::registration_icp_point_to_plane_ctx;
    use cv_core::point_cloud::PointCloud;
    use nalgebra::{Matrix4, Point3, Vector3};
    use cv_hal::cpu::CpuBackend;
    use cv_hal::compute::ComputeDevice;

    #[test]
    fn test_icp_acceleration_parity() {
        let cpu = CpuBackend::new().unwrap();
        let device = ComputeDevice::Cpu(&cpu);
        
        let mut source = PointCloud::new(Vec::new());
        let mut target = PointCloud::new(Vec::new());
        
        // Create a simple plane with normals
        for y in 0..10 {
            for x in 0..10 {
                let p = Point3::new(x as f32 * 0.1, y as f32 * 0.1, 0.0);
                source.points.push(p);
                // Offset target slightly
                target.points.push(p + Vector3::new(0.02, 0.01, 0.0));
            }
        }
        target.normals = Some(vec![Vector3::new(0.0, 0.0, 1.0); 100]);
        
        let init = Matrix4::identity();
        let res_ctx = registration_icp_point_to_plane_ctx(&source, &target, 0.2, &init, 20, &device);
        
        if let Some(res) = res_ctx {
            println!("ICP Result: fitness={}, rmse={}, iterations={}", res.fitness, res.inlier_rmse, res.num_iterations);
            assert!(res.fitness > 0.9);
            assert!(res.inlier_rmse < 0.05);
        } else {
            panic!("ICP failed to return a result");
        }
    }
}
