use rerun::RecordingStream;
use cv_core::point_cloud::PointCloud;
// use cv_core::image::Image; // Future

/// A wrapper for Rerun logging.
pub struct RerunLogger {
    rec: RecordingStream,
}

impl RerunLogger {
    /// Connect to a remote Rerun viewer or spawn one.
    pub fn new(application_id: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let (rec, _storage) = rerun::RecordingStreamBuilder::new(application_id).memory()?;
        // For now, memory recording or save to file?
        // Usually user wants spawn().
        // rerun::native_viewer::spawn(rec.clone())?; // Requires "native_viewer" feature
        
        // Let's default to a simple setup.
        Ok(Self { rec })
    }
    
    pub fn spawn(application_id: &str) -> Result<Self, Box<dyn std::error::Error>> {
         let rec = rerun::RecordingStreamBuilder::new(application_id).spawn()?;
         Ok(Self { rec })
    }

    /// Log a point cloud.
    pub fn log_point_cloud(&self, entity_path: &str, pc: &PointCloud) -> Result<(), Box<dyn std::error::Error>> {
        // Collect points
        let points: Vec<[f32; 3]> = pc.points.iter().map(|p| [p.x, p.y, p.z]).collect();
        let mut archetype = rerun::Points3D::new(points.clone());

        // Handle Colors
        if let Some(colors) = &pc.colors {
            // Rerun Color accepts [u8; 4] or u32. 
            // We'll use [u8; 4] (RGBA) for simplicity.
            let colors_rerun: Vec<[u8; 4]> = colors.iter()
                .map(|c| [
                    (c.x * 255.0) as u8, 
                    (c.y * 255.0) as u8, 
                    (c.z * 255.0) as u8, 
                    255
                ])
                .collect();
            archetype = archetype.with_colors(colors_rerun);
        }

        self.rec.log(entity_path, &archetype)?;

        // Handle Normals separately using Arrows3D
        if let Some(normals) = &pc.normals {
            let normals_vec: Vec<[f32; 3]> = normals.iter().map(|n| [n.x, n.y, n.z]).collect();
            let arrows = rerun::Arrows3D::from_vectors(normals_vec).with_origins(points);
            self.rec.log(format!("{}/normals", entity_path), &arrows)?;
        }

        Ok(())
    }
}
