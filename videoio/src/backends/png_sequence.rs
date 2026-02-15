use crate::{Result, VideoWriter, VideoError};
use image::GrayImage;
use std::path::{Path, PathBuf};
use std::fs;

#[derive(Debug)]
pub struct PngSequenceWriter {
    directory: PathBuf,
    prefix: String,
    frame_count: usize,
}

impl PngSequenceWriter {
    pub fn new(directory: &Path, prefix: &str) -> Result<Self> {
        if !directory.exists() {
            fs::create_dir_all(directory).map_err(VideoError::Io)?;
        }
        
        Ok(Self {
            directory: directory.to_path_buf(),
            prefix: prefix.to_string(),
            frame_count: 0,
        })
    }
}

impl VideoWriter for PngSequenceWriter {
    fn write(&mut self, frame: &GrayImage) -> Result<()> {
        let filename = format!("{}_{:06}.png", self.prefix, self.frame_count);
        let path = self.directory.join(filename);
        
        frame.save(&path).map_err(|e| VideoError::Backend(format!("Failed to save frame: {}", e)))?;
        self.frame_count += 1;
        Ok(())
    }
}
