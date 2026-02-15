//! 3D File I/O Module
//! 
//! Supports reading and writing point clouds and meshes in various formats:
//! - PLY (Polygon File Format)
//! - OBJ (Wavefront Object)
//! - STL (STereoLithography)
//! - PCD (Point Cloud Data - PCL format)

pub mod ply;
pub mod obj;
pub mod stl;
pub mod pcd;
pub mod mesh;

pub use ply::{read_ply, write_ply};
pub use obj::{read_obj, write_obj, ObjMesh};
pub use stl::{read_stl, write_stl, write_stl_ascii, write_stl_binary};
pub use pcd::{read_pcd, write_pcd, PcdData};
pub use mesh::TriangleMesh;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum IoError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
    #[error("Invalid data: {0}")]
    InvalidData(String),
}

pub type Result<T> = std::result::Result<T, IoError>;
