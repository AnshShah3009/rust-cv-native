//! 3D File I/O Module
//!
//! Supports reading and writing point clouds and meshes in various formats:
//! - PLY (Polygon File Format)
//! - OBJ (Wavefront Object)
//! - STL (STereoLithography)
//! - PCD (Point Cloud Data - PCL format)

pub mod mesh;
pub mod obj;
pub mod pcd;
pub mod ply;
pub mod stl;

pub use obj::{read_obj, write_obj, ObjMesh};
pub use pcd::{read_pcd, write_pcd, write_pcd_binary, PcdData};
pub use ply::{read_ply, write_ply};
pub use stl::{read_stl, write_stl, write_stl_ascii, write_stl_binary};
// TriangleMesh is now the authoritative definition from cv-3d, re-exported here for convenience
pub use mesh::TriangleMesh;

pub use cv_core::{Error, Result};
