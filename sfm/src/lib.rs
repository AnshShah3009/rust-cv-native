//! Structure from Motion module
//!
//! This crate provides algorithms for 3D reconstruction from
//! multiple 2D images.

pub mod bundle_adjustment;
pub mod triangulation;

pub use triangulation::*;

pub trait BundleAdjustment {
    fn optimize(&mut self);
}
