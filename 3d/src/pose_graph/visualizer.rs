//! Pose Graph Visualizer
//!
//! Provides visualization utilities for pose graph optimization,
//! including graph structure visualization and trajectory plotting.

use nalgebra::{Point3, Vector3, Vector4};
use std::collections::HashMap;

/// Node in the pose graph visualization
#[derive(Debug, Clone)]
pub struct PoseNode {
    pub id: usize,
    pub position: Point3<f64>,
    pub rotation: Vector4<f64>,
    pub is_fixed: bool,
    pub color: [u8; 3],
}

/// Edge between pose nodes
#[derive(Debug, Clone)]
pub struct PoseEdge {
    pub from: usize,
    pub to: usize,
    pub color: [u8; 3],
    pub weight: f64,
}

/// Visualizer for pose graph optimization results
pub struct PoseGraphVisualizer {
    pub nodes: Vec<PoseNode>,
    pub edges: Vec<PoseEdge>,
    pub trajectory: Vec<Point3<f64>>,
    pub loop_closures: Vec<(usize, usize)>,
}

impl PoseGraphVisualizer {
    /// Create a new visualizer
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            trajectory: Vec::new(),
            loop_closures: Vec::new(),
        }
    }

    /// Add a pose node
    pub fn add_node(
        &mut self,
        id: usize,
        position: Point3<f64>,
        rotation: Vector4<f64>,
        is_fixed: bool,
    ) {
        let color = if is_fixed {
            [255, 0, 0] // Red for fixed nodes
        } else {
            [0, 0, 255] // Blue for variable nodes
        };

        self.nodes.push(PoseNode {
            id,
            position,
            rotation,
            is_fixed,
            color,
        });
        self.trajectory.push(position);
    }

    /// Add an edge between nodes
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        self.edges.push(PoseEdge {
            from,
            to,
            color: [100, 100, 100],
            weight,
        });
    }

    /// Mark two nodes as loop closure
    pub fn add_loop_closure(&mut self, from: usize, to: usize) {
        self.loop_closures.push((from, to));
        // Find and update the edge color
        for edge in &mut self.edges {
            if (edge.from == from && edge.to == to) || (edge.from == to && edge.to == from) {
                edge.color = [0, 255, 0]; // Green for loop closures
            }
        }
    }

    /// Get trajectory as a list of points
    pub fn get_trajectory(&self) -> &[Point3<f64>] {
        &self.trajectory
    }

    /// Export trajectory to PLY format for visualization
    pub fn export_trajectory_ply(&self) -> String {
        let mut output = String::new();

        output.push_str("ply\n");
        output.push_str("format ascii 1.0\n");
        output.push_str(&format!("element vertex {}\n", self.trajectory.len()));
        output.push_str("property float x\n");
        output.push_str("property float y\n");
        output.push_str("property float z\n");
        output.push_str("property uchar red\n");
        output.push_str("property uchar green\n");
        output.push_str("property uchar blue\n");
        output.push_str("end_header\n");

        for (i, point) in self.trajectory.iter().enumerate() {
            // Color gradient from blue to red
            let t = i as f64 / self.trajectory.len().max(1) as f64;
            let r = (t * 255.0) as u8;
            let b = ((1.0 - t) * 255.0) as u8;

            output.push_str(&format!(
                "{} {} {} {} 0 {}\n",
                point.x, point.y, point.z, r, b
            ));
        }

        output
    }

    /// Export full graph to PLY format
    pub fn export_graph_ply(&self) -> String {
        let mut output = String::new();

        // Count total vertices (nodes + edge endpoints)
        let n_vertices = self.nodes.len() + self.edges.len() * 2;

        output.push_str("ply\n");
        output.push_str("format ascii 1.0\n");
        output.push_str(&format!("element vertex {}\n", n_vertices));
        output.push_str("property float x\n");
        output.push_str("property float y\n");
        output.push_str("property float z\n");
        output.push_str("property uchar red\n");
        output.push_str("property uchar green\n");
        output.push_str("property uchar blue\n");
        output.push_str(&format!("element edge {}\n", self.edges.len()));
        output.push_str("property int vertex1\n");
        output.push_str("property int vertex2\n");
        output.push_str("property uchar red\n");
        output.push_str("property uchar green\n");
        output.push_str("property uchar blue\n");
        output.push_str("end_header\n");

        // Write nodes
        for node in &self.nodes {
            output.push_str(&format!(
                "{} {} {} {} {} {}\n",
                node.position.x,
                node.position.y,
                node.position.z,
                node.color[0],
                node.color[1],
                node.color[2]
            ));
        }

        // Write edge endpoints (duplicated for coloring)
        let node_count = self.nodes.len();
        for edge in &self.edges {
            if let (Some(from_node), Some(to_node)) = (
                self.nodes.iter().find(|n| n.id == edge.from),
                self.nodes.iter().find(|n| n.id == edge.to),
            ) {
                output.push_str(&format!(
                    "{} {} {} {} {} {}\n",
                    from_node.position.x,
                    from_node.position.y,
                    from_node.position.z,
                    edge.color[0],
                    edge.color[1],
                    edge.color[2]
                ));
                output.push_str(&format!(
                    "{} {} {} {} {} {}\n",
                    to_node.position.x,
                    to_node.position.y,
                    to_node.position.z,
                    edge.color[0],
                    edge.color[1],
                    edge.color[2]
                ));
            }
        }

        // Write edges
        for (i, edge) in self.edges.iter().enumerate() {
            if let (Some(from_idx), Some(to_idx)) = (
                self.nodes.iter().position(|n| n.id == edge.from),
                self.nodes.iter().position(|n| n.id == edge.to),
            ) {
                let edge_vertex1 = node_count + i * 2;
                let edge_vertex2 = node_count + i * 2 + 1;
                output.push_str(&format!(
                    "{} {} {} {} {}\n",
                    from_idx, to_idx, edge.color[0], edge.color[1], edge.color[2]
                ));
            }
        }

        output
    }

    /// Compute trajectory length
    pub fn trajectory_length(&self) -> f64 {
        if self.trajectory.len() < 2 {
            return 0.0;
        }

        let mut length = 0.0;
        for i in 1..self.trajectory.len() {
            length += (self.trajectory[i] - self.trajectory[i - 1]).norm();
        }
        length
    }

    /// Get bounding box of the trajectory
    pub fn bounding_box(&self) -> (Point3<f64>, Point3<f64>) {
        if self.trajectory.is_empty() {
            return (Point3::origin(), Point3::origin());
        }

        let mut min = self.trajectory[0];
        let mut max = self.trajectory[0];

        for point in &self.trajectory {
            min.x = min.x.min(point.x);
            min.y = min.y.min(point.y);
            min.z = min.z.min(point.z);
            max.x = max.x.max(point.x);
            max.y = max.y.max(point.y);
            max.z = max.z.max(point.z);
        }

        (min, max)
    }

    /// Reset the visualizer
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.trajectory.clear();
        self.loop_closures.clear();
    }
}

impl Default for PoseGraphVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Visualization statistics
#[derive(Debug, Clone)]
pub struct PoseGraphStats {
    pub num_poses: usize,
    pub num_edges: usize,
    pub num_loop_closures: usize,
    pub trajectory_length: f64,
    pub mean_edge_error: f64,
}

impl PoseGraphStats {
    /// Compute stats from a visualizer
    pub fn from_visualizer(visualizer: &PoseGraphVisualizer) -> Self {
        Self {
            num_poses: visualizer.nodes.len(),
            num_edges: visualizer.edges.len(),
            num_loop_closures: visualizer.loop_closures.len(),
            trajectory_length: visualizer.trajectory_length(),
            mean_edge_error: 0.0, // Would need actual error computation
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visualizer_new() {
        let viz = PoseGraphVisualizer::new();
        assert!(viz.nodes.is_empty());
        assert!(viz.edges.is_empty());
    }

    #[test]
    fn test_add_node() {
        let mut viz = PoseGraphVisualizer::new();
        viz.add_node(
            0,
            Point3::new(1.0, 2.0, 3.0),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            false,
        );
        assert_eq!(viz.nodes.len(), 1);
        assert_eq!(viz.trajectory.len(), 1);
    }

    #[test]
    fn test_add_edge() {
        let mut viz = PoseGraphVisualizer::new();
        viz.add_node(
            0,
            Point3::new(0.0, 0.0, 0.0),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            false,
        );
        viz.add_node(
            1,
            Point3::new(1.0, 0.0, 0.0),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            false,
        );
        viz.add_edge(0, 1, 1.0);
        assert_eq!(viz.edges.len(), 1);
    }

    #[test]
    fn test_trajectory_length() {
        let mut viz = PoseGraphVisualizer::new();
        viz.add_node(
            0,
            Point3::new(0.0, 0.0, 0.0),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            false,
        );
        viz.add_node(
            1,
            Point3::new(1.0, 0.0, 0.0),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            false,
        );
        viz.add_node(
            2,
            Point3::new(1.0, 1.0, 0.0),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            false,
        );

        let length = viz.trajectory_length();
        assert!((length - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_export_trajectory_ply() {
        let mut viz = PoseGraphVisualizer::new();
        viz.add_node(
            0,
            Point3::new(0.0, 0.0, 0.0),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            false,
        );

        let ply = viz.export_trajectory_ply();
        assert!(ply.contains("ply"));
        assert!(ply.contains("element vertex 1"));
    }

    #[test]
    fn test_bounding_box() {
        let mut viz = PoseGraphVisualizer::new();
        viz.add_node(
            0,
            Point3::new(0.0, 0.0, 0.0),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            false,
        );
        viz.add_node(
            1,
            Point3::new(1.0, 2.0, 3.0),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            false,
        );

        let (min, max) = viz.bounding_box();
        assert_eq!(min.x, 0.0);
        assert_eq!(max.x, 1.0);
    }
}
