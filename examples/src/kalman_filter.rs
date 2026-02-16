//! Example: Kalman Filter for 2D Tracking
//!
//! Run with: cargo run --example kalman_filter

use cv_slam::kalman::{utils, KalmanFilter, KalmanFilterState};
use nalgebra::SVector;

fn main() {
    println!("=== Kalman Filter 2D Tracking Example ===\n");

    // Create a constant velocity model for 2D tracking
    // State: [x, y, vx, vy]
    // Measurement: [x, y]
    let kf = utils::constant_velocity_2d(
        dt = 0.1,                // time step
        process_noise = 0.01,    // process noise
        measurement_noise = 0.1, // measurement noise
    );

    // Initialize state
    let mut state = KalmanFilterState::zero();

    println!("Initial state: {:?}", state.x);
    println!("Initial covariance:\n{}", state.p);

    // Simulate tracking measurements
    let measurements = vec![
        (0.1, 0.1),
        (0.25, 0.15),
        (0.45, 0.22),
        (0.70, 0.35),
        (1.0, 0.5),
    ];

    let u = SVector::<f64, 4>::zeros(); // No control input

    println!("\n--- Tracking Progress ---");
    for (i, (mx, my)) in measurements.iter().enumerate() {
        let z = nalgebra::Vector2::new(*mx, *my);

        kf.step(&mut state, &u, &z);

        println!("\nStep {}: measurement = ({}, {})", i + 1, mx, my);
        println!(
            "  Estimated state: [{:.3}, {:.3}, {:.3}, {:.3}]",
            state.x[0], state.x[1], state.x[2], state.x[3]
        );
        println!(
            "  Std dev: [{:.3}, {:.3}, {:.3}, {:.3}]",
            state.std_dev()[0],
            state.std_dev()[1],
            state.std_dev()[2],
            state.std_dev()[3]
        );
    }

    // Final position estimate
    println!("\n=== Final Results ===");
    println!("Estimated position: ({:.3}, {:.3})", state.x[0], state.x[1]);
    println!("Estimated velocity: ({:.3}, {:.3})", state.x[2], state.x[3]);
    println!("Position uncertainty (1σ): ±{:.3}", state.std_dev()[0]);

    println!("\n=== Example Complete ===");
}
