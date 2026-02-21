import cv_native
import numpy as np
import time
import sys

def main():
    print(f"Initializing SLAM with Latency priority...")
    try:
        # Initialize SLAM with WorkloadHint.Latency
        slam = cv_native.PySlam(cv_native.PyWorkloadHint.Latency)
    except Exception as e:
        print(f"Failed to initialize SLAM: {e}")
        return

    print("SLAM initialized successfully.")

    # Generate synthetic frames
    # Simulating a camera moving along X axis observing a static scene
    width, height = 640, 480
    num_frames = 10
    
    # Create a static background pattern (random points)
    np.random.seed(42)
    bg_points = np.random.randint(0, [width, height], (1000, 2))
    
    print(f"Processing {num_frames} frames...")
    
    for i in range(num_frames):
        # Create blank image
        frame = np.zeros((height, width), dtype=np.uint8)
        
        # Draw "stars" moving opposite to camera motion (camera moves right -> stars move left)
        shift_x = int(i * 5)
        
        for pt in bg_points:
            x, y = pt
            # Shift point
            curr_x = x - shift_x
            if 0 <= curr_x < width and 0 <= y < height:
                # Draw a small cross or dot
                frame[y, curr_x] = 255
                if curr_x + 1 < width: frame[y, curr_x+1] = 255
                if y + 1 < height: frame[y+1, curr_x] = 255

        # Process frame
        start_time = time.time()
        try:
            pose_flat, tracked_indices = slam.process_frame(frame)
            end_time = time.time()
            
            # Pose is a flat 4x4 matrix
            pose_matrix = np.array(pose_flat).reshape(4, 4)
            translation = pose_matrix[:3, 3]
            
            print(f"Frame {i}: Tracked {len(tracked_indices)} features | Pos: {translation} | Time: {(end_time - start_time)*1000:.2f}ms")
            
        except Exception as e:
            print(f"Frame {i} Error: {e}")

    print("Demo complete.")

if __name__ == "__main__":
    main()
