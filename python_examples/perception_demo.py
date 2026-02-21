import cv_native
import numpy as np
import time
import os
import sys

def main():
    print("Initializing Unified Perception Pipeline...")
    
    # 1. Initialize Runtime with Latency priority
    try:
        slam = cv_native.PySlam(cv_native.PyWorkloadHint.Latency)
        print("SLAM initialized.")
    except Exception as e:
        print(f"SLAM initialization failed: {e}")
        return

    # 2. Initialize Video Capture
    # In this environment, we expect a directory of images or a fallback
    video_path = "test_data/images" # Default to a directory
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    print(f"Opening video/image sequence: {video_path}")
    try:
        cap = cv_native.PyVideoCapture(video_path)
    except Exception as e:
        print(f"Failed to open video: {e}")
        # Create dummy capture simulation if file not found
        cap = None

    # 3. Initialize DNN (Optional)
    dnn = None
    model_path = "python_examples/models/mobilenetv2.onnx"
    if os.path.exists(model_path):
        print(f"Loading DNN model: {model_path}")
        try:
            dnn = cv_native.PyDnnNet(model_path, cv_native.PyWorkloadHint.Throughput)
            print("DNN model loaded.")
        except Exception as e:
            print(f"Failed to load DNN: {e}")
    else:
        print(f"DNN model not found at {model_path}, skipping inference.")

    # 4. Processing Loop
    frame_count = 0
    while True:
        frame = None
        if cap:
            frame_bytes = cap.read()
            if frame_bytes is None:
                break
            # Reshape flat bytes to image? 
            # PyVideoCapture.read returns Option<Vec<u8>>, but doesn't return shape.
            # The current binding implementation returns flat bytes of the GrayImage.
            # We need width/height to reconstruct.
            # Limitation: PyVideoCapture needs to return shape or numpy array.
            # For now, let's simulate frame generation if capture fails or is dummy.
            pass
        
        # Simulation for Demo if no real video
        if frame is None:
            if frame_count > 10: break
            width, height = 640, 480
            frame = np.zeros((height, width), dtype=np.uint8)
            # Add synthetic motion
            cv_native_dummy_draw(frame, frame_count)
        
        start_time = time.time()
        
        # A. Run SLAM Tracking
        pose_flat, tracked_feats = slam.process_frame(frame)
        
        # B. Run DNN Inference (if available)
        detections = []
        if dnn:
            try:
                # Dnn expects (C, H, W) or (H, W)?
                # PyDnnNet.forward expects PyReadonlyArray2<u8> (H, W) -> Grayscale
                # Real models need RGB usually.
                # Our Dnn wrapper converts Luma8 -> Float Tensor.
                # This is a simplification for the demo.
                outputs = dnn.forward(frame)
                # Parse outputs (model dependent)
                detections = outputs
            except Exception as e:
                print(f"Inference failed: {e}")

        dt = time.time() - start_time
        
        # Output status
        pose_mat = np.array(pose_flat).reshape(4, 4)
        trans = pose_mat[:3, 3]
        print(f"Frame {frame_count}: Tracked {len(tracked_feats)} | Pos: {trans} | FPS: {1.0/dt:.1f}")
        
        frame_count += 1

    print("Demo complete.")

def cv_native_dummy_draw(frame, i):
    # Draw moving dots
    h, w = frame.shape
    for j in range(50):
        x = (j * 30 + i * 2) % w
        y = (j * 50 + i) % h
        frame[y, x] = 255
        if x+1 < w: frame[y, x+1] = 255
        if y+1 < h: frame[y+1, x] = 255

if __name__ == "__main__":
    main()
