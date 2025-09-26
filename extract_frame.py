import cv2
import os

def extract_frame(video_path, output_path, frame_number=100):
    """Extract a specific frame from video"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read frame
    ret, frame = cap.read()
    
    if ret:
        cv2.imwrite(output_path, frame)
        print(f"Frame {frame_number} extracted and saved to {output_path}")
        print(f"Frame size: {frame.shape}")
        return True
    else:
        print(f"Error: Could not read frame {frame_number}")
        return False
    
    cap.release()

if __name__ == "__main__":
    video_path = "example.mp4"
    output_path = "test_frame.jpg"
    
    if os.path.exists(video_path):
        extract_frame(video_path, output_path)
    else:
        print(f"Video file {video_path} not found")