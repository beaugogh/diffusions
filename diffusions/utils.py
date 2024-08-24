import cv2
import os


# Function to extract frames from a video until reaching the desired frame count
def extract_video_frames(video_file):
    cap = cv2.VideoCapture(video_file)

    frame_rate = 24  # Desired frame rate (1 frame every 0.5 seconds)
    frame_count = 0

    video_name = os.path.splitext(os.path.basename(video_file))[0].split(".")[0]
    video_dir = os.path.dirname(video_file)
    output_dir = os.path.join(video_dir, f"{video_name}_frames")
    os.makedirs(output_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        frame_count += 1

        # Only extract frames at the desired frame rate
        if frame_count % int(cap.get(5) / frame_rate) == 0:
            output_file = f"{output_dir}/frame_{frame_count}.jpg"
            cv2.imwrite(output_file, frame)
            print(f"Frame {frame_count} has been extracted and saved as {output_file}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = r"/home/bo/workspace/diffusions/videos/54JuJutsuTechniques.mp4" 
    extract_video_frames(video_path)
