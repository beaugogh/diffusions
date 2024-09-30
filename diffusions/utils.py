import cv2
import os
import os.path as osp
from tqdm import tqdm


# extract frames from a video
def video_to_images(input_video_path, output_images_dir=None, fps=24):
    cap = cv2.VideoCapture(input_video_path)

    # frame_rate = 24  # Desired frame rate (1 frame every 0.5 seconds)
    frame_rate = fps
    frame_count = 0

    video_name = os.path.splitext(os.path.basename(input_video_path))[0].split(".")[0]
    video_dir = os.path.dirname(input_video_path)
    if not output_images_dir:
        output_images_dir = os.path.join(video_dir, f"{video_name}_frames")
    os.makedirs(output_images_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        frame_count += 1

        # Only extract frames at the desired frame rate
        if frame_count % int(cap.get(5) / frame_rate) == 0:
            output_file = f"{output_images_dir}/frame_{frame_count}.jpg"
            cv2.imwrite(output_file, frame)
            print(f"Frame {frame_count} has been extracted and saved as {output_file}")

    cap.release()
    cv2.destroyAllWindows()


# compose images into a video
def images_to_video(input_images_dir, output_video_path=None, fps=8):
    if not output_video_path:
        output_video_path = osp.join(input_images_dir, "output_vid.mp4")
    imgs = sorted(os.listdir(input_images_dir))
    first_img = cv2.imread(osp.join(input_images_dir, imgs[0]))
    h, w, c = first_img.shape
    size = (w, h)
    print("img size: ", size)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"DIVX"), fps, size)

    for filename in tqdm(imgs):
        if ".mp4" not in filename:
            img = cv2.imread(osp.join(input_images_dir, filename))
            out.write(img)

    out.release()


if __name__ == "__main__":
    # video_path = r"/home/bo/workspace/diffusions/assets/live_portrait/animations/ynw1-square-modified-out-squared--driving_beau2.mp4"
    # output_dir = "/home/bo/workspace/diffusions/assets/live_portrait/animations/ynw1-square-modified-out-squared_frames"
    # video_to_images(video_path, output_images_dir=output_dir, fps=24)

    images_dir = "/home/bo/workspace/diffusions/assets/temp"
    output_path = "/home/bo/workspace/diffusions/assets/ynw.mp4"
    images_to_video(images_dir, output_path, fps=29)
