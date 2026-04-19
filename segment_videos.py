import os
from moviepy import VideoFileClip

# SCRIPT 1 FOR COLLECTED DATASET
# Description: This script segments raw 5-minute videos into 10-second clips for further analysis.

# ==============
# CONFIGURATION
# ==============
INPUT_FOLDER = "input_videos"        # Folder containing your 5-min raw videos
OUTPUT_FOLDER = "output_segments_1seccont"    # Folder to save 3s clips
SEGMENT_DURATION = 1                # seconds per segment (formerly 10 seconds)

# Create output folder if it doesn’t exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# List all video files in the input folder
video_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".mp4", ".avi", ".mov"))]

print(f"Found {len(video_files)} video(s) to segment...")

# Sequential student naming (student01, student02, student03, ...)
for idx, filename in enumerate(video_files, start=1):
    input_path = os.path.join(INPUT_FOLDER, filename)
    base_name = f"student{idx+14:02d}"  # student

    # Load the video
    video = VideoFileClip(input_path)
    video_duration = video.duration  # in seconds

    # Compute number of segments
    num_segments = int(video_duration // SEGMENT_DURATION)
    print(f"Processing {filename} → {num_segments} segments as {base_name}_segment_xx.mp4")

    # Create subclips
    for i in range(num_segments):
        start_time = i * SEGMENT_DURATION
        end_time = start_time + SEGMENT_DURATION
        output_name = f"{base_name}_segment_{i+1:02d}.mp4"
        output_path = os.path.join(OUTPUT_FOLDER, output_name)

        # Cut and save each 10s segment
        segment = video.subclipped(start_time, end_time)  
        segment.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac"
        )

    video.close()

print("\nAll videos segmented successfully!")
print(f"Output saved to: {os.path.abspath(OUTPUT_FOLDER)}")
