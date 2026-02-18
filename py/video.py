import os
import shutil
import uuid
import subprocess
import folder_paths # type: ignore
from comfy.comfy_types import IO, ComfyNodeABC
from comfy_api.input_impl import VideoFromFile


class Soze_AppendToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_video": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "append_video": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "save_filename_path": ("STRING", {
                    "default": "",
                    "multiline": False
                })
            },
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "append_video"
    CATEGORY = "Soze Nodes"
    OUTPUT_NODE = True

    def append_video(self, base_video, append_video, save_filename_path):
        base_exists = base_video and os.path.exists(base_video)
        append_exists = append_video and os.path.exists(append_video)

        if not base_exists and not append_exists:
            return ("",)  # Nothing to do

        # Determine output filename
        filename = ""
        if save_filename_path:
            filename = save_filename_path.split(os.sep)[-1]
        elif base_exists:
            filename = os.path.basename(base_video)
        elif append_exists:
            filename = os.path.basename(append_video)
            
        output_file = os.path.join(folder_paths.get_output_directory(), save_filename_path if save_filename_path else filename)
        
        # Temp file for processing
        temp_dir = folder_paths.get_temp_directory()
        temp_filename = f"temp_{uuid.uuid4().hex}_{filename}"
        if not temp_filename.endswith(".mp4"):
            temp_filename += ".mp4"
        temp_output_file = os.path.join(temp_dir, temp_filename)
        
        if base_exists and append_exists:
            print(f"Appending video... Base: {base_video}, Append: {append_video}")
            
            # Use ffmpeg with scale2ref to ensure resolution compatibility.
            # We explicitly map [v] and [a] to output.
            # This implementation assumes both videos have audio streams.
            command = [
                "ffmpeg", "-y",
                "-i", base_video,
                "-i", append_video,
                "-filter_complex", "[1:v][0:v]scale2ref[v1][v0];[v0][0:a][v1][1:a]concat=n=2:v=1:a=1[v][a]",
                "-map", "[v]",
                "-map", "[a]",
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "slow",
                "-c:a", "aac",
                "-b:a", "192k",
                temp_output_file
            ]
            
            try:
                subprocess.run(command, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.decode() if e.stderr else str(e)
                print(f"FFmpeg failed: {error_msg}")
                raise RuntimeError(f"FFmpeg failed to append videos: {error_msg}")

        elif base_exists:
             shutil.copy2(base_video, temp_output_file)
        elif append_exists:
             shutil.copy2(append_video, temp_output_file)

        # Move to final location
        if os.path.exists(output_file):
             try:
                os.remove(output_file) # Ensure clean overwrite
             except Exception as e:
                output_file_new = output_file.replace(".mp4", f"_new_{uuid.uuid4().hex}.mp4")
                print(f"Warning: Could not remove existing file, saving to {output_file_new}: {e}")
                output_file = output_file_new
        
        shutil.move(temp_output_file, output_file)
        
        print(f"Video merging completed! Output: {output_file}")
        return (output_file,)
    
    
    
class Soze_LoadVideosFromFolder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Input_Folder": ("STRING", {"default": ""}),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000000, "control_after_generate": True, "step": 1}),
            }
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING", "STRING", "BOOL")
    RETURN_NAMES = ("Video", "Input_Path", "Video_Filename_Path", "Video_Filename", "Video_Filename_No_Ext", "Video_Changed")
    FUNCTION = "load_videos_from_folder"

    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Return a value that changes each time to force re-execution
        return float("NaN")

    def load_videos_from_folder(self, Input_Folder, index):
        if not os.path.isdir(Input_Folder):
            raise FileNotFoundError(f"Folder not found: {Input_Folder}")
        dir_files = os.listdir(Input_Folder)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"Folder only has {len(dir_files)} files in it: {Input_Folder}")

        # Filter files by extension
        valid_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.mpeg', '.mpg', '.flv', '.mp3', '.wav', '.aac']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(Input_Folder, x) for x in dir_files]

        # Check if index is valid
        if index >= len(dir_files):
            raise IndexError(f"Index {index} is out of range. Only {len(dir_files)} videos found.")

        # Load only the single video at the specified index
        video_path = dir_files[index]
        
        if os.path.isdir(video_path):
            raise ValueError(f"Path at index {index} is a directory, not a video file.")

        input_filepath = folder_paths.get_annotated_filepath(video_path)
        input_filename = os.path.basename(input_filepath)
        input_filename_no_ext = os.path.splitext(input_filename)[0]
        video = VideoFromFile(input_filepath)
        
        video_changed = True        
        return video, Input_Folder, video_path, input_filename, input_filename_no_ext, video_changed
    


