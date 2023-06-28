from moviepy.editor import *

# Set video duration in seconds
duration = 1

# Set video dimensions
width = 640
height = 480

# Set label properties
text = "Please upload a video."
fontsize = 30
fontcolor = 'white'
background = 'black'

# Create a TextClip with the label
label = TextClip(text, fontsize=fontsize, color=fontcolor, bg_color=background)

# Set the label's position to be centered
label = label.set_position(('center', 'center'))

# Create a black video clip with the specified dimensions and duration
background_clip = ColorClip((width, height), col=[0, 0, 0]).set_duration(duration)

# Overlay the label on the background clip
final_clip = CompositeVideoClip([background_clip, label])

# Set the final clip's duration
final_clip = final_clip.set_duration(duration)

# Set the final clip's FPS
final_clip = final_clip.set_fps(30)

# Save the final clip as an MP4 video file
final_clip.write_videofile("upload_message.mp4", codec='libx264')
