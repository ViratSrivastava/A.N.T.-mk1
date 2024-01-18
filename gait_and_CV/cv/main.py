import cv2 # Import OpenCV
import numpy as np # Import NumPy
import tkinter as tk # Import Tkinter
from PIL import Image, ImageTk, ImageDraw  # Import ImageDraw
from datetime import datetime # Import datetime

def main():
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    window = tk.Tk()

    video_label = tk.Label(window)
    color_space_label = tk.Label(window, text="RGB")

    def update_video_feed():
        ret, frame = cap.read()

        current_color_space = color_space_label["text"]

        if current_color_space == "Grayscale" and frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif current_color_space == "RGB" and frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Convert the frame to PIL format
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        # Display the processed frame with resolution, FPS, time information
        draw = ImageDraw.Draw(image)
        resolution_text = f"Resolution: {width}x{height}"
        fps_text = f"FPS: {fps}"
        time_text = f"Time: {datetime.now().strftime('%H:%M:%S')}"  # Current time
    
        
        draw.text((10, 20), resolution_text, font=None, fill=(0, 255, 0))
        draw.text((10, 40), fps_text, font=None, fill=(0, 255, 0))
        draw.text((10, 60), time_text, font=None, fill=(0, 255, 0))

        video_label.imgtk = ImageTk.PhotoImage(image=image)
        video_label.configure(image=video_label.imgtk)

        window.after(10, update_video_feed)

    def toggle_color_space():
        """
        Toggles the color space between RGB and Grayscale.
        """
        # Toggle the color space label text
        if color_space_label["text"] == "RGB":
            color_space_label["text"] = "Grayscale"
        else:
            color_space_label["text"] = "RGB"

    def toggle_camera():
        """
        Toggles between the regular camera and the infrared camera.
        """
        # Toggle the camera label text
        if toggle_button["text"] == "Switch to Infrared":
            toggle_button["text"] = "Switch to Regular"
            cap.set(cv2.CAP_PROP_CONVERT_RGB, False)  # Enable infrared camera
        else:
            toggle_button["text"] = "Switch to Infrared"
            cap.set(cv2.CAP_PROP_CONVERT_RGB, True)  # Enable regular camera

    # Create a button to toggle the color space
    toggle_button = tk.Button(window, text="Toggle Color Space", command=toggle_color_space)
    # Create a button to toggle the camera
    camera_button = tk.Button(window, text="Switch to Infrared", command=toggle_camera)
    # Create a button to stop execution
    stop_button = tk.Button(window, text="Stop Execution", command=window.quit)

    # Pack the labels, buttons, and video feed into the window
    color_space_label.pack()
    toggle_button.pack()
    camera_button.pack()
    stop_button.pack()
    video_label.pack()

    # Start updating the video feed
    update_video_feed()

    # Start the Tkinter event loop
    window.mainloop()

# Call the main function to initialize the video feed and buttons
main()
