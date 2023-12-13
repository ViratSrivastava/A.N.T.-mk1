import cv2

def edge_detection_canny_harris(video_path, low_threshold=50, high_threshold=150, harris_block_size=2, harris_ksize=3, harris_k=0.04, blur_kernel_size=3):
    """
    Applies Canny edge detection and Harris corner detection to each frame of a video.

    Parameters:
    - video_path (str): Path to the input video file.
    - low_threshold (int, optional): Lower threshold for Canny edge detection (default: 50).
    - high_threshold (int, optional): Higher threshold for Canny edge detection (default: 150).
    - harris_block_size (int, optional): Neighborhood size for Harris corner detection (default: 2).
    - harris_ksize (int, optional): Aperture parameter for the Sobel operator in Harris corner detection (default: 3).
    - harris_k (float, optional): Harris detector free parameter (default: 0.04).
    - blur_kernel_size (int, optional): Size of the Gaussian blur kernel (default: 3).

    Returns:
    None

    Displays two windows:
    - 'Edge Detection (Canny)': Displays the result of Canny edge detection on the input video frames.
    - 'Corner Detection (Harris)': Displays the result of Harris corner detection on the input video frames.

    Press 'q' to exit the video playback.

    Example:
    ```python
    input_video_path = 'input/car.mp4'
    low_threshold = 50
    high_threshold = 150
    harris_block_size = 2
    harris_ksize = 3
    harris_k = 0.04
    blur_kernel_size = 3

    edge_detection_canny_harris(input_video_path, low_threshold, high_threshold, harris_block_size, harris_ksize, harris_k, blur_kernel_size)
    ```
    """
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0), low_threshold, high_threshold)

        corners = cv2.cornerHarris(gray, blockSize=harris_block_size, ksize=harris_ksize, k=harris_k)
        corners = cv2.dilate(corners, None)
        frame[corners > 0.01 * corners.max()] = [0, 0, 255]

        cv2.imshow('Edge Detection (Canny)', edges)
        cv2.imshow('Corner Detection (Harris)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path = 'input/car.mp4'
    low_threshold = 50
    high_threshold = 150
    harris_block_size = 2
    harris_ksize = 3
    harris_k = 0.04
    blur_kernel_size = 3

    edge_detection_canny_harris(input_video_path, low_threshold, high_threshold, harris_block_size, harris_ksize, harris_k, blur_kernel_size)