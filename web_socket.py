import asyncio
import websockets
import cv2
import numpy as np
import time

# --- Configuration ---
HOST = "0.0.0.0"  # Listen on all available network interfaces
PORT = (
    8765  # Port for WebSocket connection. Make sure this port is open in your firewall!
)

# --- Computer Vision Placeholder Functions ---
# IMPORTANT: These are highly simplified placeholders.
# You will need to implement robust CV algorithms here.


async def detect_badminton_court(frame):
    """
    Placeholder for badminton court detection.
    Should return a representation of the detected court (e.g., corner points, homography matrix).
    For a real system, this would involve:
    - Edge detection (Canny)
    - Line detection (Hough Transform)
    - Geometric analysis to find parallel/perpendicular lines that form the court.
    - Perspective transformation (Homography) if the camera isn't perfectly overhead.
    """
    # Simulate a basic court detection (e.g., always found for demo)
    # In a real scenario, this would involve complex image processing.
    # For now, let's assume a fixed rectangle for the 'in' area relative to image size.
    h, w, _ = frame.shape
    court_rect = (
        int(w * 0.1),
        int(h * 0.2),
        int(w * 0.8),
        int(h * 0.6),
    )  # (x, y, width, height)
    return (
        court_rect,
        True,
    )  # Return court rectangle and a boolean indicating if detected


async def detect_shuttlecock(frame):
    """
    Placeholder for shuttlecock detection.
    Should return the shuttlecock's position (x, y) if detected, and a boolean.
    For a real system, this would involve:
    - Background subtraction
    - Object detection using a pre-trained model (e.g., YOLO, SSD, custom trained model)
    - Tracking algorithms (e.g., Kalman filter) for smoother results
    """
    # Simulate shuttlecock detection
    # For a real system, you might use:
    # `model.predict(frame)` to get bounding boxes
    # This is just for demonstration purposes.

    # Try to find a bright spot or use a simple color filter
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Example for white/yellow shuttlecock (adjust these ranges)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shuttlecock_pos = None
    shuttlecock_detected = False

    if contours:
        # Find the largest contour (might be the shuttlecock)
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 50:  # Minimum area to consider
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                shuttlecock_pos = (cX, cY)
                shuttlecock_detected = True

    return shuttlecock_pos, shuttlecock_detected


async def judge_line(shuttlecock_pos, court_rect):
    """
    Placeholder for line judgment logic.
    Determines if the shuttlecock landed in or out.
    `court_rect` is (x, y, width, height) of the assumed court 'in' area.
    """
    if shuttlecock_pos is None or court_rect is None:
        return "UNKNOWN"

    sx, sy = shuttlecock_pos
    cx, cy, cw, ch = court_rect

    # Simple check if shuttlecock is within the court rectangle
    # In a real system, you'd account for perspective, line thickness, etc.
    is_in_x = cx <= sx <= (cx + cw)
    is_in_y = cy <= sy <= (cy + ch)

    if is_in_x and is_in_y:
        return "IN"
    else:
        return "OUT"


# --- WebSocket Handler ---
async def badminton_line_judge_handler(websocket):
    print(f"Client connected: {websocket.remote_address}")
    try:
        while True:
            try:
                # Receive image bytes from the client
                # Assuming the client sends JPEG bytes directly
                message = await websocket.recv()

                # Convert bytes to a numpy array
                np_array = np.frombuffer(message, np.uint8)

                # Decode the image using OpenCV
                frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

                if frame is None:
                    print("Failed to decode image. Skipping frame.")
                    continue

                # --- Perform Computer Vision Tasks ---
                start_time_cv = time.time()

                # 1. Court Detection
                court_rect, court_detected = await detect_badminton_court(
                    frame.copy()
                )  # Pass a copy if frame is modified

                # 2. Shuttlecock Detection
                shuttlecock_pos, shuttlecock_detected = await detect_shuttlecock(
                    frame.copy()
                )

                # 3. Line Judgment
                judgment_result = "NO_DETECTION"
                if court_detected and shuttlecock_detected:
                    judgment_result = await judge_line(shuttlecock_pos, court_rect)
                elif not court_detected:
                    judgment_result = "NO_COURT"
                elif not shuttlecock_detected:
                    judgment_result = "NO_SHUTTLECOCK"

                end_time_cv = time.time()
                cv_latency_ms = (end_time_cv - start_time_cv) * 1000

                # --- Visualize Results on Frame (for server-side debugging) ---
                if court_detected and court_rect:
                    cx, cy, cw, ch = court_rect
                    cv2.rectangle(
                        frame, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2
                    )  # Green rectangle for court

                if shuttlecock_detected and shuttlecock_pos:
                    cv2.circle(
                        frame, shuttlecock_pos, 5, (0, 0, 255), -1
                    )  # Red circle for shuttlecock

                # Add text overlay
                cv2.putText(
                    frame,
                    f"Judgment: {judgment_result}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"CV Latency: {cv_latency_ms:.2f}ms",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Display the processed frame
                cv2.imshow("Backend Processed Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit the display
                    break

                # Send judgment result back to client
                await websocket.send(judgment_result)

            except websockets.exceptions.ConnectionClosedOK:
                print("Client disconnected gracefully.")
                break
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"Client disconnected with error: {e}")
                break
            except Exception as e:
                print(f"Error processing frame or sending data: {e}")
                # Optionally, send an error message back to the client
                await websocket.send(f"ERROR: {e}")

    finally:
        print(f"Handler for {websocket.remote_address} closing.")
        cv2.destroyAllWindows()  # Close any OpenCV windows when handler exits


# --- Main Server Function ---
async def main():
    print(f"Starting WebSocket server on ws://{HOST}:{PORT}")
    async with websockets.serve(badminton_line_judge_handler, HOST, PORT):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
