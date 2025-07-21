import asyncio
import websockets
import json
import io
import os
import time
import cv2  # For OpenCV processing
import numpy as np  # For numerical operations with OpenCV

# --- Configuration ---
WEBSOCKET_HOST = "0.0.0.0"  # Listen on all available interfaces
WEBSOCKET_PORT = 8765
SAVE_RECEIVED_IMAGES = True  # Set to True to save images to disk
SAVE_DIR = "received_camera_frames"
DISPLAY_FRAMES_OPENCV = True  # Set to True to display frames using OpenCV
APPLY_CV_PROCESSING = (
    True  # Apply a simple OpenCV filter if DISPLAY_FRAMES_OPENCV is True
)

# Ensure save directory exists
if SAVE_RECEIVED_IMAGES:
    os.makedirs(SAVE_DIR, exist_ok=True)


async def image_stream_handler(websocket):
    """
    Handles an individual WebSocket connection for image streaming.
    """
    client_address = websocket.remote_address
    print(f"Client connected from {client_address}")

    # Variables to store incoming data
    current_metadata = None

    try:
        async for message in websocket:
            if isinstance(message, str):
                # Assume string messages are metadata (JSON)
                try:
                    current_metadata = json.loads(message)
                    # print(f"Received metadata from {client_address}: {current_metadata}")
                except json.JSONDecodeError:
                    print(
                        f"Received malformed JSON metadata from {client_address}: {message}"
                    )
            elif isinstance(message, bytes):
                # Assume binary messages are JPEG image data
                if current_metadata:
                    frame_number = current_metadata.get("frame_number", "N/A")
                    timestamp_ms = current_metadata.get(
                        "timestamp", 0
                    )  # Milliseconds since epoch
                    width = current_metadata.get("width", "N/A")
                    height = current_metadata.get("height", "N/A")
                    size_bytes = len(message)

                    print(
                        f"[{client_address}] Frame {frame_number} (W:{width}, H:{height}, Size:{size_bytes} bytes)"
                    )

                    # --- Process the JPEG image ---
                    try:
                        # Convert bytes to numpy array for OpenCV
                        np_arr = np.frombuffer(message, np.uint8)
                        frame = cv2.imdecode(
                            np_arr, cv2.IMREAD_COLOR
                        )  # Decode JPEG to BGR image

                        if frame is None:
                            print(
                                f"[{client_address}] Failed to decode JPEG frame {frame_number}"
                            )
                            continue

                        # Option 1: Save the image to disk
                        if SAVE_RECEIVED_IMAGES:
                            # Use timestamp for unique filename to avoid overwriting if frame_number resets
                            filename = os.path.join(
                                SAVE_DIR,
                                f"frame_{frame_number}_{int(timestamp_ms)}.jpeg",
                            )
                            cv2.imwrite(filename, frame)
                            # print(f"[{client_address}] Saved image to {filename}")

                        # Option 2: Display and Process with OpenCV
                        if DISPLAY_FRAMES_OPENCV:
                            processed_frame = frame.copy()

                            if APPLY_CV_PROCESSING:
                                # Example 1: Convert to grayscale
                                processed_frame = cv2.cvtColor(
                                    processed_frame, cv2.COLOR_BGR2GRAY
                                )
                                # Convert back to 3 channels for display consistency if needed, or just show grayscale
                                processed_frame = cv2.cvtColor(
                                    processed_frame, cv2.COLOR_GRAY2BGR
                                )  # For coloring text

                                # Example 2: Add text overlay
                                text = f"Frame: {frame_number}"
                                cv2.putText(
                                    processed_frame,
                                    text,
                                    (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 255, 0),
                                    2,
                                    cv2.LINE_AA,
                                )

                                # Example 3: Edge Detection (uncomment to try)
                                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                processed_frame = cv2.Canny(gray, 50, 150)
                                processed_frame = cv2.cvtColor(
                                    processed_frame, cv2.COLOR_GRAY2BGR
                                )

                            # Display the frame
                            cv2.imshow(
                                f"Live Stream from {client_address}", processed_frame
                            )
                            key = cv2.waitKey(1)  # Wait for 1ms for a key press
                            if key == ord("q"):  # Press 'q' to quit the display window
                                print(
                                    f"[{client_address}] 'q' pressed. Closing display window."
                                )
                                cv2.destroyWindow(f"Live Stream from {client_address}")
                                # Note: This only closes the window for THIS client's stream.
                                # You might want a more global shutdown mechanism.

                        # Reset metadata for the next frame
                        current_metadata = None

                        await websocket.send("result")

                    except Exception as e:
                        print(
                            f"[{client_address}] Error processing image {frame_number}: {e}"
                        )
                        current_metadata = (
                            None  # Reset metadata on processing error too
                        )
                else:
                    print(
                        f"[{client_address}] Received image without preceding metadata. Size: {len(message)} bytes"
                    )
                    # If metadata is crucial, you might discard this frame or log an error.
            else:
                print(
                    f"[{client_address}] Received unexpected message type: {type(message)}"
                )

    except websockets.exceptions.ConnectionClosedOK:
        print(f"Client {client_address} disconnected cleanly.")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Client {client_address} disconnected with error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred with client {client_address}: {e}")
    finally:
        # Clean up resources if necessary
        if DISPLAY_FRAMES_OPENCV:
            cv2.destroyWindow(
                f"Live Stream from {client_address}"
            )  # Ensure window is closed on disconnect


async def main():
    """
    Main function to start the WebSocket server.
    """
    print(f"Starting WebSocket server on ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
    print(f"Saving images: {SAVE_RECEIVED_IMAGES} (to '{SAVE_DIR}')")
    print(f"Displaying frames (OpenCV): {DISPLAY_FRAMES_OPENCV}")
    print(f"Applying CV processing: {APPLY_CV_PROCESSING}")

    # Start the server
    async with websockets.serve(image_stream_handler, WEBSOCKET_HOST, WEBSOCKET_PORT):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user (Ctrl+C).")
    finally:
        # Ensure all OpenCV windows are closed if the server is stopped
        if DISPLAY_FRAMES_OPENCV:
            cv2.destroyAllWindows()
