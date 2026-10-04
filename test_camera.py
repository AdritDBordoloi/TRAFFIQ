import cv2
import sys

def test_camera_stream(camera_index=0):
    print(f"[INFO] Initializing camera device index: {camera_index}...")
    
    # Use DirectShow backend on Windows for faster initialization
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_index, backend)

    if not cap.isOpened():
        print(f"[ERROR] Could not access camera index {camera_index}.")
        print("[TIP] If using an external webcam or phone camera app, try changing camera_index to 1 or 2.")
        return

    print("[SUCCESS] Camera stream locked. Press 'q' on your keyboard to close the window.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Failed to grab frame. Camera may have been disconnected.")
            break

        # Display basic overlay text on the live video feed
        cv2.putText(
            frame, 
            "TRAFFIQ - Camera Stream Online (Press 'q' to exit)", 
            (20, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 255, 0), 
            2
        )

        cv2.imshow("TRAFFIQ Camera Diagnostic", frame)

        # Press 'q' to break out of the loop and close the feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean release of hardware resources
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera stream terminated cleanly.")

if __name__ == "__main__":
    test_camera_stream(0)