import streamlit as st
import cv2
from PIL import Image
import numpy as np
import HandTrackingModule as htm
import time
import autopy

def main():
    st.title("Virtual Mouse using Hand Tracking")
    st.write("Welcome to the Virtual Mouse application!")
    st.write("This application uses hand tracking to control the mouse cursor on your screen.")
    st.write("Simply move your index finger to move the mouse, and bring your index and middle fingers close to click.")
    image_url = "ai.jpg"
    st.image(image_url, caption="Example Image", width=300)
    wCam, hCam = 640, 480
    frameR = 100  # Frame Reduction
    smoothening = 7

    pTime = 0
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    detector = htm.handDetector(maxHands=1)
    wScr, hScr = autopy.screen.size()

    while True:
        # Read frame from camera
        success, frame = cap.read()

        # Convert the frame to RGB and display it using Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB")

        # Process the frame
        img = detector.findHands(frame)
        lmList, bbox = detector.findPosition(img)
        fingers = detector.fingersUp()

        if len(fingers) >= 3 and fingers[1] == 1 and fingers[2] == 0:
            x1, y1 = lmList[8][1:]
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            if 0 <= clocX <= wScr and 0 <= clocY <= hScr:
                autopy.mouse.move(wScr - clocX, clocY)

            plocX, plocY = clocX, clocY

        if len(fingers) >= 3 and fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            if length < 40:
                autopy.mouse.click()

        # Calculate and display frame rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        st.write(f"FPS: {int(fps)}")

# Run the Streamlit application
if __name__ == "__main__":
    main()
