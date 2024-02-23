import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class HandTracker:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5,
                 model_complexity=1, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.model_complexity = model_complexity
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands,
                                         self.model_complexity,
                                         self.detection_confidence,
                                         self.tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, image, draw=True):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)

        if draw and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(image, hand_landmarks,
                                            self.mp_hands.HAND_CONNECTIONS)
        return image

    def find_position(self, image, hand_no=0, draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            hand_landmarks = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(image, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lm_list

def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    is_hand_closed = False
    target_volume = volume.GetMasterVolumeLevelScalar()  # Initialize target volume

    while True:
        success, image = cap.read()
        image = tracker.find_hands(image)
        lm_list = tracker.find_position(image)

        if len(lm_list) != 0:
            thumb = lm_list[4]
            index = lm_list[8]

            thumb_x, thumb_y = thumb[1], thumb[2]
            index_x, index_y = index[1], index[2]

            distance = np.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)

            max_distance = 250  # Maximum distance value
            volume_change = int((distance / max_distance) * 2000)  # Adjust as needed

            current_volume = volume.GetMasterVolumeLevelScalar()

            if distance < 50:
                is_hand_closed = True
            elif distance > 100:
                is_hand_closed = False

            # Adjust the target volume based on hand gesture
            if is_hand_closed:
                target_volume = max(0.0, current_volume - (volume_change / 5000.0))
            else:
                target_volume = min(1.0, current_volume + (volume_change / 5000.0))

            # Gradually adjust the volume towards the target
            volume_step = 0.01  # Adjust as needed for the desired speed
            if current_volume < target_volume:
                current_volume = min(current_volume + volume_step, target_volume)
            elif current_volume > target_volume:
                current_volume = max(current_volume - volume_step, target_volume)

            # Set the new volume
            volume.SetMasterVolumeLevelScalar(current_volume, None)

            print(f"Ses seviyesi: {current_volume * 100:.2f}%")

        cv2.imshow("Video", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
