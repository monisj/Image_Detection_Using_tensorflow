pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord("r"):  # Record pressing r
                sign_recorder.record()
            elif pressedKey == ord("q"):  # Break pressing q
                break