import face_recognition
image = face_recognition.load_image_file("biden.jpg")
face_landmarks_list = face_recognition.face_landmarks(image,model="cnn")
print(face_landmarks_list)