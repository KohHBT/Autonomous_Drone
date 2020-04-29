import face_recognition

picture_of_me = face_recognition.load_image_file("Andrew.jpg")
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]


unknown_picture = face_recognition.load_image_file("Thuy.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]


results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

if results[0] == True:
    print("It's a picture of Andrew!")
else:
    print("It's not a picture of Andrew!")