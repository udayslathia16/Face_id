import face_recognition
import numpy as np
import cv2
import cmake
from datetime import datetime
import copy
import csv


video_capture=cv2.VideoCapture(0)

piyush_faces=face_recognition.load_image_file("faces/piyush.jpg")
piyush_encodings=face_recognition.face_encodings(piyush_faces)[0]
uday_faces=face_recognition.load_image_file("faces/uday.jpg")
uday_encodings=face_recognition.face_encodings(uday_faces)[0]
vidhya_faces=face_recognition.load_image_file("faces/vidhya.jpg")
vidhya_encodings=face_recognition.face_encodings(vidhya_faces)[0]
shivam_faces=face_recognition.load_image_file("faces/shivam.jpg")
shivam_encodings=face_recognition.face_encodings(shivam_faces)[0]
pranav_faces=face_recognition.load_image_file("faces/pranav.jpg")
pranav_encodings=face_recognition.face_encodings(pranav_faces)[0]
farhan_faces=face_recognition.load_image_file("faces/farhan.jpg")
farhan_encodings=face_recognition.face_encodings(farhan_faces)[0]

known_face_encodings=(piyush_encodings,uday_encodings,vidhya_encodings,shivam_encodings,pranav_encodings,farhan_encodings)
known_face_name=("piyush","Uday","Vidhya","Shivam","Pranav","Farhan")


stu= copy.copy(known_face_name)
students=list(stu)

face_locations =[]
face_encodings =[]

now=datetime.now()
current_date=now.strftime("%Y-%m-%d")


f=open(f"{current_date}.csv","w+",newline="")
lnwriter=csv.writer(f)

while True:
    _, frame=video_capture.read()
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame=cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)


    face_locations=face_recognition.face_locations(rgb_small_frame)
    face_encodings=face_recognition.face_encodings(rgb_small_frame,face_locations)

    for face_encoding in face_encodings:
        mathes=face_recognition.compare_faces(known_face_encodings,face_encoding)
        face_distance=face_recognition.face_distance(known_face_encodings,face_encoding)
        best_match_index= np.argmin(face_distance)

        if(mathes[best_match_index]):
            name=known_face_name[best_match_index]


        if name in known_face_name:
            font=cv2.FONT_HERSHEY_SIMPLEX
            bottomleftCornerOfText=(10,100)
            fontScale=1.5
            fontColor =(255,0,0)
            thickness=3
            lineType=2
            cv2.putText(frame,name + " Present",bottomleftCornerOfText,font,fontScale,fontColor,thickness,lineType)



        if name in students:
            students.remove(name)
            current_time=now.strftime("%H-%M-%S")
            lnwriter.writerow([name,current_time])

    cv2.imshow("Attandance",frame)
    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()