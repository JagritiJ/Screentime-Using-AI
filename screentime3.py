# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 20:55:08 2020

@author: jagri
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 14:13:32 2020
@author: jagri
"""

from pytube import YouTube
import os
import cv2
from tqdm import tqdm
import numpy as np
import keras

import tensorflow as tf

from glob import glob

import face_recognition
#requires visual studio and cmake

from mtcnn.mtcnn import MTCNN

from sklearn.cluster import KMeans

import tensorflow as tf

import matplotlib.pyplot as plt

import shutil


def downloadYouTube(vid_url, path):
    yt = YouTube(vid_url)
    yt = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    print(yt)
    yt.download(path)
    if not os.path.exists(path):
        os.makedirs(path)

def break_into_frames():
    
    cap = cv2.VideoCapture('BTS (방탄소년단) FAKE LOVE Official MV.mp4')
    i =0
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    
    image_folder = 'C:/Users/jagri/Downloads/Projects/screentime/train/data_1'
    while True:
        ret, frame = cap.read()
    
        if ret == False:
            break
        cv2.imwrite(image_folder+'/'+str(i)+'.jpg', frame)
        i+=1
    
    cap.release()
    cv2.destroyAllWindows()

def find_faces_extract_store_in_folder():
    print("extract just the faces from the frame, this involves cropping")
    

def face_recognition_module(image1, image2):
    print("inside face recognition module")
    #https://towardsdatascience.com/face-detection-recognition-and-emotion-detection-in-8-lines-of-code-b2ce32d4d5de
    face_locations = face_recognition.face_locations(image1)
    print(len(face_locations))
    print(face_locations)
    if face_locations !=0:
        for face_location in face_locations:
            top, right, bottom, left = face_locations[0]
            face_image = image1[top:bottom, left:right]
            encoding_1 = face_recognition.face_encodings(image1)[0]   
            
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image',face_image)
            cv2.waitKey(0)
            
            
    # print(encoding_1)

    face_locations2 = face_recognition.face_locations(image2)
    print(len(face_locations2))
    print(face_locations2)
    if face_locations2 !=0:
        for face_location in face_locations2:
            top, right, bottom, left = face_locations2[0]
            face_image2 = image2[top:bottom, left:right]
            encoding_2 = face_recognition.face_encodings(image2)[0]   
            
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image',face_image2)
            cv2.waitKey(0)
        
    cv2.destroyAllWindows()
    
    results = face_recognition.compare_faces([encoding_1], encoding_2,tolerance=0.50)
    print("face comparison results ", results)
    
    return face_image
    
def face_match_check(image1, image2):
    #https://towardsdatascience.com/face-detection-recognition-and-emotion-detection-in-8-lines-of-code-b2ce32d4d5de
   
   
    # face_image1 = face_recognition_module(image1)
    # encoding_1 = face_recognition.face_encodings(image1)[0]
    
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image',face_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  

  
    
    # face_locations2 = face_recognition.face_locations(image2)
    # top2, right2, bottom2, left2 = face_locations2[0]
    # face_image2 = image1[top2:bottom2, left2:right2]
    # encoding_2 = face_recognition.face_encodings(image2)[0]
    

   

    # results = face_recognition.compare_faces([encoding_1], encoding_2,tolerance=0.50)
    # print("face comparison results ", results)
       

def face_detection_using_dl_mtcnn(image, path,i):
    
    detector = MTCNN()
    # detect faces in the image
    faces = detector.detect_faces(image)
    # display faces on the original image
    # draw_faces(image, faces)
    print("faces ", faces)
    j=0
    for face in faces:
	# get coordinates
        x, y, w, h = face['box']
    	
        # extract the face and save in another folder
        
        extracted_face_by_cropping = image[y:y+h, x:x+w]
        # cv2.imshow("extracted_face", extracted_face_by_cropping)
        
        faces_folder = 'faces'
        if not os.path.exists(faces_folder):
            os.makedirs(faces_folder)
        
        if extracted_face_by_cropping.size !=0:
            cv2.imwrite(path+'/'+faces_folder+'/'+str(i)+str(j)+'.jpg', extracted_face_by_cropping)
        
        j+=1
        # create the shape
        cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(image, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        for key, value in face['keypoints'].items():
            print(key)
            print(value)
			# create and draw dot
            # cv2.circle(image, center_coordinates, radius, color, thickness) 
            cv2.circle(image, value, radius=2, color=(0, 0, 255), thickness =5 )
   
   
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image',image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()  

def convert_image_to_nparray(path):
    
    
    # if not os.path.exists(dest):
    #     os.makedirs(path)
    print(" convert_image_to_nparray ")
    faces_folder_path = path+'/faces'
    face_images = glob(faces_folder_path+"/*.jpg")

    #For passing the image data to the unsupervised ml model, we need to convert images to numpy array
    width, height = 224, 224
    images=[]
    
    for i in tqdm(range(len(face_images))):
        # read the image from file, resize and add to a list
        img = cv2.imread(face_images[i])
        img = cv2.resize(img , (224,224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    
    images = np.float32(images).reshape(len(images), -1)
    images /= 255
    
    images_new = images
    return face_images, images_new
    
def image_clustering_unsupervised(images_new, face_images, path):
     
    print(" image_clustering_unsupervised ")
    
    print("Before copying file:") 
    print(os.listdir(path)) 

    model = KMeans(n_clusters=10, n_jobs=-1, random_state=728)
    model.fit(images_new)
    print("labels ", model.labels_)
    
    predictions = model.predict(images_new)
    
    print(predictions)
    unique_clusters = list(set(predictions))
   
    print("unique_clusters ", unique_clusters)
    
    for i in range(0,images_new.shape[0]):
        path_name = face_images[i].split('\\')[0]
        image_name = face_images[i].split('\\')[1]
        image_name_parts = [path_name, image_name]
        image_full_path = "/".join(image_name_parts) 
        
        if model.labels_[i] ==0:
            c1 = plt.scatter(images_new[i,0], images_new[i,1], c='r', marker='+' )
            print(image_full_path, ".....",  path+'/'+str('0') )
            dest_folder = path+'/'+str('0')
            dest = dest_folder+'/'+ image_name
            print("dest ", dest)
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            shutil.copy2(image_full_path, dest) 
        
        if model.labels_[i] ==1:
            c1 = plt.scatter(images_new[i,0], images_new[i,1], c='g', marker='+' )
            dest_folder = path+'/'+str('1')
            dest = dest_folder+'/'+ image_name
            print("dest ", dest)
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            shutil.copy2(image_full_path, dest) 
            
        if model.labels_[i] ==2:
            c1 = plt.scatter(images_new[i,0], images_new[i,1], c='b', marker='+' )
            dest_folder = path+'/'+str('2')
            dest = dest_folder+'/'+ image_name
            print("dest ", dest)
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            shutil.copy2(image_full_path, dest) 
            
        if model.labels_[i] ==3:
            c1 = plt.scatter(images_new[i,0], images_new[i,1], c='#1f77b4', marker='+' )
            dest_folder = path+'/'+str('3')
            dest = dest_folder+'/'+ image_name
            print("dest ", dest)
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            shutil.copy2(image_full_path, dest) 
            
        if model.labels_[i] ==4:
            c1 = plt.scatter(images_new[i,0], images_new[i,1], c='#ff7f0e', marker='+' )
            dest_folder = path+'/'+str('4')
            dest = dest_folder+'/'+ image_name
            print("dest ", dest)
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            shutil.copy2(image_full_path, dest) 
            
        if model.labels_[i] ==5:
            c1 = plt.scatter(images_new[i,0], images_new[i,1], c='#2ca02c', marker='+' )
            dest_folder = path+'/'+str('5')
            dest = dest_folder+'/'+ image_name
            print("dest ", dest)
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            shutil.copy2(image_full_path, dest) 
            
        if model.labels_[i] ==6:
            c1 = plt.scatter(images_new[i,0], images_new[i,1], c='#d62728', marker='+' )
            dest_folder = path+'/'+str('6')
            dest = dest_folder+'/'+ image_name
            print("dest ", dest)
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            shutil.copy2(image_full_path, dest) 
            
        if model.labels_[i] ==7:
            c1 = plt.scatter(images_new[i,0], images_new[i,1], c='#9467bd', marker='+' )
            dest_folder = path+'/'+str('7')
            dest = dest_folder+'/'+ image_name
            print("dest ", dest)
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            shutil.copy2(image_full_path, dest) 
            
        if model.labels_[i] ==8:
            c1 = plt.scatter(images_new[i,0], images_new[i,1], c='#8c564b', marker='+' )
            dest_folder = path+'/'+str('8')
            dest = dest_folder+'/'+ image_name
            print("dest ", dest)
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            shutil.copy2(image_full_path, dest) 
            
        if model.labels_[i] ==9:
            c1 = plt.scatter(images_new[i,0], images_new[i,1], c='#e377c2', marker='+' )
            dest_folder = path+'/'+str('9')
            dest = dest_folder+'/'+ image_name
            print("dest ", dest)
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            shutil.copy2(image_full_path, dest) 
            
def screentime_for_a_character(path):
    
    print("Screentimes")
    
    cap = cv2.VideoCapture('BTS (방탄소년단) FAKE LOVE Official MV.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    
    print('fps = ' + str(fps))
    print('number of frames = ' + str(frame_count))
    print('duration (S) = ' + str(duration))
    minutes = int(duration/60)
    seconds = duration%60
    print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))

    i =0
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    print("total frames in 0 folder", len(glob(path+'/'+str('0')+ "/*.jpg")))
    
    char0 = (len(glob(path+'/'+str('0')+ "/*.jpg"))) * fps
    char1 = (len(glob(path+'/'+str('1')+ "/*.jpg"))) * fps
    char2 = (len(glob(path+'/'+str('2')+ "/*.jpg"))) * fps
    char3 = (len(glob(path+'/'+str('3')+ "/*.jpg"))) * fps            
    char4 = (len(glob(path+'/'+str('4')+ "/*.jpg"))) * fps
    char5 = (len(glob(path+'/'+str('5')+ "/*.jpg"))) * fps
    char6 = (len(glob(path+'/'+str('6')+ "/*.jpg"))) * fps
    char7 = (len(glob(path+'/'+str('7')+ "/*.jpg")))* fps
    char8 = (len(glob(path+'/'+str('8')+ "/*.jpg"))) * fps
    char9 = (len(glob(path+'/'+str('9')+ "/*.jpg"))) * fps         
    
    print("char 9 has screentime of ", char9)
    print("total screentimes of all",(char0+char1+char2+char3+char4+char5+char6+char7+char8+char9))
    
    cap.release()
        
def feature_extraction():
    print()
     # resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    # my_new_model = tf.keras.models.Sequential()
    # my_new_model.add(tf.keras.ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
    
    # # Say not to train first layer (ResNet) model. It is already trained
    # my_new_model.layers[0].trainable = False

    # def extract_vector(face_images):
    #     resnet_feature_list = []
    
    #     for im in glob.glob(face_images):
    
    #         im = cv2.imread(im)
    #         im = cv2.resize(im,(224,224))
    #         img = preprocess_input(np.expand_dims(im.copy(), axis=0))
    #         resnet_feature = my_new_model.predict(img)
    #         resnet_feature_np = np.array(resnet_feature)
    #         resnet_feature_list.append(resnet_feature_np.flatten())
    
    #     return np.array(resnet_feature_list)        
    
def face_detection_easyfacenet():
    print("easyfacenet library needs 1.7 tensorflow version, so leaving it for now, ERROR: No matching distribution found for tensorflow==1.7 (from easyfacenet)")    


def google_facenet_algorithm():
    print()
    #https://missinglink.ai/guides/tensorflow/tensorflow-face-recognition-three-quick-tutorials/    
    
def face_detection_using_dlib():
    print()

    
    
def face_eyes_nose_mouth_body_detection_haarcascades(image, face_cascade, eye_cascade, eye_with_glasses_classifier, nose_cascade, mouth_cascade, body_classifier):
    
    #https://stackoverflow.com/questions/38279632/viola-jones-in-python-with-opencv-detection-mouth-and-nose
    # For nose and mouth - http://alereimondo.no-ip.org/OpenCV/34
    
    # img = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print ("Found {0} faces!".format(len(faces)))
    
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(image, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
        eyes_with_glasses = eye_with_glasses_classifier.detectMultiScale(gray, 1.3, 5)
        nose =  nose_cascade.detectMultiScale(gray, 1.3, 5)
        mouth = mouth_cascade.detectMultiScale(gray, 1.7, 11)
        
    
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(image, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
            cv2.putText(image, 'Eyes', (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        for (ex,ey,ew,eh) in eyes_with_glasses:
            cv2.rectangle(image, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
            cv2.putText(image, 'Eyes', (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)    
        for (nx, ny, nw, nh) in nose:
            cv2.rectangle(image, (nx, ny), (nx + nw, ny + nh), (0, 0, 255), 2)
            cv2.putText(image, 'Nose', (nx, ny-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        for (mx, my, mw, mh) in mouth:
            cv2.rectangle(image, (mx, my), (mx + mw, my + mh), (0, 0, 0), 2)
            cv2.putText(image, 'Mouth', (mx, my-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    bodies = body_classifier.detectMultiScale(gray, 1.1, 3)
    for (bx,by,bw,bh) in bodies:
        cv2.rectangle(image, (bx, by), (bx+bw, by+bh), (0, 255, 255), 2)
        cv2.putText(image, 'Body', (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

def human_body_detection(image, body_classifier):
    
    # img = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    bodies = body_classifier.detectMultiScale(gray, 1.1, 3)
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in bodies:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)
   
    cv2.imshow('img',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def smile_detection(image, smile_classifier):
    print("Method to detect faces")
    
    # img = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smiles = smile_classifier.detectMultiScale(gray, 1.1, 3)
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in smiles:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)
   
    cv2.imshow('img',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return smiles
    
def auto_capture_when_smile_in_video(face_classifier, smile_classifier):
    print("https://data-flair.training/blogs/python-project-capture-selfie-by-detecting-smile/")
    
    video = cv2.VideoCapture(0)
    
    while True:
        success,img = video.read()
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(grayImg,1.1,4)
        cnt=1
        keyPressed = cv2.waitKey(1)
        for x,y,w,h in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),3)
            smiles = smile_classifier.detectMultiScale(grayImg,1.8,15)
            for x,y,w,h in smiles:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(100,100,100),5)
                print("Image "+str(cnt)+"Saved")
                path=r'C:/Users/jagri/Downloads/Projects/screentime/img'+str(cnt)+'.jpg'
                cv2.imwrite(path,img)
                cnt +=1
                if(cnt>=2):    
                    break
                    
        cv2.imshow('live video',img)
        if(keyPressed & 0xFF==ord('q')):
            break
    video.release()                                  
    cv2.destroyAllWindows()

    
def car_detection(image, root_path_for_haarcascade):
    
    print("Method to detect faces")
    
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    car_classifier = cv2.CascadeClassifier(root_path_for_haarcascade+'/haarcascade_car.xml')
    cars = car_classifier.detectMultiScale(gray, 1.1, 3)
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in cars:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
   
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    


def main():
    
    video_link = 'https://www.youtube.com/watch?v=7C2z4GqqS5E'
    path = 'C:/Users/jagri/Downloads/Projects/screentime'
    root_path_for_haarcascade_cuda = 'C:/Users/jagri/Downloads/datasets/opencv-3.4/opencv-3.4/data/haarcascades_cuda'
    root_path_for_haarcascade = 'C:/Users/jagri/Downloads/datasets/opencv-3.4/opencv-3.4/data/haarcascades'
   
    images_folder = 'C:/Users/jagri/Downloads/Projects/screentime/train/data_1'
    glob_path = 'C:/Users/jagri/Downloads/Projects/screentime/train'
    
    #if downloaded = 1, means the video is already downloaded, so don't run that method again. Else give the variable any number apart from 1, say 0
    downloaded = 1   
    if downloaded != 1:
        downloadYouTube(video_link, path)
    
    #if break_into_frames = 1, means the video is already broken into frames, so don't run that method again. Else give the variable any number apart from 1, say 0
    break_into_frames = 1
    if break_into_frames !=1:
        break_into_frames() 
    
    
    #if face detection and extraction of faces = 1, means it's already done, so don't run that method again. Else give the variable any number apart from 1, say 0
    face_detection_and_extraction = 1
    if face_detection_and_extraction !=1:
        images = glob(glob_path+"/data_1/*.jpg")
        i=0
        for i in tqdm(range(len(images))):
            #images = "detection-challenge/train_1\FAKE_aagfhgtpmv.mp4_frame0.jpg"
            
            path_name = images[i].split('\\')[0]
            image_name = images[i].split('\\')[1]
            image_name_parts = [path_name, image_name]
            image_full_path = "/".join(image_name_parts) 
            
            img = cv2.imread(image_full_path)
            # cv2.imshow('image_full_path',img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            if img.size !=0:            
                face_detection_using_dl_mtcnn(img, path, i+1)
    
    #if convert_image_to_nparray = 1, means it's already done, so don't run that method again. Else give the variable any number apart from 1, say 0
    convert_image_to_nparray =1
    if convert_image_to_nparray !=1:
        array = convert_image_to_nparray(path)
    
    image_clustering_unsupervised =1
    if image_clustering_unsupervised !=1:    
        image_clustering_unsupervised(array)
        
  
    option =7
    if option ==1:
        face_classifier = cv2.CascadeClassifier(root_path_for_haarcascade_cuda +'/haarcascade_frontalface_default.xml')
        eye_with_glasses_classifier = cv2.CascadeClassifier(root_path_for_haarcascade_cuda +'/haarcascade_eye_tree_eyeglasses.xml')
        eye_classifier = cv2.CascadeClassifier(root_path_for_haarcascade_cuda +'/haarcascade_eye.xml')
        nose_classifier = cv2.CascadeClassifier('C:/Users/jagri/Downloads/datasets/opencv-3.4/opencv-3.4/data/Nose25x15.1/Nariz.xml')
        mouth_classifier = cv2.CascadeClassifier('C:/Users/jagri/Downloads/datasets/opencv-3.4/opencv-3.4/data/Mouth25x15.1/Mouth.xml') 
        body_classifier = cv2.CascadeClassifier(root_path_for_haarcascade_cuda+'/haarcascade_fullbody.xml')
        smile_classifier = cv2.CascadeClassifier(root_path_for_haarcascade_cuda +'/haarcascade_smile.xml')
    
        face_eyes_nose_mouth_body_detection_haarcascades(img, face_classifier, eye_classifier, eye_with_glasses_classifier, nose_classifier, mouth_classifier, body_classifier)
    
    if option ==2:
        test_image_path = 'C:/Users/jagri/Downloads/Projects/screentime/test1.JPG'
        test_image_path2 = 'C:/Users/jagri/Downloads/Projects/screentime/test3.JPG'
        test_image_path3 = 'C:/Users/jagri/Downloads/Projects/screentime/test8.JPG'
        group_pic = 'C:/Users/jagri/Downloads/Projects/screentime/test7.JPG'
        # print(image_full_path)
        img = cv2.imread(test_image_path)
        # test1 = 'C:/Users/jagri/Downloads/Projects/screentime/test1.JPG'
        # test2 = 'C:/Users/jagri/Downloads/Projects/screentime/test3.JPG'
        
      
        img2 = cv2.imread(test_image_path2)
        img3 = cv2.imread(test_image_path3)
        group = cv2.imread(group_pic)
        
        face_recognition_module(img, img3)
    
    if option ==3:
        # able to detect different skin tones of faces. Different culture people. People with glasses
        face_detection_using_dl_mtcnn(img, path)
    
    # Auto capture of smile
    if option ==4:
        
        face_classifier = cv2.CascadeClassifier(root_path_for_haarcascade_cuda +'/haarcascade_frontalface_default.xml')
        eye_with_glasses_classifier = cv2.CascadeClassifier(root_path_for_haarcascade_cuda +'/haarcascade_eye_tree_eyeglasses.xml')
        eye_classifier = cv2.CascadeClassifier(root_path_for_haarcascade_cuda +'/haarcascade_eye.xml')
        nose_classifier = cv2.CascadeClassifier('C:/Users/jagri/Downloads/datasets/opencv-3.4/opencv-3.4/data/Nose25x15.1/Nariz.xml')
        mouth_classifier = cv2.CascadeClassifier('C:/Users/jagri/Downloads/datasets/opencv-3.4/opencv-3.4/data/Mouth25x15.1/Mouth.xml') 
        body_classifier = cv2.CascadeClassifier(root_path_for_haarcascade_cuda+'/haarcascade_fullbody.xml')
        smile_classifier = cv2.CascadeClassifier(root_path_for_haarcascade_cuda +'/haarcascade_smile.xml')
    
        auto_capture_when_smile_in_video(face_classifier, smile_classifier)
        print()
    
    # Face match check    
    if option==5:
        test1 = 'C:/Users/jagri/Downloads/Projects/screentime/test1.JPG'
        test2 = 'C:/Users/jagri/Downloads/Projects/screentime/test3.JPG'
        
        img1 = cv2.imread(test1)
        img2 = cv2.imread(test2)
        face_match_check(test1, test2)
    
    if option ==6:
        face_images, images_new = convert_image_to_nparray(path)
        image_clustering_unsupervised(images_new, face_images, path)
    
    if option ==7:
        screentime_for_a_character(path)
        

main()    

''' Fixed issue with pytube cipher - changed in extract.py under pytube
https://github.com/nficano/pytube/issues/591 '''

# https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php