'''
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
Thi is drwaing on screen 
below code is for the above
IMPORTANT

'''
from ctypes import c_wchar_p
import speech_recognition as sr
from cv2 import cv2
import os
import mediapipe as mp
import time
import pickle
import pandas as pd
import math
import numpy as np
import multiprocessing

def run(flag_1,flag_2,LReg):
    colors = [(255, 0, 0),(255, 165, 0), (255, 255, 0), (0, 128, 0), (0, 0, 255), (75, 0, 130), (238, 130, 238)]
    color_in_use = colors[1]
    cols=[]
    for i in range(1,43):
        cols.append("col-"+str(i))

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1700)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1700)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    
    pTime = 0
    cTime = 0
    pts = []
    dst=100
    z= 0
    temp_list=[]
    flag =False
    mid_x =0
    mid_y =0

    while True:

        

        success, img = cap.read()
        img = cv2.flip(img,1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        h, w, c = img.shape
        # print(results.multi_hand_landmarks)
        symbol =""
        pr=0
        if results.multi_hand_landmarks:
            cnt=0
            
            for handLms in results.multi_hand_landmarks:
                data_points = []
                l=[]
                l2=[]
                #print(cnt)
                cnt=cnt+1
                
                thumb_x = 0
                index_y = 0
                for id, lm in enumerate(handLms.landmark):
                    
                    
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id == 4:
                        thumb_x = cx
                        thumb_y = cy
                        thumb_z = lm.z
                        
                    if id == 20:
                        pinky_x = cx
                        pinky_y = cy
                        pinky_z = lm.z

                    if id == 12:
                        middle_x = cx
                        middle_y = cy
                        middle_z = lm.z
                        
                    if id == 8:
                        index_x = cx
                        index_y = cy
                        index_z = lm.z
                    data_points.extend([cx,cy])
                    #cv2.putText(img, str(int(id)), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,255), 1)
                for i in range(0,42,2):
                    l.append(int(data_points[i])-int(data_points[0]))
                    l.append(int(data_points[i+1])-int(data_points[1]))
                l2.append(l)
                df = pd.DataFrame(l2, columns = cols)
                df['index_y']=df['col-18']/df['col-12']
                df['middle_y']=df['col-26']/df['col-20']
                df['ring_y']=df['col-34']/df['col-28']
                df['pinky_y']=df['col-42']/df['col-36']
                df['thumb_y']=df['col-6']/df['col-10']
                df['thumb_x']=abs(df['col-5']/df['col-9'])
                df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

                try:
                    symbol = LReg.predict(df)
                    #print(symbol)
                    prob = LReg.predict_proba(df)
                    pr = prob[0,prob.argmax(1).item()]
                    #print(pr,symbol)
                except:
                    pass   
                if pr>0.99999999:
                    if symbol=="stop":
                        pass
                        #pts=[]
                        #pts.append([])
                    elif symbol == "thumbsup":
                        ind = colors.index(color_in_use)
                        length = len(colors)
                        if ind+1== length:
                            color_in_use = colors[0]
                        else:
                            color_in_use = colors[ind+1]

                # if flag_1.value:
                #     print("inside draw clean")
                #     pts=[]
                #     pts.append([])
                #     flag_1.value = False
                dst2 = math.sqrt(((pinky_x-thumb_x)*(pinky_x-thumb_x))+((pinky_y-thumb_y)*(pinky_y-thumb_y))) 
                dst = math.sqrt(((middle_x-index_x)*(middle_x-index_x))+((middle_y-index_y)*(middle_y-index_y)))


                
            if flag_1.value == 1:
                print("inside draw clean")
                pts=[]
                pts.append([])
                flag_1.value = 0
            elif flag_1.value == 2:
                print("inside color change")
                ind = colors.index(color_in_use)
                length = len(colors)
                if ind+1== length:
                    color_in_use = colors[0]
                else:
                    color_in_use = colors[ind+1]
                flag_1.value = 0
            elif flag_1.value == 3:
                print("inside undoing last one")
                pts = pts[:-1]
                flag_1.value = 0


            if dst2<50:
                pts=[]
                pts.append([])

            mid_x = index_x
            mid_y = index_y
            
            cv2.drawMarker(img,(mid_x,mid_y),color=(125,125,0), markerType=cv2.MARKER_STAR, thickness=2)

        if dst >50:
            
            if flag:
                temp_list.append([mid_x, mid_y])
                pts[-1]=temp_list
            else:
                temp_list.append([mid_x, mid_y])
                pts.append(temp_list)
            #length of lines in screen
            #print(len(pts)-1)
            flag = True
            #print(pts)
        else:
            temp_list=[]
            flag=False
        try:
            for pt in pts:    
                for i in range(len(pt)-1):
                    cv2.line(img, (pt[i][0]   ,  pt[i][1]), (pt[i+1][0],pt[i+1][1]) , color_in_use, thickness=8)
        except:
            print("not working")
        cv2.imshow("Image", img)
        cv2.waitKey(1)


def event(flag_1,flag_2):
    r = sr.Recognizer()   
    while True:
        #flag_1.value = False
        
        with sr.Microphone() as source:
            print("Talk")
            audio_text = r.listen(source,phrase_time_limit=5)
            print("Time over, thanks")    
            try:
                text = (r.recognize_google(audio_text)).lower()
                print(text)
                if "erase" in text or "clean" in text:
                    print("inside cleanng")
                    flag_1.value = 1
                    time.sleep(2)
                if "colour" in text or "colours" in text:
                    print("inside color")
                    flag_1.value = 2
                    time.sleep(2)
                if "undo" in text:
                    print("undo")
                    flag_1.value = 3
                    time.sleep(2)

                else:
                    continue
            except:
                continue


def run_alone(LReg):
    colors = [(255, 0, 0),(255, 165, 0), (255, 255, 0), (0, 128, 0), (0, 0, 255), (75, 0, 130), (238, 130, 238)]
    color_in_use = colors[1]
    cols=[]
    for i in range(1,43):
        cols.append("col-"+str(i))

    cap = cv2.VideoCapture(0)

    


    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1700)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1700)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    
    pTime = 0
    cTime = 0
    pts = []
    dst=100
    z= 0
    temp_list=[]
    flag =False
    mid_x=0
    mid_y=0

    while True:

        

        success, img = cap.read()
        img = cv2.flip(img,1)
        
        
        
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        h, w, c = img.shape
        # print(results.multi_hand_landmarks)
        symbol =""
        pr=0
        if results.multi_hand_landmarks:
            cnt=0
            
            for handLms in results.multi_hand_landmarks:
                data_points = []
                l=[]
                l2=[]
                #print(cnt)
                cnt=cnt+1
                
                thumb_x = 0
                index_y = 0
                for id, lm in enumerate(handLms.landmark):
                    
                    
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    

                    if id == 4:
                        thumb_x = cx
                        thumb_y = cy
                        thumb_z = lm.z
                        
                    if id == 20:
                        pinky_x = cx
                        pinky_y = cy
                        pinky_z = lm.z

                    if id == 12:
                        middle_x = cx
                        middle_y = cy
                        middle_z = lm.z
                        
                    if id == 8:
                        index_x = cx
                        index_y = cy
                        index_z = lm.z
                    data_points.extend([cx,cy])
                    #cv2.putText(img, str(int(id)), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,255), 1)
                for i in range(0,42,2):
                    l.append(int(data_points[i])-int(data_points[0]))
                    l.append(int(data_points[i+1])-int(data_points[1]))
                l2.append(l)
                df = pd.DataFrame(l2, columns = cols)
                df['index_y']=df['col-18']/df['col-12']
                df['middle_y']=df['col-26']/df['col-20']
                df['ring_y']=df['col-34']/df['col-28']
                df['pinky_y']=df['col-42']/df['col-36']
                df['thumb_y']=df['col-6']/df['col-10']
                df['thumb_x']=abs(df['col-5']/df['col-9'])
                df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

                try:
                    symbol = LReg.predict(df)
                    #print(symbol)
                    prob = LReg.predict_proba(df)
                    pr = prob[0,prob.argmax(1).item()]
                    #print(pr,symbol)
                except:
                    pass   
                if pr>0.99999999:
                    if symbol=="stop":
                        pass
                        #pts=[]
                        #pts.append([])
                    elif symbol == "thumbsup":
                        ind = colors.index(color_in_use)
                        length = len(colors)
                        if ind+1== length:
                            color_in_use = colors[0]
                        else:
                            color_in_use = colors[ind+1]
                        # from pynput.keyboard import Key, Controller
                        # import time
                        # keyboard = Controller()
                        # key1 = "ctrl"
                        # key2 = "d"
                        # keyboard.press(Key.alt)
                        # keyboard.press(Key.tab)
                        # keyboard.release(Key.tab)
                        # keyboard.press(Key.tab)
                        # keyboard.release(Key.tab)
                        # keyboard.release(Key.alt)
                        
                        # cv2.putText(img, str("will mute in next 5 second"), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,0), 1)
                        # time.sleep(5)
                        # keyboard.press(Key.ctrl)
                        # keyboard.press(key2)
                        # keyboard.release(Key.ctrl)
                        # keyboard.release(key2)

                # if flag_1.value:
                #     print("inside draw clean")
                #     pts=[]
                #     pts.append([])
                #     flag_1.value = False
                dst2 = math.sqrt(((pinky_x-thumb_x)*(pinky_x-thumb_x))+((pinky_y-thumb_y)*(pinky_y-thumb_y))) 
                dst = math.sqrt(((middle_x-index_x)*(middle_x-index_x))+((middle_y-index_y)*(middle_y-index_y)))
                # mid_x= int((middle_x+index_x)/2)
                # mid_y = int((middle_y+index_y)/2)
                # z = (middle_z + index_z)*-500
                if dst2<50:
                    pts=[]
                    pts.append([])
                mid_x = index_x
                mid_y = index_y

                cv2.putText(img, str(int(dst)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,255), 1)
                #cv2.putText(img, str(z), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,0), 1)
                #cv2.putText(img, str(dst/z), (10, 150), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,0), 1)
                
                cv2.drawMarker(img,(mid_x,mid_y),color=(125,125,0), markerType=cv2.MARKER_STAR, thickness=1)

        if dst >80:
            #if (dst/z)<0.5:
            if flag:
                temp_list.append([mid_x, mid_y])
                pts[-1]=temp_list
            else:
                temp_list.append([mid_x, mid_y])
                pts.append(temp_list)
            #length of lines in screen
            #print(len(pts)-1)
            flag = True
            #print(pts)
        else:
            temp_list=[]
            flag=False
        try:
            for pt in pts:    
                for i in range(len(pt)-1):
                    cv2.line(img, (pt[i][0]   ,  pt[i][1]), (pt[i+1][0],pt[i+1][1]) , color_in_use, thickness=8)
        except:
            print("not working")
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.split(os.path.abspath(__file__))[0],".."))
    LReg= pickle.load(open("/home/akshay/data/personal/Python_projects/Hand_gesture/models/basicLR_5.pkl", 'rb'))
    #pts = []
    is_voice = False
    
    if is_voice:
        flag_1 = multiprocessing.Value('i',1)
        flag_2 = multiprocessing.Value('i',False)

        p1 = multiprocessing.Process(target=run, args=(flag_1,flag_2,LReg))

        p2 = multiprocessing.Process(target=event, args=(flag_1,flag_2))
        p1.start()
        print("starting process 2")
        #time.sleep(10)
        p2.start()
    else:
        run_alone(LReg)
