import cv2
import mediapipe as mp
import time
import math
import speech_recognition as sr
import webbrowser
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

def search_on_youtube(string_to_search,maximize_window=True):
    driver = webdriver.Chrome()
    if maximize_window:
        driver.maximize_window()

    url = "https://www.youtube.com/"

    driver.get(url) # go to youtube.com
    driver.find_element(By.XPATH,'//*[text()="I Agree"]').click() # accept cookies 
    search_element = driver.find_element(By.NAME,"search_query") 
    search_element.click()
    time.sleep(0.2)
    search_element.send_keys(string_to_search+Keys.ENTER) # search "string_to_search" in youtube
    #time.sleep(1)
    #driver.find_element(By.CLASS_NAME,"title-and-badge.style-scope.ytd-video-renderer").click()
    time.sleep(5) # wait 5 seconds before closing
    driver.quit()

def search_on_google(string_to_search,maximize_window=True):
    driver = webdriver.Chrome()
    if maximize_window:
        driver.maximize_window()

    url = "https://www.google.com/"

    driver.get(url) # go to google.com
    driver.find_element(By.ID,"L2AGLb").click() # accept cookies 
    driver.find_element(By.NAME,"q").send_keys(string_to_search+Keys.ENTER) # search "string_to_search" in google
    time.sleep(5) # wait 5 seconds before closing
    driver.quit()
    
def hand_tracker():

    base_folder = "C:/Users/furka/Desktop/Python/HandTracking"

    cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    previous_time = 0
    current_time = 0
    fps_list = []
    current_fps = 0

    condition = True
    close_condition_number = 0
    wolf_condition_number = 0

    while condition:
        hand0_lms = []
        hand1_lms = []

        is_multiple_hands = False

        success, img = cap.read()
        # img = cv2.imread(base_folder+"/media/peace.jpg")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_id, handLms in enumerate(results.multi_hand_landmarks):
                for lm_id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    if hand_id == 0:
                        hand0_lms.append((cx, cy))
                        color = (255, 255, 0)
                    else:
                        is_multiple_hands = True
                        hand1_lms.append((cx, cy))
                        color = (0, 255, 0)

                    # Print the id of each point
                    cv2.putText(img, str(lm_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                # draw lines between points
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        # calculate fps --------------------------------------------------------------------------------------------

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        fps_list.append(fps)

        x = 20
        if len(fps_list) == x:
            average_fps = 0
            for el in fps_list:
                average_fps += el / x
            average_fps = int(average_fps)
            cv2.putText(img, "fps: " + str(average_fps), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            fps_list = []
            current_fps = average_fps

        else:
            cv2.putText(img, "fps: " + str(current_fps), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        # ----------------------------------------------------------------------------------------------------------

        # calculate similarity -------------------------------------------------------------------------------------

        similarity = None
        hand0_lengths = []
        hand1_lengths = []
        average_loss = 0

        if is_multiple_hands:
            for i in range(len(hand0_lms) - 1):
                length = math.sqrt(
                    (hand0_lms[i][0] - hand0_lms[i + 1][0]) ** 2 + (hand0_lms[i][1] - hand0_lms[i + 1][1]) ** 2)
                hand0_lengths.append(length)

            for i in range(len(hand1_lms) - 1):
                length = math.sqrt(
                    (hand1_lms[i][0] - hand1_lms[i + 1][0]) ** 2 + (hand1_lms[i][1] - hand1_lms[i + 1][1]) ** 2)
                hand1_lengths.append(length)

            for el1, el2 in zip(hand0_lengths, hand1_lengths):
                average_loss += abs((el1 - el2)) * 10 / (el1 + el2)
            average_loss = int(average_loss)
            similarity = 100 - average_loss

        if similarity is not None:
            similarity_string = "R-L Hand Similarity: " + str(similarity)
        else:
            similarity_string = "Show Two Hands for Similarity Analysis!"
        cv2.putText(img, similarity_string, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        # ----------------------------------------------------------------------------------------------------------

        # find how many fingers are up -----------------------------------------------------------------------------

        finger_tips = [8, 12, 16, 20]
        fingers_string = ""
        fingers_list = []
        is_thumb_at_right_of_screen = None

        if len(hand0_lms) > 0:
            is_thumb_at_right_of_screen = hand0_lms[4][0] > hand0_lms[20][0]

            if is_thumb_at_right_of_screen:
                if hand0_lms[4][0] < hand0_lms[2][0]:
                    fingers_list.append(0)
                    fingers_string += "Closed "
                else:
                    fingers_list.append(1)
                    fingers_string += "Open  "
            else:
                if hand0_lms[4][0] > hand0_lms[2][0]:
                    fingers_list.append(0)
                    fingers_string += "Closed "
                else:
                    fingers_list.append(1)
                    fingers_string += "Open  "
        for i in range(0, 4):
            if len(hand0_lms) > 0:
                if hand0_lms[finger_tips[i]][1] < hand0_lms[finger_tips[i] - 2][1]:
                    fingers_list.append(1)
                    fingers_string += "Open  "
                else:
                    fingers_list.append(0)
                    fingers_string += "Closed "

        if len(fingers_string) > 0:
            fingers_up_string = "Fingers: " + fingers_string
        else:
            fingers_up_string = "Show a Hand for Finger Modes!"

        cv2.putText(img, fingers_up_string, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        # ----------------------------------------------------------------------------------------------------------

        # exit the loop if fingers are in the specified position for some time ----------------------------------

        if len(fingers_list) > 0:
            peace_position = [0, 1, 1, 0, 0]  # specify the position that will exit the loop

            if fingers_list == peace_position:
                close_condition_number += 1
            else:
                close_condition_number = 0

            if close_condition_number == 30:
                condition = False

        # ----------------------------------------------------------------------------------------------------------

        # trigger speech recognition if fingers are in the specified position ----------------------------------------------

        recognizer = sr.Recognizer()
        text = ""

        if len(fingers_list) > 0:
            wolf_position = [0, 1, 0, 0, 1]  # specify the position that will trigger speech recognition

            if fingers_list == wolf_position:
                wolf_condition_number += 1
            else:
                wolf_condition_number = 0
            if wolf_condition_number == 30:
                with sr.Microphone() as source:
                    wolf_condition_number= 0
                    print("Speak Please")
                    audio = recognizer.listen(source,phrase_time_limit=5)
                    try:
                        text = recognizer.recognize_google(audio)
                        print("You said: "+text)
                    except:
                        print("Cannot recognize the audio")
                
        if "search on youtube" in text.lower():
            search_string = text.lower().split()[-1]
            search_on_youtube(search_string)
        
        if "exit" in text.lower():
            condition = False
        
        if "search on google" in text.lower():
            search_string = text.lower().split()[-1]
            search_on_google(search_string)
        # ----------------------------------------------------------------------------------------------------------

        cv2.imshow("Image", img)
        cv2.waitKey(1)

hand_tracker()