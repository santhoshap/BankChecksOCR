import cv2
from tkinter import filedialog
from tkinter import * 
import numpy as np
from PIL import Image
import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image
import pytesseract
import argparse
import cv2
import os

import pytesseract

from spellchecker import SpellChecker

spell = SpellChecker(distance=1)  # set at initialization

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

CLASSES = {1: {'id': 1, 'name': 'amountWord'}, 2: {'id': 2, 'name': 'amountNumber'}, 3: {'id': 3, 'name': 'date'}, 4: {'id': 4, 'name': 'validPeriod'}, 5: {'id': 5, 'name': 'ABArouting'}, 6: {'id': 6, 'name': 'signature'}}

#dictionaries
amountWords = {}
amountNumbers = {}
dates = {}
validPeriods = {}
ABAroutings = {}
signatures = {}

#colors
AMOUNTWORD_C = (24,24,169)
AMOUNTNUMBER_C = (233,82,82)
DATE_C = (112,8,8)
VALIDPERIOD_C = (193,24,193)
ABAROUTING_C = (63,169,9)
SIGNATURE_C = (21,218,249)

#detection filtering threshold values
AMOUNT_WORD_THRESH = 0.25
AMOUNT_NUMMBER_THRESH = 0.5
DATE_THRESH = 0.7
VALID_PERIOD_THRESH = 0.7
ABA_THRESH = 0.7
SIGNATURE_THRESH = 0.7


CHARACTERLIST_FOR_ALPHA_NUMERIC = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","Y","X","Y","Z",\
                "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","y","x","y","z",\
                "1","2","3","4","5","6","7","8","9","0"]

CHARACTERLIST_FOR_AMOUNT_WORD = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","Y","W","X","Y","Z",\
                "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","y","w","x","y","z",\
                "1","2","3","4","5","6","7","8","9","0","-","—","_"]

CHARACTERLIST_FOR_DATE = ["1","2","3","4","5","6","7","8","9","0","-","/","l"]

# CHARACTERLIST_FOR_DATE = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","Y","X","Y","Z",\
#                 "1","2","3","4","5","6","7","8","9","0","-","/","l"]

CHARACTERLIST_FOR_NUMERIC = ["1","2","3","4","5","6","7","8","9","0"]

CHARACTERLIST_FOR_AMOUNT_NUMBER = ["1","2","3","4","5","6","7","8","9","0",".",",","-"]

CHARACTERLIST_FOR_ALPHA_ONLY = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","Y","X","Y","Z",\
                "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","y","x","y","z"]

CHARACTERLIST = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","Y","X","Y","Z",\
                "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","y","x","y","z",\
                "1","2","3","4","5","6","7","8","9","0","-",".",",","/","_","="]
    
WORD_LIST_FOR_AMOUNT_WORD = ["and","eight","eighteen","eighty","eleven","fifteen","fifty","five","forty",\
                             "four","fourteen","hundred","hundredth","million","nine","nineteen","ninety",\
                                 "one","seven","seventeen","seventy","six","sixteen","sixty","ten","thirteen",\
                                     "thirty","thousand","three","trillion","twelve","twenty","two","dollars","cents"]



spell.word_frequency.load_words(WORD_LIST_FOR_AMOUNT_WORD)

def readImage():
    print("Select Original (Base) Image File: ")
    root = Tk().withdraw()
    fileName =  filedialog.askopenfilename(initialdir = "C:\",title = "Select Image file path",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    imageFromFile = cv2.imread(fileName,cv2.IMREAD_COLOR)
    print("Base Image Selected")
    return imageFromFile,fileName

def readTxtFile():
    print("Select text File: ")
    root = Tk().withdraw()
    fileName =  filedialog.askopenfilename(initialdir = "C:\",title = "Select Text file path",filetypes = (("text files","*.txt"),("all files","*.*")))
    #imageFromFile = cv2.imread(fileName,cv2.IMREAD_COLOR)
    print("bbox info text file Selected")
    return fileName

def read_class_names(class_file_name):
    # loads class name from a file
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def read_bbox_txt_file(bbox_file_name):
    listWords = None   

    listWordsArr = [] 
    
    with open(bbox_file_name, 'r') as filehandle:
        filecontents = filehandle.readlines()
        for line in filecontents:
            # remove linebreak which is the last character of the string
            line = line.rstrip()
            #print(line) 
            listWords = line.split(" ")
            #print(listWords)

            listWords = [int(listWords[0]),int(listWords[1]),int(listWords[2]),int(listWords[3]),float(listWords[4]),int(listWords[5])]

            listWordsArr.append(listWords)
                       
    print(listWordsArr)

    return listWordsArr


def convert_image_dtype(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

   
#pre processing functions

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((3,3),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((3,3),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#Split string into characters 
def split(word): 
    return [char for char in word]  


def is_valid_character(ch,object_type):
    if(object_type==1):
        if ch in CHARACTERLIST_FOR_AMOUNT_WORD:
            return 1
        else:
            return 0
    elif(object_type==2):
        if ch in CHARACTERLIST_FOR_AMOUNT_NUMBER:
            return 1
        else:
            return 0
    elif(object_type==3):
        if ch in CHARACTERLIST_FOR_DATE:
            return 1
        else:
            return 0    
    elif(object_type in (4,5,6)):
        if ch in CHARACTERLIST_FOR_NUMERIC:
            return 1
        else:
            return 0    
        

def extract_only_valid_characters(text,object_class):
    #filter non-alphanumeric chracters
    print('\n\nfiltering test')
    words = text.split()
    print(words)
    print('filtering test\n\n')
    
    final_text = ""
    for word in words:
        word_1 = split(word)
        print('word_1')
        print(word_1)
        new_word = ''
        for c in word_1:
            if(is_valid_character(c,object_class) == 1): # checking if indiviual chr is alphanumeric or not
                new_word = new_word + c    
        final_text = final_text+ " " +new_word
        
    #return read_text
    print('\n\nfinal_text test')
    print(final_text)
    print('final_text test\n\n')
        
    return final_text

def word_filtering_amount_word(text,object_class):
    text = text.replace("-", " ")
    words = text.split()
    full_text = ""
    misspelled = spell.unknown(words)
    print(misspelled)
    
    for word in misspelled:
        print(spell.correction(word))
        word1 = spell.correction(word)
        text.replace(word, word1)
        
    words = text.split()    
    for word in words:
        print(spell.correction(word))
        word = spell.correction(word)
        if word.lower() in WORD_LIST_FOR_AMOUNT_WORD:
            full_text = full_text + word+" "
    print(full_text)
    
    return full_text
    
    



def ocr_further_processing(img,roi_boundary,base_image_width,object_class): #image and a tuple

    (_, thresh) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    unsharp_image = thresh

    plt.figure()
    # plt.imshow(cv2.cvtColor(unsharp_image, cv2.COLOR_GRAY2RGB))
    plt.imshow(unsharp_image)
    plt.show()
        
    # kernel = np.ones((3,3),np.uint8)
    # eroded = cv2.erode(unsharp_image,kernel,iterations = 1) #white pixels gets eroded
    
    plt.figure()
    plt.imshow(thresh)
    plt.show()
    
    return thresh


def read_text_using_ocr(image,ROI,object_class = 3):  #{1:'amountWord'}, 2:'amountNumber'}, 3: 'date'}, 4:'validPeriod'}, 5: 'ABArouting'}, 6: 'signature'
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # (thresh, gray) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # (_, thresh) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    (y1,x1,y2,x2) = ROI
    bound = [y1,x1,y2,x2]
    
    # ROI_12 = thresh[y1:y2,x1:x2] #important
    ROI_12 = gray[y1:y2,x1:x2] #important
    
    # removing grid like distortions
    ROI_12 = ocr_further_processing(ROI_12,bound,image.shape[1],object_class)
	
    #custom_config = r'--oem 3 --psm 6'
    custom_config = r'--oem 3 --psm 12'
  
    if object_class == 1:
        # custom_config = r'--oem 3 --psm 12 -l eng'
        custom_config = r'--oem 3 --psm 12'
        read_text = pytesseract.image_to_string(ROI_12, config=custom_config)
    elif object_class ==2:
        # custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789,.$'
        custom_config = r'--oem 3  --psm 12'
        read_text = pytesseract.image_to_string(ROI_12, config=custom_config)
    elif object_class ==3:
        # custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789,.$'
        custom_config = r'--oem 3  --psm 12'
        read_text = pytesseract.image_to_string(ROI_12, config=custom_config)
    else:
        read_text = pytesseract.image_to_string(ROI_12, config=custom_config)
    # read_text = pytesseract.image_to_string(ROI_12, config=custom_config)
    
    final_text = extract_only_valid_characters(read_text,object_class)
    
    if object_class == 1:
        final_text = word_filtering_amount_word(final_text,1)
    
    pltImage = np.copy(ROI_12) 
    pltImage = cv2.cvtColor(pltImage, cv2.COLOR_BGR2RGB) 
    plt.figure()
   
    plt.imshow(pltImage)
    plt.text(0.1, 0.9,final_text, size=15, color='red')
    plt.show()
    
    imFileName = "test_{}.jpg".format(object_class)
    
    # cv2.imwrite(imFileName,pltImage)
    
    return final_text
    

   
def main():
    all_components = {}
    all_read_texts = {'signature': 'No Signature', 'date': '', 'ABArouting': '', 'amountNumber': '', 'validPeriod': '', 'amountWord': ''}
    lemos_index=0 #number of detected components
    
    
    print(CLASSES)
    
    test1, filename = readImage()
    bboxTextFile = readTxtFile() 
    
    #for the ease of ocr process make a permannet copy of the image
    test1copy = np.copy(test1)
    
    #extracting bboxes details into a list of lists
    bboxes_list = read_bbox_txt_file(bboxTextFile)
    print(bboxes_list)

    
    #producing dictionries for seperate classes
    for item in bboxes_list:
        category_index = int(item[5]+1)
        #CLASSES = {1: {'id': 1, 'name': 'amountWord'}, 2: {'id': 2, 'name': 'amountNumber'}, 3: {'id': 3, 'name': 'date'}, 4: {'id': 4, 'name': 'validPeriod'}, 5: {'id': 5, 'name': 'ABArouting'}, 6: {'id': 6, 'name': 'signature'}}
        category_index = item[5]+1   # 0 based nisa
        #ymin,xmin,ymax,xmax,probabality_score,category_index,category
        coord_and_class = [item[0],item[1],item[2],item[3],item[4],category_index,CLASSES[category_index]['name']]
                
        #--- amountWord --- #
        if category_index == 1: #amountWord
            amountWord_id = 'aW_1'
            #filtering the detected bounding boxes using a pre deifened threshold value
            if float(item[4])>AMOUNT_WORD_THRESH:
                if amountWords=={}:
                    amountWords.update({amountWord_id:coord_and_class})
                    all_components.update({'amountWord':coord_and_class})
                    lemos_index+=1
                    print(amountWords)
                else:
                    if float(item[4])>float(amountWords[amountWord_id][4]):
                        print("new amountWord")
                        amountWords.update({amountWord_id:coord_and_class})
                        all_components.update({'amountWord':coord_and_class})
                        lemos_index+=1
                        
        
        #--- amountNumber --- #
        if category_index == 2: #amountNumber
            amountNumber_id = 'aN_1'
            #filtering the detected bounding boxes using a pre deifened threshold value
            if float(item[4])>AMOUNT_NUMMBER_THRESH:
                if amountNumbers=={}:
                    amountNumbers.update({amountNumber_id:coord_and_class})
                    all_components.update({'amountNumber':coord_and_class})
                    lemos_index+=1
                    print(amountNumbers)
                else:
                    if float(item[4])>float(amountNumbers[amountNumber_id][4]):
                        print("new amountWord")
                        amountNumbers.update({amountNumber_id:coord_and_class})
                        all_components.update({'amountNumber':coord_and_class})
                        lemos_index+=1
                        
        #--- date --- #
        if category_index == 3: #date
            date_id = 'dt_1'
            #filtering the detected bounding boxes using a pre deifened threshold value
            if float(item[4])>DATE_THRESH:
                if dates=={}:
                    dates.update({date_id:coord_and_class})
                    all_components.update({'date':coord_and_class})
                    lemos_index+=1
                    print(dates)
                else:
                    if float(item[4])>float(dates[date_id][4]):
                        print("new amountWord")
                        dates.update({date_id:coord_and_class})
                        all_components.update({'date':coord_and_class})
                        lemos_index+=1
        
        #--- validPeriod --- #
        if category_index == 4: #validPeriod
            validPeriod_id = 'vp_1'
            #filtering the detected bounding boxes using a pre deifened threshold value
            if float(item[4])>VALID_PERIOD_THRESH:
                if validPeriods=={}:
                    validPeriods.update({validPeriod_id:coord_and_class})
                    all_components.update({'validPeriod':coord_and_class})
                    lemos_index+=1
                    print(validPeriods)
                else:
                    if float(item[4])>float(validPeriods[validPeriod_id][4]):
                        print("new amountWord")
                        validPeriods.update({validPeriod_id:coord_and_class})
                        all_components.update({'validPeriod':coord_and_class})
                        lemos_index+=1
                        
        #--- ABArouting --- #
        if category_index == 5: #ABArouting
            ABArouting_id = 'aba_1'
            #filtering the detected bounding boxes using a pre deifened threshold value
            if float(item[4])>ABA_THRESH:
                if ABAroutings=={}:
                    ABAroutings.update({ABArouting_id:coord_and_class})
                    all_components.update({'ABArouting':coord_and_class})
                    lemos_index+=1
                    print(ABAroutings)
                else:
                    if float(item[4])>float(ABAroutings[ABArouting_id][4]):
                        print("new ABArouting")
                        ABAroutings.update({ABArouting_id:coord_and_class})
                        all_components.update({'ABArouting':coord_and_class})
                        lemos_index+=1
                        
        #--- signature --- #
        if category_index == 6: #signature
            signature_id = 'sg_1'
            #filtering the detected bounding boxes using a pre deifened threshold value
            if float(item[4])>SIGNATURE_THRESH:
                if signatures=={}:
                    signatures.update({signature_id:coord_and_class})
                    all_components.update({'signature':coord_and_class})
                    lemos_index+=1
                    print(signatures)
                else:
                    if float(item[4])>float(signatures[signature_id][4]):
                        print("new amountWord")
                        signatures.update({signature_id:coord_and_class})
                        all_components.update({'signature':coord_and_class})
                        lemos_index+=1
        
    #print(mileposts.get('mp_1')[3])
    #print(mileposts['mp_1'][3])
    
    print("\n\nprinting classes seperately\n")
                
    print(amountWords)
    print(amountNumbers)
    print(dates)
    print(validPeriods)
    print(ABAroutings)
    print(signatures)
    
    print("\n\nAll Components\n")
    print(all_components) 
    
    print("\n\nNumber of items detected is --- {}\n".format(lemos_index))     
    
    
    for component in all_components.values():
        ##tensorflow provides cordinates of the format ymin=bbox[0],xmin,ymax,xmax
        (y1, x1), (y2, x2) = (component[0], component[1]), (component[2], component[3])
        ##cv2.rectangle(test1, (x1, y1), (x2, y2), (0,100,200), -1)
        
        print(component)
               
        if(component[5]==1):
            cv2.rectangle(test1, (x1, y1), (x2, y2), AMOUNTWORD_C, 4)#b,g,r
            cv2.putText(test1,'{} : {}%'.format(component[6],round(component[4]*100,2)),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,AMOUNTWORD_C,2)
            read_text_2 = read_text_using_ocr(test1copy,[y1, x1, y2, x2],1)
            read_text_2 = read_text_2.strip().replace("-"," ").replace("_", " ").replace("—", " ")
            all_read_texts.update({'amountWord':read_text_2})
            print("Read text amountWord: --- {}\n".format(read_text_2))
                        
        elif(component[5]==2):
            cv2.rectangle(test1, (x1, y1), (x2, y2), AMOUNTNUMBER_C, 4)
            cv2.putText(test1,'{} : {}%'.format(component[6],round(component[4]*100,2)),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,AMOUNTNUMBER_C,2)
            read_text_2 = read_text_using_ocr(test1copy,[y1, x1, y2, x2],2)
            read_text_2 = read_text_2.strip().replace(" ", "").replace("-",".")
            all_read_texts.update({'amountNumber':read_text_2})
            print("Read text amountNumber: --- {}\n".format(read_text_2))
            
        elif(component[5]==3):
            cv2.rectangle(test1, (x1, y1), (x2, y2), DATE_C, 4)
            cv2.putText(test1,'{} : {}%'.format(component[6],round(component[4]*100,2)),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,DATE_C,2)
            read_text_2 = read_text_using_ocr(test1copy,[y1, x1, y2, x2],3)
            read_text_2 = read_text_2.strip().replace(" ", "").replace("l", "1")
            
            if len(read_text_2) == 10:
                if read_text_2[2] == '1':
                    read_text_2 = read_text_2[:2] + '/' + read_text_2[2 + 1:]
                if read_text_2[5] == '1':
                    read_text_2 = read_text_2[:5] + '/' + read_text_2[5 + 1:]
            
            all_read_texts.update({'date':read_text_2})
            print("Read text date: --- {}\n".format(read_text_2))
            
        elif(component[5]==4):
            cv2.rectangle(test1, (x1, y1), (x2, y2), VALIDPERIOD_C, 4)
            cv2.putText(test1,'{} : {}%'.format(component[6],round(component[4]*100,2)),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,VALIDPERIOD_C,2)
            read_text_2 = read_text_using_ocr(test1copy,[y1, x1, y2, x2],4)
            read_text_2 = read_text_2.strip()
            all_read_texts.update({'validPeriod':read_text_2})
            print("Read text validPeriod: --- {}\n".format(read_text_2))
            
        elif(component[5]==5):
            cv2.rectangle(test1, (x1, y1), (x2, y2), ABAROUTING_C, 4)
            cv2.putText(test1,'{} : {}%'.format(component[6],round(component[4]*100,2)),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,ABAROUTING_C,2)
            read_text_2 = read_text_using_ocr(test1copy,[y1, x1, y2, x2],5)
            read_text_2 = read_text_2.strip().replace(" ", "")
            all_read_texts.update({'ABArouting':read_text_2})
            print("Read text ABArouting: --- {}\n".format(read_text_2))
            
        elif(component[5]==6):
            cv2.rectangle(test1, (x1, y1), (x2, y2), SIGNATURE_C, 4)
            cv2.putText(test1,'{} : {}%'.format(component[6],round(component[4]*100,2)),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,SIGNATURE_C,2)

            all_read_texts.update({'signature':"Signature Exists"})
            # print("Read text signature: --- {}\n".format(read_text_2))            

    #result_final = {}
            

        
                   
    
    ##new file name for the output image
    filename1 = filename[:filename.find('.')]+'_Lemos_1'+filename[filename.find('.'):] 
    
    ##txt file name
    filename1_txt = filename[:filename.find('.')]+'_Lemos_read_text'+'.txt'
    
    ##csv file name
    #filename1_csv = filename[:filename.find('.')]+'_LemosTest_3_further_ii'+'.csv'
    
    print(filename1)
    cv2.imwrite(filename1,test1)
    
    
    print("\nresult_final\n")
    print(all_read_texts)
    print("\nresult_final\n")    
    
    #print("\nresult_final\n")
    #print(result_final)    
    #print("\nresult_final\n")
    with open(filename1_txt, 'w') as filehandle:
            
        textLine = '[1] amountWord: {}'.format(all_read_texts['amountWord'])
        filehandle.write('%s\n' % textLine)
        
        textLine = '[2] amountNumber: {}'.format(all_read_texts['amountNumber'])
        filehandle.write('%s\n' % textLine)
        
        textLine = "[3 i] date: {}".format(all_read_texts['date'])
        filehandle.write('%s\n' % textLine)
        
        textLine = '[4] validPeriod: {}'.format(all_read_texts['validPeriod'])
        filehandle.write('%s\n' % textLine)
        
        textLine = '[5] ABArouting: {}'.format(all_read_texts['ABArouting'])
        filehandle.write('%s\n' % textLine)
        
        textLine = '[6] signature: {}'.format(all_read_texts['signature'])
        filehandle.write('%s\n' % textLine)
    
  


main()
