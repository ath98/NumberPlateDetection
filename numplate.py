from PIL.Image import new
import cv2, imutils, pytesseract
import numpy as np
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

img = cv2.imread('D:/Test IMG/test3.jpeg',cv2.IMREAD_COLOR)#Importing the image in our program
img = cv2.resize(img, (600,400) ) #Resizing recived image

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Coverting image to grayscale
gray = cv2.bilateralFilter(gray, 11, 16, 16) #Blurring out the not important parts

edged = cv2.Canny(gray, 20, 200) #Edge detection
cnt,news = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#Finding shapes in the Edged Image
img1 = img.copy()
#cv2.drawContours(img1,cnt,-1,(0,255,0),3)
#contours = imutils.grab_contours(contours)#Getting Contours together
cnt = sorted(cnt, key = cv2.contourArea, reverse = True)[:20] #Sorting the countours desending
cv2.drawContours(img1,cnt,-1,(0,255,0),3)
cv2.imshow("img1",img1)
cv2.waitKey(0)
screenCnt = None
idx=7
for c in cnt:#Looping thorugh all countours to check for rectrangles
    
    #(x,y,w,h)= cv2.boundingRect(c)
    #cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0),2)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.002 * peri, True)
    print(approx)
    if len(approx) == 4:
        screenCnt = approx
        x,y,w,h = cv2.boundingRect(c) #finds co-ordinates of the plate
        new_img=img[y:y+h,x:x+w]
        cv2.imwrite('./'+str(idx)+'.png',new_img) #stores the new image
        cv2.imshow('Test', new_img)
        cv2.waitKey(0)        
        idx+=1
        break

if screenCnt is None:
    detected = 0
    print ("No rectrangles detected")
else:
     detected = 1

if detected == 1:
    cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)#Drawing rectrsngle around the numberplate

mask = np.zeros(gray.shape,np.uint8) #Making a color mask
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)#Drawing rectrangle on a new image
new_image = cv2.bitwise_and(img,img,mask=mask)#Masking the rest of image 
cv2.imshow("Masked", new_image)
cv2.waitKey(0)

(x, y) = np.where(mask == 255)#Getting coordinates of the marked plate
(topx, topy) = (np.min(x), np.min(y))#Geting top X and Y Cordinates of the rectrangle
(bottomx, bottomy) = (np.max(x), np.max(y))#Geting bottom X and Y Cordinates of the rectrangle
Cropped = gray[topx:bottomx+1, topy:bottomy+1]#Getting all coordinates together ang cropping the image

text = pytesseract.image_to_string(Cropped, config='--psm 11')#Perfrming OCR in the cropped image
print("Number is:-",text)
img = cv2.resize(img,(500,300))
Cropped = cv2.resize(Cropped,(400,200))
cv2.imshow('Image',img)#Displaying orignal images
cv2.imshow('Numberplate',Cropped)#Displaying croped image

cv2.waitKey(0)
cv2.destroyAllWindows()