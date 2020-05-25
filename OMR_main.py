import cv2
import numpy as np
import utlis

########################
path="1.JPG"
widthImg=450
heightImg=450
questions=5
choices=5
ans=[]

########################



img=cv2.imread(path)

########### PREPROCESSING
img=cv2.resize(img,(widthImg,heightImg))
imgContours=img.copy()
imgFinal=img.copy()
imgBiggestContours=img.copy()
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur=cv2.GaussianBlur(imgGray,(5,5),1)
imgCanny=cv2.Canny(imgBlur,10,50)

try:
    ########## FINDING ALL CONTOURS
    contours,hierarchy=cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgContours,contours,-1,(0,255,0),10)

    ######## FIND RECTANGLES
    rectCon=utlis.rectContour(contours)
    biggestContours=utlis.getCornerPoints(rectCon[0])
    gradePoints=utlis.getCornerPoints(rectCon[1])
    # print(biggestContours,gradePoint)
    # print(len(biggestContours))

    if biggestContours.size !=0 and gradePoints.size !=0:
        cv2.drawContours(imgBiggestContours,biggestContours,-1,(0,255,0),20)
        cv2.drawContours(imgBiggestContours, gradePoints, -1, (255, 0, 0), 20)
        # reorder the cordinate points of req rect for finding out origin point and to be used without confusion in future
        biggestContours=utlis.reorder(biggestContours)
        gradePoints=utlis.reorder(gradePoints)

        pt1=np.float32(biggestContours)
        pt2=np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
        matrix=cv2.getPerspectiveTransform(pt1,pt2)
        imgWrapColored=cv2.warpPerspective(img,matrix,(widthImg,heightImg))

        ptG1 = np.float32(gradePoints)
        ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])  # here heiht/width can be any any val as desire
        matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
        imgGardeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))
        # cv2.imshow("Grade Display",imgGardeDisplay)

        # APPLY THRESHOLD
        imgWrapGray=cv2.cvtColor(imgWrapColored,cv2.COLOR_BGR2GRAY)
        imgThresh=cv2.threshold(imgWrapGray,170,255,cv2.THRESH_BINARY_INV)[1]

        boxes=utlis.splitBoxes(imgThresh)
        # cv2.imshow("boxes",boxes[24])
        # print(cv2.countNonZero(boxes[1]),cv2.countNonZero(boxes[2]))

        countR = 0
        countC = 0
        myPixelVal = np.zeros((questions, choices))  # TO STORE THE NON ZERO VALUES OF EACH BOX
        for image in boxes:
            # cv2.imshow(str(countR)+str(countC),image)
            totalPixels = cv2.countNonZero(image)
            myPixelVal[countR][countC] = totalPixels
            countC += 1
            if (countC == choices):
                countC = 0
                countR += 1
        # print(myPixelVal)

        # FIND THE USER ANSWERS AND PUT THEM IN A LIST

        myIndex = []
        for x in range(0, questions):
            arr = myPixelVal[x]
            myIndexVal = np.where(arr == np.amax(arr))
            myIndex.append(myIndexVal[0][0])
        # print("USER ANSWERS",myIndex)

        # COMPARE THE VALUES TO FIND THE CORRECT ANSWERS
        print("Default we are taking 5 questions and 5 options for each question.")
        print("Enter the correct ans for each ques {here 0 means 1st option is correct and so on}.. ")
        for i in range(questions):
            anss = int(input())
            ans.append(anss)
        print("Correct option of each questions are", ans)
        grading = []
        for x in range(0, questions):
            if ans[x] == myIndex[x]:
                grading.append(1)
            else:
                grading.append(0)
        # print("GRADING",grading)
        score = (sum(grading) / questions) * 100  # FINAL GRADE
        # print("SCORE",score)

        # DISPLAYING ANSWER

        imgResult=imgWrapColored.copy()
        imgResult=utlis.showAnswers(imgResult,myIndex,grading,ans,questions,choices)   # DRAW DETECTED ANSWERS

        imgRawDrawing = np.zeros_like(imgWrapColored)  # NEW BLANK IMAGE WITH WARP IMAGE SIZE
        imgRawDrawing = utlis.showAnswers(imgRawDrawing, myIndex, grading, ans, questions, choices)  # DRAW ON NEW IMAGE

        invMatrix = cv2.getPerspectiveTransform(pt2, pt1)   # INVERSE TRANSFORMATION MATRIX
        imgInvWrap = cv2.warpPerspective(imgRawDrawing, invMatrix, (widthImg, heightImg))    # INV IMAGE WARP

        # DISPLAY GRADE
        imgRawGrade = np.zeros_like(imgGardeDisplay, np.uint8)  # NEW BLANK IMAGE WITH GRADE AREA SIZE
        cv2.putText(imgRawGrade, str(int(score)) + "%", (70, 100)
                        , cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 3)  # ADD THE GRADE TO NEW IMAGE
        invMatrixG = cv2.getPerspectiveTransform(ptG2, ptG1)  # INVERSE TRANSFORMATION MATRIX
        imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))  # INV IMAGE WARP

        imgFinal=cv2.addWeighted(imgFinal,1,imgInvWrap,1,0)
        imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)


    imgBlank=np.zeros_like(img)
    imageArray=([img,imgGray,imgBlur,imgCanny],
                [imgContours,imgBiggestContours,imgWrapColored,imgThresh],
                [imgResult,imgRawDrawing,imgInvWrap,imgFinal])
except:
    imgBlank = np.zeros_like(img)
    imageArray = ([img, imgGray, imgBlur, imgBlank],
                      [imgBlank, imgBlank, imgBlank, imgBlank],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

# LABELS FOR DISPLAY
lables = [["Original","Gray","Blur","Canny"],
                  ["Contours","Biggest Contour","Wrap","Threshold",],
              ["Result","Raw Draw","Inv Wrap","Final"]]

imgStacked=utlis.stackImages(imageArray,0.5,lables)

cv2.imshow("Stacked Image",imgStacked)
cv2.imshow("Final Image",imgFinal)

cv2.waitKey(0)