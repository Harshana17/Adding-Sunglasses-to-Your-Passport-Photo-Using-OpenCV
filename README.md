# Adding-Sunglasses-to-Your-Passport-Photo-Using-OpenCV
## AIM

To develop an image processing application using OpenCV that detects the face region and overlays a transparent sunglass image on the eye region of a passport-size photograph.

## TECHNOLOGIES USED

Python

OpenCV for image processing

Numpy for array manipulations

## ALGORITHM

Import the required libraries (OpenCV, NumPy, and Matplotlib).

Load the passport-size face image.

Load the sunglasses image with its alpha (transparency) channel.

Resize the sunglasses image to match the eye region of the face.

Separate the sunglasses image into:

Color (BGR) channels

Alpha mask (transparency channel)

Create a 3-channel mask from the alpha channel.

Select the eye Region of Interest (ROI) from the face image.

Convert the ROI and sunglasses image to floating-point format.

Apply the transparency mask to remove the background from the eye region.

Blend the sunglasses image with the eye region using alpha blending.

Replace the original eye region with the blended image.

Display both the original image and the final image with sunglasses.

## PROGRAM
```
# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the Face Image
faceImage = cv2.imread("C:/Users/admin/OneDrive/Desktop/DIP/img1.png")
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")
faceImage.shape
#resized_faceImage.shape
faceImage.shape
# Load the Sunglass image with Alpha channel
# (http://pluspng.com/sunglass-png-1104.html)
glassPNG = cv2.imread("C:/Users/admin/OneDrive/Desktop/DIP/sunglass.jpg",-1)
plt.imshow(glassPNG[:,:,::-1]);plt.title("glassPNG")
# Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG,(190,50))
print("image Dimension ={}".format(glassPNG.shape))
# Separate the Color and alpha channels
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,2]
# Display the images for clarity
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(glassBGR[:,:,::-1]);plt.title('Sunglass Color channels');
plt.subplot(122);plt.imshow(glassMask1,cmap='gray');plt.title('Sunglass Alpha channel');
# Make a copy
#faceWithGlassesNaive = resized_faceImage.copy()
faceWithGlassesNaive = faceImage.copy()

# Replace the eye region with the sunglass image
faceWithGlassesNaive[135:185,110:300]=glassBGR

plt.imshow(faceWithGlassesNaive[...,::-1])
# Make the dimensions of the mask same as the input image.
# Since Face Image is a 3-channel image, we create a 3 channel image for the mask
glassMask = cv2.merge((glassMask1,glassMask1,glassMask1))

# Make the values [0,1] since we are using arithmetic operations
glassMask = np.uint8(glassMask/255)

# Make a copy
faceWithGlassesArithmetic = faceImage.copy()

# Get the eye region from the face image
eyeROI= faceWithGlassesArithmetic[135:185,110:300]

# Use the mask to create the masked eye region
maskedEye = cv2.multiply(eyeROI,(1-  glassMask ))

# Use the mask to create the masked sunglass region
maskedGlass = cv2.multiply(glassBGR,glassMask)

# Combine the Sunglass in the Eye Region to get the augmented image
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)

# Display the intermediate results
plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(maskedEye[...,::-1]);plt.title("Masked Eye Region")
plt.subplot(132);plt.imshow(maskedGlass[...,::-1]);plt.title("Masked Sunglass Region")
plt.subplot(133);plt.imshow(eyeRoiFinal[...,::-1]);plt.title("Augmented Eye and Sunglass")
# Replace the eye ROI with the output from the previous section
faceWithGlassesArithmetic[135:185,110:300]=eyeRoiFinal

# Display the final result
plt.figure(figsize=[10,10]);
plt.subplot(121);plt.imshow(faceImage[:,:,::-1]); plt.title("Original Image");
plt.subplot(122);plt.imshow(faceWithGlassesArithmetic[:,:,::-1]);plt.title("With Sunglasses");
```
<img width="402" height="547" alt="image" src="https://github.com/user-attachments/assets/64e37db9-61a1-45b5-b807-9f7ca568cbd6" />
<img width="725" height="541" alt="image" src="https://github.com/user-attachments/assets/dbac8de2-9629-4045-8081-d22ff659e928" />
<img width="1380" height="256" alt="image" src="https://github.com/user-attachments/assets/343cc3ae-3032-4f50-b4cb-caca3703ebfe" />
<img width="421" height="555" alt="image" src="https://github.com/user-attachments/assets/21087f6c-d1d5-4b57-8c04-2dc98fb65947" />
<img width="1370" height="197" alt="image" src="https://github.com/user-attachments/assets/a2424802-934e-4d72-b93f-e58c8a406888" />
<img width="1056" height="717" alt="image" src="https://github.com/user-attachments/assets/2ef60c99-513c-409c-a95b-e4733bc40ccc" />

## RESULT
The program was successfully implemented using Python and OpenCV. The sunglasses image was resized, blended using alpha masking, and accurately placed over the eye region of the passport-size photograph. The final output produced a natural-looking image with sunglasses while preserving the original facial features.

