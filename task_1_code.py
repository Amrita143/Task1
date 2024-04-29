import cv2
from rembg import remove
import numpy as np
from PIL import Image # Not required
import matplotlib.pyplot as plt # Not required

# Specify the input image path
input_image_path = '1.jpg'

def resize_image(image, max_width=800):
    # Calculate the ratio
    h, w = image.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_dim = (max_width, int(h * ratio))
        resized = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
        return resized
    return image


def main():
    # Read the image
    img = cv2.imread(input_image_path)
    resized_img = resize_image(img)  # Resize the image to fit on the screen
    # cv2.imshow('Original Image', resized_img)
    # cv2.waitKey(0)
    while True:
        cv2.destroyAllWindows()
        img = resized_img.copy()
        r = cv2.selectROI(img)
            
        cv2.destroyAllWindows()

        # Check if any ROI was selected
        if r[2] == 0 or r[3] == 0:  # ROI width or height is 0 if nothing is selected
            print("No ROI selected. Exiting program.")
            break  # Exit the loop and end the program
            
        # Crop image 
        cropped_image = img[int(r[1]):int(r[1]+r[3]),  
                            int(r[0]):int(r[0]+r[2])] 

        # Remove the background
        output = remove(cropped_image)

        # Save and display the output image
        # output_path = 'output.jpg'
        # cv2.imwrite(output_path, output)
        # cv2.imshow('Output Image', output)

        # Convert the image to grayscale
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

        # plt.imshow(gray, cmap='gray')
        # plt.show()
        ret, thresh = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY) # threshold value is very low to get high contrast
        # plt.imshow(thresh, cmap='gray')
        # plt.show()
        # cv2.waitKey(0)

        # Find the contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # image_counturs = cv2.drawContours(output.copy(), contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
        # plt.imshow(cv2.cvtColor(image_counturs, cv2.COLOR_BGR2RGB))
        # plt.show()
        # Adjust contour coordinates and draw on the original image
        for cnt in contours:
            # Shift contour coordinates
            cnt += (int(r[0]), int(r[1]))
            cv2.drawContours(img, [cnt], 0, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the final result
        cv2.imshow('Final Result', img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):  # Quit the program
            break
        elif key == ord('c'):
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()

