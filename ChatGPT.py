import cv2
import numpy as np

from skimage import io
import os
from os.path import join
from skimage.metrics import structural_similarity as ssim


def get_edges(image_path):
    # read image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    return edges




def similarity():
    
    filenames = os.listdir('output_folder')
    filenames.sort()
    # Load the reference image
    input_image = join('output_folder', 'IMG_001_1_7_3_int.jpg')
    ref_image = io.imread(input_image)
    input_image1 = join('output_folder', 'IMG_001_1_7_1_int.jpg')
    ref_image1 = io.imread(input_image1)
    s1 = ssim(ref_image, ref_image1)
    # Load the list of images
    images = [io.imread(join('output_folder', i)) for i in filenames]
    
    # Compute the SSIM similarity scores for each image
    similarity_scores = [ssim(ref_image, img) for img in images]

    # Find the index of the most similar image
    most_similar_index = np.argmax(similarity_scores)


def similarity2():
    method = cv2.TM_SQDIFF_NORMED
    filenames = os.listdir('output_folder')
    filenames.sort()
    # Load the reference image
    input_image = join('output_folder', 'IMG_001_1_7_3_int.jpg')
    ref_image = io.imread(input_image)
    ref_image = cv2.flip(ref_image, 1)
    # Load the list of images
    images = [io.imread(join('output_folder', i)) for i in filenames]
    
    # Compute the SSIM similarity scores for each image
    similarity_scores = [cv2.matchTemplate(ref_image,img,  method) for img in images]




# function to check similarity between two images
def check_similarity(img1, img2):
    # convert the images to grayscale
    grayA = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))


def check(small_image, large_image):
    method = cv2.TM_SQDIFF_NORMED
    result = cv2.matchTemplate(large_image,small_image,  method)

    # We want the minimum squared difference
    mn,_,mnLoc,_ = cv2.minMaxLoc(result)

    # Draw the rectangle:
    # Extract the coordinates of our best match
    MPx,MPy = mnLoc

    # Step 2: Get the size of the template. This is the same size as the match.
    trows,tcols = small_image.shape[:2]

    # Step 3: Draw the rectangle on large_image
    cv2.rectangle(large_image, (MPx,MPy),(MPx+tcols,MPy+trows),(0,0,255),2)

    # Display the original image with the rectangle around the match.
    cv2.imshow('output',large_image)

    # The image is only displayed if we call this
    cv2.waitKey(0)




def compare_images(src_img_filename: str, image_filenames: list,folder_name: str):
    # Load the source image

    src_img = cv2.imread(join(folder_name, src_img_filename))
    # cv2.imshow("test1",src_img)
    # cv2.waitKey(0)
    # src_img = cv2.flip(src_img, 1)
    # cv2.imshow("test1",src_img)
    # cv2.waitKey(0)
    # Create a list to store the similarity scores
    scores = []

    # Loop over the list of images
    for img in image_filenames:
        # Load the image
        image = cv2.imread(join(folder_name, img))

        # Resize the images to have the same size
        src_img = cv2.resize(src_img, (500, 500))
        image = cv2.resize(image, (500, 500))

        # Convert the images to grayscale
        # src_img_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute the mean squared error between the two images
        # mse = np.mean((src_img_gray - image_gray) ** 2)
        mse = np.mean((src_img - image) ** 2)

        # Store the MSE score in the list of scores
        scores.append(mse)
       

    # Find the index of the image with the lowest MSE score
    # best_match_index = scores.index(min(scores))

    # Return the name of the most similar image
    # return image_filenames[best_match_index]
    return scores


def similarity1(image_name: str,folder_name: str):
    filenames = os.listdir(folder_name)
    filenames.sort()
    ret = compare_images(image_name, filenames,folder_name)
    return ret


def main():
    # similarity()
    # small_image = cv2.imread(join('output_folder', 'IMG_001_1_7_3_int.jpg'))
    # flipHorizontal = cv2.flip(small_image, 1)
    # large_image = cv2.imread(join('pieces', 'IMG_001_1_6.jpg'))
    # check(flipHorizontal, large_image)
    res = similarity1('IMG_001_1_7_3_int.jpg','sides')


if __name__ == "__main__":
    main()