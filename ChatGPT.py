
import cv2
from skimage import io
from skimage.metrics import structural_similarity as ssim
import numpy as np

def get_edges(image_path):
    # read image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    return edges


edges = get_edges("path/to/image.jpg")

# load the puzzle piece image
piece = cv2.imread("path/to/puzzle_piece.jpg", cv2.IMREAD_GRAYSCALE)

# load the full puzzle image
puzzle = cv2.imread("path/to/puzzle.jpg", cv2.IMREAD_GRAYSCALE)

# create a ORB object
orb = cv2.ORB_create()

# find keypoints and descriptors for the puzzle piece
kp1, des1 = orb.detectAndCompute(piece, None)

# find keypoints and descriptors for the full puzzle
kp2, des2 = orb.detectAndCompute(puzzle, None)

# create a Brute-Force Matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# match the descriptors
matches = bf.match(des1, des2)

# sort the matches based on the distance
matches = sorted(matches, key=lambda x: x.distance)

# draw the matches
img_matches = cv2.drawMatches(piece, kp1, puzzle, kp2, matches[:10], None, flags=2)

# show the matches
cv2.imshow("Matches", img_matches)
cv2.waitKey(0)


def similarity():
    

    # Load the reference image
    ref_image = io.imread('reference.jpg')

    # Load the list of images
    images = [io.imread(f'image_{i}.jpg') for i in range(1, 10)]

    # Compute the SSIM similarity scores for each image
    similarity_scores = [ssim(ref_image, img) for img in images]

    # Find the index of the most similar image
    most_similar_index = np.argmax(similarity_scores)

