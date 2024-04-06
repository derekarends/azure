import os
import requests
import cv2
import numpy as np

from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, OperationStatusType
from dotenv import load_dotenv
load_dotenv()

ENDPOINT = os.environ['AZURE_FACE_ENDPOINT']
KEY = os.environ['AZURE_FACE_KEY']

# Create an authenticated FaceClient.
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

# Detect a face in an image that contains a single face
single_face_image_url = 'https://raw.githubusercontent.com/Microsoft/Cognitive-Face-Windows/master/Data/detection1.jpg'
single_image_name = os.path.basename(single_face_image_url)
detected_faces = face_client.face.detect_with_url(url=single_face_image_url)
if not detected_faces:
    raise Exception('No face detected from image {}'.format(single_image_name))

# Convert width height to a point in a rectangle
def getRectangle(faceDictionary):
    rect = faceDictionary.face_rectangle
    left = rect.left
    top = rect.top
    bottom = left + rect.height
    right = top + rect.width
    return ((left, top), (bottom, right))


# Download the image from the url
response = requests.get(single_face_image_url)
nparr = np.frombuffer(response, np.uint8)
image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# For each face returned use the face rectangle and draw a red box.
print('Drawing rectangle around face... see popup for results.')
for face in detected_faces:
    top_left, bottom_right = getRectangle(face)
    cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)

# Display the image in the users default image browser.
cv2.imshow('Image with Bounding Box', image)
cv2.waitKey(0)
cv2.destroyAllWindows()