"""
USAGE:
    python ai_image.py

    Set the environment variables with your own values before running the sample:
    1) AZURE_AI_SERVICES_URL - the endpoint to your azure ai resource.
    2) AZURE_AI_SERVICES_KEY - your azure ai API key


    VisualFeatures.TAGS: Identifies tags about the image, including objects, scenery, setting, and actions
    VisualFeatures.OBJECTS: Returns the bounding box for each detected object
    VisualFeatures.CAPTION: Generates a caption of the image in natural language
    VisualFeatures.DENSE_CAPTIONS: Generates more detailed captions for the objects detected
    VisualFeatures.PEOPLE: Returns the bounding box for detected people
    VisualFeatures.SMART_CROPS: Returns the bounding box of the specified aspect ratio for the area of interest
    VisualFeatures.READ: Extracts readable text

    pip install azure-ai-vision-imageanalysis
"""
import os
import cv2

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
load_dotenv()

# Get the endpoint and key from the environment
endpoint = os.environ["AZURE_AI_SERVICES_URL"]
key = os.environ["AZURE_AI_SERVICES_KEY"]


def get_objects():
    """
    Load an image and identify objects with tags in the image
    This will draw bounding boxes around the objects in the image
    """
    print("\n -- objects")

    image_path = "./images/ButterflyWithMoon.webp"
    with open(image_path, "rb") as image_file:
        file_bytes = image_file.read()

    client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    result = client.analyze(
        image_data=file_bytes,
        visual_features=[VisualFeatures.OBJECTS]
    )

    image = cv2.imread(image_path)

    values = result.objects.values()
    for value in values:
        for obj in value:
            bbox = obj.bounding_box
            cv2.putText(
                image,
                str(obj.tags),
                (bbox['x'], bbox['y'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (36, 255, 12), 2
            )
            cv2.rectangle(
                image,
                (bbox['x'], bbox['y']),
                (bbox['x'] + bbox['w'],
                 bbox['y'] + bbox['h']),
                (0, 255, 0), 2
            )

    cv2.imshow('Image with Bounding Box', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_caption():
    """
    Generate a caption for an image
    """
    print("\n -- caption")

    image_path = "./images/ButterflyWithMoon.webp"
    with open(image_path, "rb") as image_file:
        file_bytes = image_file.read()

    client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    result = client.analyze(
        image_data=file_bytes,
        visual_features=[VisualFeatures.CAPTION, VisualFeatures.DENSE_CAPTIONS],
    )

    image = cv2.imread(image_path)

    dense_captions = result['denseCaptionsResult']
    for caption in dense_captions['values']:
        bbox = caption['boundingBox']
        cv2.rectangle(image, (bbox['x'], bbox['y']),
                      (bbox['x'] + bbox['w'],
                       bbox['y'] + bbox['h']),
                      (0, 255, 0), 2)
        cv2.putText(image, caption['text'], (bbox['x'], bbox['y'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow('Image with Bounding Box and Text', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Caption: {result['captionResult']}")


def get_words():
    """
    Load an image and extract readable text
    """
    print("\n -- read")

    image_path = "./images/SpanishSign.png"
    with open(image_path, "rb") as image_file:
        file_bytes = image_file.read()

    client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    result = client.analyze(
        image_data=file_bytes,
        visual_features=[VisualFeatures.READ],
    )

    image = cv2.imread(image_path)

    for block in result.read.blocks:
        for line in block.lines:
            line_box = line['boundingPolygon']
            top_left = (line_box[0]['x'], line_box[0]['y'])
            bottom_right = (line_box[2]['x'], line_box[2]['y'])
            cv2.rectangle(
                image,
                top_left,
                bottom_right,
                (0, 0, 255), 2
            )
            for word in line.words:
                word_box = word['boundingPolygon']
                top_left = (word_box[0]['x'], word_box[0]['y'])
                bottom_right = (word_box[2]['x'], word_box[2]['y'])
                cv2.rectangle(
                    image,
                    top_left,
                    bottom_right,
                    (0, 255, 0), 2
                )
                print(f"Word: {word['text']}")
                print(f"Confidence: {word['confidence']}")

    cv2.imshow('Image with Bounding Box', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # get_objects()
    # get_caption()
    get_words()
