import os

from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from dotenv import load_dotenv
load_dotenv()

endpoint = os.environ["VISION_ENDPOINT"]
prediction_key = os.environ["VISION_PREDICTION_KEY"]

prediction_credentials = ApiKeyCredentials(in_headers={
    "Prediction-key": prediction_key
})
predictor = CustomVisionPredictionClient(endpoint, prediction_credentials)


def classify():
    print("\n -- classify")
    classify_project_id = os.environ["CUSTOM_VISION_CLASSIFY_PROJECT_ID"]
    publish_iteration_name = "Iteration1"

    with open(os.path.join("./images/LittleYellowCar.png"), "rb") as image_contents:
        results = predictor.classify_image(classify_project_id, publish_iteration_name, image_contents.read())

        for prediction in results.predictions:
            print("\t" + prediction.tag_name +
                  ": {0:.2f}%".format(prediction.probability * 100))


def detect():
    print("\n -- detect")
    detect_project_id = os.environ["CUSTOM_VISION_DETECT_PROJECT_ID"]
    publish_iteration_name = "Iteration1"

    with open(os.path.join("./images/LittleYellowCar.png"), "rb") as image_contents:
        results = predictor.detect_image(detect_project_id, publish_iteration_name, image_contents.read())

    for prediction in results.predictions:
        print(
            "\t" + prediction.tag_name + ": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f}, bbox.width = {3:.2f}, bbox.height = {4:.2f}".format(
                prediction.probability * 100, prediction.bounding_box.left, prediction.bounding_box.top,
                prediction.bounding_box.width, prediction.bounding_box.height))


if __name__ == "__main__":
    classify()
