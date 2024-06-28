import os
import cv2

from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from dotenv import load_dotenv
load_dotenv()

endpoint = os.environ["AZURE_AI_SERVICES_URL"]
prediction_key = os.environ["AZURE_AI_SERVICES_KEY"]

prediction_credentials = ApiKeyCredentials(in_headers={
    "Prediction-key": prediction_key
})
predictor = CustomVisionPredictionClient(endpoint, prediction_credentials)


def classify():
    print("\n -- classify")
    classify_project_id = os.environ["CUSTOM_VISION_CLASSIFY_PROJECT_ID"]
    publish_iteration_name = "Iteration2"

    with open(os.path.join("./images/LittleYellowCar.png"), "rb") as image_contents:
        results = predictor.classify_image(classify_project_id, publish_iteration_name, image_contents.read())

        for prediction in results.predictions:
            print("\t" + prediction.tag_name +
                  ": {0:.2f}%".format(prediction.probability * 100))


def detect():
    print("\n -- detect")
    detect_project_id = os.environ["CUSTOM_VISION_DETECT_PROJECT_ID"]
    publish_iteration_name = "Iteration1"

    image_path = os.path.join("./images/SoccerBall.png")
    with open(image_path, "rb") as image_contents:
        results = predictor.detect_image(detect_project_id, publish_iteration_name, image_contents.read())

    image = cv2.imread(image_path)
    for prediction in results.predictions:
        top_left = (int(prediction.bounding_box.left * image.shape[1]), int(prediction.bounding_box.top * image.shape[0]))
        bottom_right = (int((prediction.bounding_box.left + prediction.bounding_box.width) * image.shape[1]),
                        int((prediction.bounding_box.top + prediction.bounding_box.height) * image.shape[0]))

        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        text = f"{prediction.tag_name}: {prediction.probability * 100:.2f}%"
        text_position = (top_left[0], top_left[1] - 10 if top_left[1] > 20 else top_left[1] + 10)
        cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('Image with Bounding Box', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # classify()
    detect()

