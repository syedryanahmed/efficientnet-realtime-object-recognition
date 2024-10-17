import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained EfficientNet model
model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=True)

# Load the class labels for ImageNet
class_labels = tf.keras.utils.get_file('ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json')

# Read the labels
with open(class_labels, 'r') as f:
    class_labels_map = eval(f.read())

# Initialize camera
video_capture = cv2.VideoCapture(0)

# Set confidence threshold (adjust as needed)
threshold = 0.3

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Resize frame to fit model input size
    resized_frame = cv2.resize(frame, (224, 224))

    # Preprocess the frame for the model
    preprocessed_frame = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(resized_frame, axis=0))

    # Make predictions
    predictions = model.predict(preprocessed_frame)

    # Get the top prediction index and confidence
    top_prediction_idx = np.argmax(predictions[0])
    top_prediction_confidence = predictions[0][top_prediction_idx]

    # Check if the confidence is above the threshold
    if top_prediction_confidence > threshold:
        # Get the label of the top prediction
        label = class_labels_map[str(top_prediction_idx)][1]
    else:
        label = "Unknown"

    # Display label and confidence on the frame
    cv2.putText(frame, f"{label} ({top_prediction_confidence:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Object Recognizer', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV window
video_capture.release()
cv2.destroyAllWindows()