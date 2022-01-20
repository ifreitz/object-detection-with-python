import cv2
import torch
import numpy as np
from time import time

class ObjectDetectorVersionOne:
    stream = cv2.VideoCapture(1)

    torch.cuda.is_available = lambda : False # Use CPU
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) # force_reload=True
    
    def score_frame(self, frame):
        """Identify available device to make the prediction and uses it to load and infer the frame.
        Once it has results it will extract the labels and coordinated along with scores for each object detected in the frame.
        """
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)
            frame = [frame]
            results = self.model(frame)
            labels = results.xyxyn[0][:, -1].numpy()
            cord = results.xyxyn[0][:, :-1].numpy()
            
            return labels, cord
        except Exception as exc:
            print(exc)
            raise ValueError(exc)

    def plot_boxes(self, results, frame):
        """Takes the results and the frame as input and plots boxes over all the
        objects which have a score higher than our threshold.
        """
        try:
            labels, cord = results
            n = len(labels)
            x_shape, y_shape = frame.shape[1], frame.shape[0]

            for i in range(n):
                row = cord[i]

                if row[4] < 0.2:
                    continue

                x1 = int(row[0] * x_shape)
                y1 = int(row[1] * y_shape)
                x2 = int(row[2] * x_shape)
                y2 = int(row[3] * y_shape)

                bgr = (0, 255, 0)
                classes = self.model.names
                label_font = cv2.FONT_HERSHEY_SIMPLEX

                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, classes[int(labels[i])], (x1, y1), label_font, 0.9, bgr, 2)

                return frame
        except Exception as exc:
            print(exc)
            raise ValueError(exc)

    def __init__(self):
        """Initiate the object detection"""
        try:
            player = self.stream
            assert player.isOpened()

            while True:
                start_time = time()
                ret, frame = player.read()
                assert ret

                results = self.score_frame(frame)
                
                if len(results[0]) > 0:
                    frame = self.plot_boxes(results, frame)

                end_time = time()
                fps = 1/np.round(end_time - start_time, 3)
                print(f"Frames Per Second : {fps}")

                cv2.imshow('Object detector. Click "q" to quit.', frame)
                if cv2.waitKey(1) == ord('q'):
                    break
        except Exception as exc:
            print(exc)
            raise ValueError(exc)
            
detector = ObjectDetectorVersionOne()

detector.stream.release()
cv2.destroyAllWindows()