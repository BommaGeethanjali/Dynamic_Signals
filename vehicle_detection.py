import cv2
import numpy as np
import os
from centroid_tracker import CentroidTracker
from tkinter import messagebox

class VehicleDetector:
    def __init__(self):
        weights_path = "yolov3.weights"
        config_path = "yolov3.cfg"
        classes_path = "coco.names"

        missing_files = []
        for path in [weights_path, config_path, classes_path]:
            if not os.path.exists(path):
                missing_files.append(path)

        if missing_files:
            error_msg = (
                "The following YOLO model files are missing:\n" +
                "\n".join(missing_files) +
                "\n\nPlease download them and place them in the Dynamic_Signals directory:\n" +
                "- yolov3.weights: https://pjreddie.com/media/files/yolov3.weights\n" +
                "- yolov3.cfg: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg\n" +
                "- coco.names: https://github.com/pjreddie/darknet/blob/master/data/coco.names"
            )
            messagebox.showerror("Missing YOLO Files", error_msg)
            raise FileNotFoundError(f"Missing YOLO files: {', '.join(missing_files)}")

        self.net = cv2.dnn.readNet(weights_path, config_path)
        layer_names = self.net.getLayerNames()
        unconnected_layers = self.net.getUnconnectedOutLayers()
        
        if isinstance(unconnected_layers, np.ndarray) and unconnected_layers.ndim == 1:
            self.output_layers = [layer_names[i - 1] for i in unconnected_layers]
        else:
            self.output_layers = [layer_names[i[0] - 1] for i in unconnected_layers]
        
        with open(classes_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        self.ct = CentroidTracker()
        self.vehicle_types = ["car", "bus", "truck", "motorbike"]
        self.emergency_types = ["ambulance", "fire truck"]  # Add emergency vehicle types

    def detect_vehicles(self, video_path):
        try:
            video = cv2.VideoCapture(video_path)
            cumulative_count = 0
            emergency_detected = False
            window_name = "Traffic Detection"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            while True:
                ret, frame = video.read()
                if not ret:
                    break

                # Check if window is closed
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                frame = cv2.resize(frame, (416, 416))
                height, width, channels = frame.shape
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                
                self.net.setInput(blob)
                outs = self.net.forward(self.output_layers)
                
                class_ids, confidences, boxes = [], [], []
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        class_name = self.classes[class_id]
                        if confidence > 0.5:
                            if class_name in self.vehicle_types or class_name in self.emergency_types:
                                center_x = int(detection[0] * width)
                                center_y = int(detection[1] * height)
                                w = int(detection[2] * width)
                                h = int(detection[3] * height)
                                x = int(center_x - w / 2)
                                y = int(center_y - h / 2)
                                boxes.append([x, y, x + w, y + h])
                                confidences.append(float(confidence))
                                class_ids.append(class_id)
                                if class_name in self.emergency_types:
                                    emergency_detected = True

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                rects = [boxes[i] for i in range(len(boxes)) if i in indexes]
                
                objects = self.ct.update(rects)
                count = len(objects)
                cumulative_count = max(cumulative_count, count)

                for (objectID, centroid) in objects.items():
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

                cv2.putText(frame, f"Current Vehicles: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Max Vehicles: {cumulative_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if emergency_detected:
                    cv2.putText(frame, "Emergency Vehicle Detected!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow(window_name, frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            video.release()
            cv2.destroyAllWindows()
            
            return cumulative_count * 2, emergency_detected  # Return green time and emergency flag
        
        except Exception as e:
            print(f"Detection Error: {e}")
            return 0, False