from ultralytics import YOLO
import cv2
import numpy as np
import logging
import TaskType


class VideoAnalyzer:
    __model_path = ""
    __video_path = ""
    __box_amount = 2
    __new_showed = False
    __task_type = 0
    __output_video_name = ""

    def __init__(self, model_path):
        self.model_path = model_path
        logging.basicConfig(filename='analyzed_info.log',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            encoding='utf-8',
                            level=logging.INFO)

    def get_model_path(self):
        return self.model_path

    def set_model_path(self, new_model_path):
        self.model_path = new_model_path

    def count_boxes(self, video_path, output_video_name):
        self.__output_video_name = output_video_name
        self.__video_path = video_path
        self.__task_type = TaskType.TaskType.count_boxes
        self.__start_video_analyze()

    def check_human_pull_out_boxes(self, video_path, output_video_name):
        self.__output_video_name = output_video_name
        self.__video_path = video_path
        self.__task_type = TaskType.TaskType.check_people_pull_out
        self.__start_video_analyze()

    def __start_video_analyze(self):
        model = YOLO(self.model_path)
        video_name = self.__output_video_name
        cap = cv2.VideoCapture(self.__video_path)
        video = cv2.VideoWriter(video_name, 0, 1, (int(640), int(640)))

        i = 0
        while cap.isOpened():

            ret, frame = cap.read()
            i += 1
            if not ret:
                break
            if i % 30 != 0:
                continue

            frame = cv2.resize(frame, (640, 640))
            frame_time = cap.get(cv2.CAP_PROP_POS_MSEC)
            results = model.track(frame, classes=(0, 1, 3), verbose=False, tracker="botsort.yaml")
            boxes = results[0].boxes
            frame = self.__get_marked_frame(boxes, frame, frame_time)
            video.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        video.release()

    def __get_marked_frame(self, boxes, frame, frame_time):
        for box in boxes:
            box_id = box.id

            x1, y1, x2, y2 = np.squeeze(box.xyxy.tolist())
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
            cv2.circle(frame, (int(abs(x1 - x2) / 2 + min(x1, x2)), int(abs(y1 - y2) / 2) + min(y1, y2)), radius=5,
                       color=(0, 0, 255), thickness=-1)

            if box_id is not None:
                cv2.putText(frame, str(int(box_id.item())), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

        if self.__task_type == TaskType.TaskType.count_boxes:
            self.__count_boxes(boxes, frame_time)
        if self.__task_type == TaskType.TaskType.check_people_pull_out:
            self.__check_human_pull_out(boxes, frame_time)

        return frame

    def __count_boxes(self, boxes, frame_time):
        previous_count = self.__box_amount * 1
        for box in boxes:
            x1, y1, x2, y2 = np.squeeze(box.xyxy.tolist())
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            if not self.__new_showed and self.__intersects(x1, x2, y1, y2):
                self.__box_amount += 1
                self.__new_showed = True
        if previous_count == self.__box_amount:
            self.__new_showed = False
        if self.__box_amount >= 12:
            logging.info(f"{int(frame_time // 1000 // 60)} min {int(frame_time // 1000 % 60)} seconds: Boxes unload")
            self.__box_amount -= 12

    def __intersects(self, other_x_min, other_x_max, other_y_min, other_y_max):
        x_min = 490
        x_max = 580
        y_min = 140
        y_max = 250
        return not (x_max < other_x_min
                    or x_min > other_x_max
                    or y_max < other_y_min
                    or y_min > other_y_max)

    def __check_human_pull_out(self, boxes, frame_time):
        if self.__box_amount != 0 and self.__box_amount > len(boxes):
            logging.info(f"{int(frame_time // 1000 // 60)} min {int(frame_time // 1000 % 60)} seconds: Box pull out by people")
        self.__box_amount = len(boxes)
