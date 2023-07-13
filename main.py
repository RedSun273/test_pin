#
'''
Trained model link:  https://drive.google.com/file/d/1-3hRBeaDPJvavotVeuF3rUBdWqLfctxd/view?usp=sharing

For task people pull out:
https://drive.google.com/file/d/1gX4VVL_tEmIwcDuSx-pqhA4sjDi7QmEd/view?usp=sharing
https://drive.google.com/file/d/1t6UXoveMPlY8MP2UxI_MC7pffycDrdNx/view?usp=sharing

For count task:
https://drive.google.com/file/d/1YrcIJuljUKLtqzHwVxxd29YtHK4Nf2tC/view?usp=sharing
'''

'''
My paths:

Count boxes:
/home/danil/Downloads/1108402-video.h264.mp4

People pull out boxes:
/home/danil/Downloads/upper29-05-2.mp4
/home/danil/Downloads/upper29-05.mp4
'''
from VideoAnalyzer import VideoAnalyzer


def main():
    video_analyzer = VideoAnalyzer("/home/danil/PycharmProjects/test_box_predictor/yolov8m_pintask.pt")

    video_analyzer.check_human_pull_out_boxes("/home/danil/Downloads/upper29-05.mp4", "count_boxes.avi")


if __name__ == "__main__":
    main()
