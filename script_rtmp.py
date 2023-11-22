import argparse
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

import supervision as sv

COLORS = sv.ColorPalette.default()

ZONE_IN_POLYGONS = [ # 진입 영역
    np.array([
        [87, 532],[1783, 532],[1783, 612],[87, 612]
    ])
]

class DetectionsManager:
    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}

    def update(
        self,
        detections_all: sv.Detections,
        detections_in_zones: List[sv.Detections],
    ) -> sv.Detections:
        detections_all = sv.Detections.merge(detections_in_zones) # 영역 내 데이터만 처리
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            for tracker_id in detections_in_zone.tracker_id:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)
        if detections_all:
            detections_all.class_id = np.vectorize(
                lambda x: self.tracker_id_to_zone_id.get(x, -1)
            )(detections_all.tracker_id)
        return detections_all[detections_all.class_id != -1]


def initiate_polygon_zones(
    polygons: List[np.ndarray],
    frame_resolution_wh: Tuple[int, int],
    triggering_position: sv.Position = sv.Position.CENTER,
) -> List[sv.PolygonZone]:
    return [
        sv.PolygonZone(
            polygon=polygon,
            frame_resolution_wh=frame_resolution_wh,
            triggering_position=triggering_position,
        )
        for polygon in polygons
    ]


class VideoProcessor:
    def __init__(
        self,
        source_weights_path: str,
        source_video_path: str,
        target_video_path: str = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path

        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()

        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        self.zones_in = initiate_polygon_zones(
            ZONE_IN_POLYGONS, self.video_info.resolution_wh, sv.Position.CENTER
        )

        self.LOOKUP = sv.ColorLookup.CLASS
        self.box_annotator = sv.BoxAnnotator(color=COLORS, thickness=1)
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=1
        )
        self.detections_manager = DetectionsManager()

    def process_video(self): #1
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )
        print(frame_generator)

        if self.target_video_path:
            with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                    annotated_frame = self.process_frame(frame)
                    sink.write_frame(annotated_frame)
        else:
            with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                cap = cv2.VideoCapture("rtmp://localhost:1935/live/mist1")
                while True:
                    # print(cap)
                    ret, frame = cap.read()
                    # print(frame)
                    # frame = sink.write_frame(frame=frame)
                    # print(frame)
                    annotated_frame = self.process_frame(frame)
                    cv2.imshow("OpenCV View", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                
            # for frame in tqdm(frame_generator, total=self.video_info.total_frames):
            #     for frame in list(frame_generator):
            #         print(frame)
            #         annotated_frame = self.process_frame(frame)
            #         cv2.imshow("OpenCV View", annotated_frame)
            #         if cv2.waitKey(1) & 0xFF == ord("q"):
            #             break
            cv2.destroyAllWindows()

    def annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray: #3
        global annotated_frame
        if frame is not None:
            annotated_frame = frame.copy()
            for i, zone_in in enumerate(self.zones_in):
                annotated_frame = sv.draw_polygon(
                    annotated_frame, zone_in.polygon, COLORS.colors[i+1]
                )
                sv.PolygonZoneAnnotator(
                    zone=zone_in,
                    color=COLORS.colors[i+1],
                    thickness=1,
                    text_thickness=1,
                    text_scale=1
                )

            labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
            #annotated_frame = self.trace_annotator.annotate(annotated_frame, detections) # 트래킹 경로
            annotated_frame = self.box_annotator.annotate(
                annotated_frame, detections, labels
            )

        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray: #2
        results = self.model(
            frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold, device=0, imgsz=1920
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        detections.class_id = np.zeros(len(detections))
        detections = self.tracker.update_with_detections(detections)

        detections_in_zones = []
        
        for zone_in in self.zones_in:
            detections_in_zone = detections[zone_in.trigger(detections=detections)]
            detections_in_zones.append(detections_in_zone)
            
        detections = self.detections_manager.update(
            detections, detections_in_zones
        )
        return self.annotate_frame(frame, detections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Traffic Flow Analysis with YOLO and ByteTrack"
    )

    parser.add_argument(
        "--source_weights_path",
        required=True,
        help="Path to the source weights file",
        type=str,
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        default=None,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )

    args = parser.parse_args()
    processor = VideoProcessor(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )
    processor.process_video()
