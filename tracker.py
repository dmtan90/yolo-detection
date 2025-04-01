from deep_sort_realtime.deepsort_tracker import DeepSort


class Tracker:
    def __init__(self, embedder, embedder_gpu):
        #DeepSORT configuration ref here https://github.com/levan92/deep_sort_realtime
        self.object_tracker = DeepSort(
            # max_age defines the maximum number of consecutive frames a track can go without a successful detection association before it is considered lost and deleted from the tracker's active list.
            max_age=10,
            # n_init determines how many frames the tracker should wait before assigning a new id
            n_init=4,
            #nms_max_overlap specifies the maximum allowed overlap between bounding boxes during non-maximum suppression (NMS)
            nms_max_overlap=0.3,
            #max_iou_distance parameter defines the maximum intersection-over-union (IoU) distance between object detections
            max_iou_distance=0.9,
            #max_cosine_distance gating threshold for cosine distance metric (object appearance).
            max_cosine_distance=0.9,
            #nn_budget: It sets the budget for the nearest-neighbor search.
            nn_budget=None,
            gating_only_position=False,
            override_track_class=None,
            embedder=embedder,#mobilenet or clip_RN50 for ReID
            half=True,
            bgr=True,
            embedder_gpu=embedder_gpu,
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )

    def track(self, detections, frame):
        tracks = self.object_tracker.update_tracks(detections, frame=frame)

        tracking_ids = []
        boxes = []

        for track in tracks:
            if not track.is_confirmed():
                continue
            tracking_ids.append(track.track_id)
            ltrb = track.to_ltrb()
            boxes.append(ltrb)

        return tracking_ids, boxes
