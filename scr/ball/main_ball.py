import cv2
import time
import numpy as np

from scr.ball import tracker, catboost_filter, detection, feature_generation
from scr.court import court_detector
from scr.shared import tools, config

cap = cv2.VideoCapture(config.INPUT_VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

BALL_MODEL = detection.load_model(config.BALL_MODEL_PATH)
CAT_BOOST_FILTER = catboost_filter.cat_boost_load(config.CAT_BOOST_MODEL_PATH)

COURT_MODEL = court_detector.load_court_model(config.COURT_MODEL_PATH)

input_ball_name, output_ball_name, _, _, _ = detection.model_params(BALL_MODEL)

input_court_name, output_court_name, _, _ = court_detector.court_model_params(COURT_MODEL)

frame_id_counter = 0
track_id_counter = 0

buffer_frame = []
all_track = {}
json_database = {}

court_corners_array = []
court_corners = None

start_time = time.time()

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    if frame_id_counter % 30 == 0 or frame_id_counter == total_frames:
        tools.progress_bar_drawing(frame_id_counter, total_frames, start_time)

    frame_id_counter += 1

    if court_corners is None:
        court_detection = court_detector.detect_court(frame, COURT_MODEL, input_court_name, output_court_name)

        if court_detection is not None:
            corners = court_detector.court_mask_corners(court_detection["mask"])

            if corners is not None:
                court_corners_array.append(corners)

        if len(court_corners_array) >= config.BUFFER_COURT:
            court_corners = tuple(np.median(np.array(court_corners_array), axis=0).astype(np.int32))

    preprocessed_frame = detection.preprocess_for_onnx(frame, config.INPUT_W, config.INPUT_H)
    buffer_frame.append(preprocessed_frame)

    if len(buffer_frame) == config.SEQ_LEN:

        #Detection_part

        INPUT = detection.input_tensor(buffer_frame)
        buffer_frame.pop(0)

        OUTPUT = detection.inference(BALL_MODEL, input_ball_name, output_ball_name, INPUT)
        visible, decode_x, decode_y = detection.heatmap_decoder(OUTPUT, config.HEATMAP_TRASH)

        x, y = detection.original_dimension(decode_x, decode_y,
                                            config.INPUT_W, config.INPUT_H,
                                            config.VIDEO_WIDTH, config.VIDEO_HEIGHT)

        detect = detection.create_dict(frame_id_counter, visible, x, y)

        if detect["visible"] == 1:
            frame = tools.detect_drawing(frame, detect)

        #Tracker_part

        matching = tracker.match_detection(all_track, detect, config.MATCH_CONFIRMED_DIST, config.MATCH_UNCONFIRMED_DIST)

        if matching:
            matching_id = list(matching.keys())[-1]
            tracker.update_track(all_track[matching_id], detect, False)

        elif detect["visible"] == 1:

            if len(all_track) > config.LEN_MAX_TRACK:

                all_track = tracker.clearing_track_queue(all_track, config.LEN_MAX_TRACK)

            track_id_counter += 1
            all_track[track_id_counter] = tracker.create_track(detect)

        all_track = tracker.gap_cleaner(all_track, detect,
                                        config.GAP_WITH_CONFIRMED, config.GAP_WITHOUT_CONFIRMED)

        main_id = tracker.select_main_ball(all_track)

        if main_id is not None:

            catboost_track = feature_generation.feature_generator(all_track[main_id])

            if len(all_track[main_id]["x"]) >= 2:

                is_a_ball = catboost_filter.cat_boost_filter(
                    catboost_track, CAT_BOOST_FILTER, config.Probability)

            else:
                is_a_ball = False

            if is_a_ball:

                all_track[main_id]["confirmed"] = True

                frame = tools.track_drawing(frame, all_track[main_id], main_id)

    if court_corners is not None:
        frame = tools.court_drawing(frame, court_corners)

    cv2.imshow("frame", frame)
    cv2.pollKey()

cap.release()
cv2.destroyAllWindows()

end_time = time.time()

print(f"Целевое значение {round(total_frames / 30 / 60, 2)} минут")