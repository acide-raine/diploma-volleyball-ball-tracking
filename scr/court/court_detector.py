import onnxruntime as ort
import numpy as np
import cv2

def load_court_model(model_path: str) -> ort.InferenceSession:
    session = ort.InferenceSession(
        model_path,
        providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]
    )
    return session


def court_model_params(session: ort.InferenceSession) -> tuple[str, list[str], list, list]:
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    input_shape = session.get_inputs()[0].shape
    output_shapes = [output.shape for output in session.get_outputs()]

    return input_name, output_names, input_shape, output_shapes


def court_letterbox(frame: np.ndarray, input_h: int, input_w: int) -> tuple[np.ndarray, float, int, int, int, int]:
    orig_h, orig_w = frame.shape[:2]
    ratio = min(input_h / orig_h, input_w / orig_w)

    new_w, new_h = int(round(orig_w * ratio)), int(round(orig_h * ratio))
    img = cv2.resize(frame, (new_w, new_h))

    dw, dh = input_w - new_w, input_h - new_h
    pad_x, pad_y = int(round(dw / 2 - 0.1)), int(round(dh / 2 - 0.1))

    img = cv2.copyMakeBorder(img, pad_y, dh - pad_y, pad_x, dw - pad_x,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))

    return img, ratio, pad_x, pad_y, new_w, new_h


def court_preprocess(frame: np.ndarray, input_h: int, input_w: int) -> tuple[np.ndarray, tuple]:
    img, ratio, pad_x, pad_y, new_w, new_h = court_letterbox(frame, input_h, input_w)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img, (ratio, pad_x, pad_y, new_w, new_h)


def court_inference(session: ort.InferenceSession, input_name: str, output_names: list[str], preprocessed_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    outputs = session.run(output_names, {input_name: preprocessed_frame})

    predictions = outputs[0][0].T
    prototypes = outputs[1][0]

    return predictions, prototypes


def court_xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    result = np.empty_like(boxes)

    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    return result


def court_bbox_decoder(predictions: np.ndarray, conf_threshold: float, iou_threshold: float):
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    coefficients = predictions[:, 5:]

    valid = scores > conf_threshold
    boxes, scores, coefficients = boxes[valid], scores[valid], coefficients[valid]

    if len(scores) == 0:
        return None

    boxes_xyxy = court_xywh_to_xyxy(boxes)
    boxes_nms = [[box[0], box[1], box[2] - box[0], box[3] - box[1]] for box in boxes_xyxy]

    indices = cv2.dnn.NMSBoxes(boxes_nms, scores.tolist(), conf_threshold, iou_threshold)

    if len(indices) == 0:
        return None

    indices = np.asarray(indices).reshape(-1)
    best = indices[np.argmax(scores[indices])]

    return boxes_xyxy[best], float(scores[best]), coefficients[best]


def court_original_bbox(bbox: np.ndarray, transform: tuple, orig_w: int, orig_h: int) -> tuple[int, int, int, int]:
    ratio, pad_x, pad_y, _, _ = transform
    bbox = bbox.copy()

    bbox[[0, 2]] = (bbox[[0, 2]] - pad_x) / ratio
    bbox[[1, 3]] = (bbox[[1, 3]] - pad_y) / ratio

    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, orig_w - 1)
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, orig_h - 1)

    return tuple(bbox.astype(int))


def court_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def court_mask_decoder(coefficients: np.ndarray, prototypes: np.ndarray, transform: tuple,
                       bbox: tuple[int, int, int, int], orig_w: int, orig_h: int,
                       input_h: int, input_w: int, threshold: float = 0.5) -> np.ndarray:
    mask_logits = coefficients @ prototypes.reshape(prototypes.shape[0], -1)
    mask_logits = mask_logits.reshape(prototypes.shape[1], prototypes.shape[2])
    mask = court_sigmoid(mask_logits)

    mask = cv2.resize(mask, (input_w, input_h))

    _, pad_x, pad_y, new_w, new_h = transform
    mask = mask[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    mask = cv2.resize(mask, (orig_w, orig_h))

    binary_mask = (mask > threshold).astype(np.uint8)

    x1, y1, x2, y2 = bbox
    bbox_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    bbox_mask[y1:y2 + 1, x1:x2 + 1] = 1

    return binary_mask * bbox_mask


def detect_court(frame: np.ndarray, session: ort.InferenceSession, input_name: str,
                 output_names: list[str], input_h: int = 640, input_w: int = 640,
                 conf_threshold: float = 0.95, iou_threshold: float = 0.95) -> dict | None:

    orig_h, orig_w = frame.shape[:2]

    preprocessed_frame, transform = court_preprocess(frame, input_h, input_w)
    predictions, prototypes = court_inference(session, input_name, output_names, preprocessed_frame)

    detection = court_bbox_decoder(predictions, conf_threshold, iou_threshold)

    if detection is None:
        return None

    bbox_model, confidence, coefficients = detection
    bbox = court_original_bbox(bbox_model, transform, orig_w, orig_h)

    mask = court_mask_decoder(coefficients, prototypes, transform, bbox,
                              orig_w, orig_h, input_h, input_w)

    return {"bbox": bbox, "mask": mask, "confidence": confidence}


def court_mask_corners(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)

    for epsilon in (0.01, 0.02, 0.03, 0.04, 0.05):
        approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            points = approx.reshape(4, 2).astype(np.float32)

            sums = points.sum(axis=1)
            diffs = np.diff(points, axis=1).ravel()

            return np.array([
                points[np.argmin(sums)],
                points[np.argmin(diffs)],
                points[np.argmax(sums)],
                points[np.argmax(diffs)]
            ])

    return None

