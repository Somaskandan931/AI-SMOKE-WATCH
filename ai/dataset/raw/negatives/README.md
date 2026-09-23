Drop your own "vehicle, no smoke" photos here (jpg/jpeg/png).

download_from_roboflow.py's multi-source merge mode (`--sources sources.yaml`)
automatically folds every image in this folder into the training set as a
background image with an empty label file (no boxes) -- i.e. a real
negative example, per PRD 14.3: negative examples matter more than extra
positives for avoiding false-positive smoke detections on dust, fog,
steam, or shadows.

No labeling needed. Just drop images in and re-run the merge script.
