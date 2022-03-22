from course_ocr.task1.course_ocr_t1.metrics import measure_crop_accuracy
from pathlib import Path

path = Path("/home/artem/PycharmProjects/course_ocr/course_ocr/task1")

acc = measure_crop_accuracy(
    path / 'pred.json',
    path / 'gt.json'
)

print(acc)