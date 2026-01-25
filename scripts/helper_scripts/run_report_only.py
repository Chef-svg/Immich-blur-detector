import insight_reporter

MODEL_NAME = "quality_model_focal_224_final.h5"
DATA_SOURCE = "prepared/data_formatted_native/val"
REPORT_FILE = "report_quality_model_focal_224.txt"

print(" Generating Insight Report for existing model...")
insight_reporter.generate_report(
    model_path=MODEL_NAME,
    data_dir=DATA_SOURCE,
    image_size=(224, 224),
    report_file=REPORT_FILE,
)
