# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="./pascal_voc_seg"
mkdir -p ${WORK_DIR}

cd ${WORK_DIR}
tar -xf "../VOCtrainval_11-May-2012.tar"
cp "../trainaug.txt" "./VOCdevkit/VOC2012/ImageSets/Segmentation"
unzip "../SegmentationClassAug.zip" -d "./VOCdevkit/VOC2012"
rm -r "./VOCdevkit/VOC2012/__MACOSX"

cd ${CURRENT_DIR}

# Root path for PASCAL VOC 2012 dataset.
PASCAL_ROOT="${WORK_DIR}/VOCdevkit/VOC2012"

# Remove the colormap in the ground truth annotations.
SEG_FOLDER="${PASCAL_ROOT}/SegmentationClassAug"
SEMANTIC_SEG_FOLDER="${PASCAL_ROOT}/SegmentationClassAugRaw"

echo "Removing the color map in ground truth annotations..."
python3.7 ./remove_gt_colormap.py \
  --original_gt_folder="${SEG_FOLDER}" \
  --output_dir="${SEMANTIC_SEG_FOLDER}"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

IMAGE_FOLDER="${PASCAL_ROOT}/JPEGImages"
LIST_FOLDER="${PASCAL_ROOT}/ImageSets/Segmentation"

echo "Converting PASCAL VOC 2012 dataset..."
python3.7 ./build_voc2012_data.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEMANTIC_SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="jpg" \
  --output_dir="${OUTPUT_DIR}"
