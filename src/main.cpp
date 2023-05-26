#include <opencv2/opencv.hpp>
#include <iostream>

#define WINDOW_NAME "presenter"
#define INPUT_VIDEO 0

#define FACE_DETECTION_MODEL_FILENAME "face_detection_yunet_2022mar.onnx"
#define FACE_DETECTION_CONFIG_FILENAME ""

#define CELLS_PER_ROW 8
#define CELLS_PER_COLUMN 9

#define YUNET_FACE_X_INDEX 0
#define YUNET_FACE_Y_INDEX 1
#define YUNET_FACE_WIDTH_INDEX 2
#define YUNET_FACE_HEIGHT_INDEX 3

cv::Scalar faceColor(0, 255, 0);

int main() {
    cv::Size2f screenSize(1000, 1000);
    cv::Size2f cellSize(screenSize.width / CELLS_PER_COLUMN, screenSize.height / CELLS_PER_ROW);

    cv::VideoCapture sourceVideo(INPUT_VIDEO);
    cv::Size sourceSize(static_cast<int>(sourceVideo.get(cv::CAP_PROP_FRAME_WIDTH)),
                        static_cast<int>(sourceVideo.get(cv::CAP_PROP_FRAME_HEIGHT)));
    std::cout << "Using video source " << sourceSize << std::endl;

    // Create detector
    cv::Ptr<cv::FaceDetectorYN> detector = cv::FaceDetectorYN::create(FACE_DETECTION_MODEL_FILENAME,
                                                                      FACE_DETECTION_CONFIG_FILENAME,
                                                                      screenSize,
                                                                      0.9f, 0.3f,
                                                                      5000,
                                                                      cv::dnn::DNN_BACKEND_DEFAULT,
                                                                      cv::dnn::DNN_TARGET_CPU);
    std::cout << "Detector created" << std::endl;

    // Create window
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE | cv::WINDOW_OPENGL);
    cv::resizeWindow(WINDOW_NAME, screenSize);
    std::cout << "Window created" << std::endl;

    // Target dimensions, where the faces will be detected
    cv::Rect2i regionOfInterest(270, 136, 206, 136);
//    cv::Scalar regionOfInterestColor(255, 0, 0);

    // Read input image
    while (cv::getWindowProperty(WINDOW_NAME, cv::WindowPropertyFlags::WND_PROP_VISIBLE) == 1.0
           && sourceVideo.isOpened()) {
        cv::Mat image;
        if (!sourceVideo.read(image) || !image.data) {
            std::cerr << "Failed to read video" << std::endl;
            break;
        }
//        std::cout << "Image received" << std::endl;

        // Take only region of interest
        cv::Mat preparedImage = cv::Mat(image, regionOfInterest);
//        cv::Mat preparedImage = image;

        // Prepare input
        cv::resize(preparedImage, preparedImage, screenSize);
        cv::Mat imageBlob = cv::dnn::blobFromImage(preparedImage);
//        std::cout << "Detector input prepared" << std::endl;

        // Detect faces
        cv::Mat faces;
        detector->detect(preparedImage, faces);
//        std::cout << "Detected " << faces.rows << " faces" << std::endl;

        // Draw grid
        float cellWidth = screenSize.width / CELLS_PER_COLUMN;
        float cellHeight = screenSize.height / CELLS_PER_ROW;
        for (int columnIndex = 0; columnIndex < CELLS_PER_COLUMN; columnIndex++) {
            for (int rowIndex = 0; rowIndex < CELLS_PER_ROW; rowIndex++) {
                // Draw horizontal line
                cv::Point2f start(0, cellHeight * static_cast<float>(rowIndex));
                cv::Point2f end(screenSize.width, cellHeight * static_cast<float>(rowIndex));
                cv::line(preparedImage, start, end, cv::Scalar(255, 255, 0));
            }

            // Draw vertical lines
            cv::Point2f start(cellWidth * static_cast<float>(columnIndex), 0);
            cv::Point2f end(cellWidth * static_cast<float>(columnIndex), screenSize.height);
            cv::line(preparedImage, start, end, cv::Scalar(255, 255, 0));
        }

        // Print rectangle around faces
        for (int faceIndex = 0; faceIndex < faces.rows; faceIndex++) {
            cv::Rect2f dimension(faces.at<float>(faceIndex, YUNET_FACE_X_INDEX),
                                 faces.at<float>(faceIndex, YUNET_FACE_Y_INDEX),
                                 faces.at<float>(faceIndex, YUNET_FACE_WIDTH_INDEX),
                                 faces.at<float>(faceIndex, YUNET_FACE_HEIGHT_INDEX));
            cv::rectangle(preparedImage, dimension, faceColor, 2);

            // fixme
            float cellRow = (dimension.x + (dimension.width / 2)) / cellSize.width;
            float cellColumn = (dimension.y + (dimension.height / 2)) / cellSize.height;

            std::string text = std::string("Row : " + std::to_string(cellRow)
                                           + std::string(", Cell : ")) + std::to_string(cellColumn);
            cv::putText(preparedImage, "", cv::Point2f(dimension.x, dimension.y), cv::FONT_HERSHEY_SIMPLEX, 2.0,
                        cv::Scalar(0, 0, 255), 2);
        }

        // Create window and display image with the detected faces
//        std::cout << "Frame " << frame++ << " rendered" << std::endl;
        faces.release();
        imageBlob.release();

        cv::imshow(WINDOW_NAME, preparedImage);
        cv::updateWindow(WINDOW_NAME);
        cv::waitKey(1);

        preparedImage.release();
        image.release();
    }

    //
    sourceVideo.release();
    cv::destroyWindow(WINDOW_NAME);
    return 0;
}
