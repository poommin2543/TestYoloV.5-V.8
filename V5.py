import cv2
import torch
import random

# Define colors for each class
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]

def load_model(weights_path, device):
    # Model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)  
    model.to(device).eval()
    return model
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    weights = 'Path\\bestV5.pt'  # path to your weights file
    model = load_model(weights, device)

    # Open a handle to the default system webcam (usually device 0)
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    while True:
        # Capture a single frame from the webcam
        ret, frame = cap.read()
        
        if not ret:
            break

        # Convert the image from BGR to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform inference
        # Perform inference
        results = model(img)

        # Process detections
        for *xyxy, conf, cls in results.xyxy[0].tolist():  # detections per image
            # Write results
            label = f'{results.names[int(cls)]}: {conf:.2f}'
            plot_one_box(xyxy, frame, label=label, line_thickness=3)

        # Display the image
        cv2.imshow('Webcam', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture handle
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()