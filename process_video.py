import os
import cv2
import PIL
import torch
import numpy as np
import shutil
from moviepy.editor import VideoFileClip

# Trying to reduce computational cost
def resize_frame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation import (
                                             DeepLabV3_MobileNet_V3_Large_Weights
                                             )


def load_model(model_name: str):
    model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
    transforms = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1.transforms()

    model.eval()

    _ = model(torch.randn(1, 3, 520, 520))

    return model, transforms

def create_binary_mask(outputs, background_class=0):
    labels = torch.argmax(outputs.squeeze(), dim=0).numpy()
    mask = (labels == background_class).astype(np.uint8)
    return mask

def remove_background(original_image, binary_mask):
    image_bgr = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    inverse_mask = abs(1 - binary_mask)
    result = cv2.bitwise_and(image_bgr, image_bgr, mask=inverse_mask)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

#Helper funciton to make frame square
def crop_to_square(frame):
    # Get the dimensions of the frame
    height, width = frame.shape[:2]

    # Determine the size of the square (it will be the smaller dimension of the frame)
    square_size = min(height, width)

    # Calculate the top left corner of the square crop
    top_left_x = (width - square_size) // 2
    top_left_y = (height - square_size) // 2

    # Crop the frame to a square
    square_frame = frame[top_left_y:top_left_y + square_size, top_left_x:top_left_x + square_size]

    return square_frame


#assumes input frame is already cropped to a square
def arrange_frames_circular(frame, scale=1):

    # Original dimensions
    frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale))) #making frame replicates smaller


    height, width = frame.shape[:2]
    assert height == width, "The frame must be square."

    # Create a large canvas to hold the 3x3 grid
    grid_size = height * 3
    canvas = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)

    # Positions for rotated frames (excluding center and corners)
    positions = {
        (0, 1): 180,     # Top middle
        (1, 0): 270,   # Middle left
        (1, 2): 90,    # Middle right
        (2, 1): 0    # Bottom middle
    }

    # Fill the specified positions with rotated frames
    for (row, col), angle in positions.items():
        # Calculate the top-left corner of the position in the grid
        top_left_x = col * height
        top_left_y = row * width

        # Rotate the frame
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        rotated_frame = cv2.warpAffine(frame, M, (width, height))

        # Place the rotated frame on the canvas
        canvas[top_left_y:top_left_y + height, top_left_x:top_left_x + width] = rotated_frame
    return canvas


def fade_in_effect(video_path, output_path, fade_in_seconds):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fade_in_frames = fade_in_seconds * fps

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break


        if current_frame < fade_in_frames:
            # Calculate the weight of the original frame (increases with each frame)
            alpha = current_frame / fade_in_frames
            beta = 1 - alpha  # Weight of the black frame (decreases with each frame)

            # Create a black frame
            black_frame = np.zeros(frame.shape, frame.dtype)

            # Compute weighted sum of the black frame and the current frame
            faded_frame = cv2.addWeighted(frame, alpha, black_frame, beta, 0)
            out.write(faded_frame)
        else:
            out.write(frame)
        current_frame += 1

    cap.release()
    out.release()


def fade_out_effect(video_path, output_path, fade_out_seconds):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fade_out_start_frame = total_frames - fade_out_seconds * fps

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame >= fade_out_start_frame:
            # Calculate the weight of the original frame (decreases with each frame)
            alpha = (total_frames - current_frame) / (total_frames - fade_out_start_frame)
            beta = 1 - alpha  # Weight of the black frame (increases with each frame)

            # Create a black frame
            black_frame = np.zeros(frame.shape, frame.dtype)

            # Compute weighted sum of the black frame and the current frame
            faded_frame = cv2.addWeighted(frame, alpha, black_frame, beta, 0)
            out.write(faded_frame)
        else:
            out.write(frame)

        current_frame += 1

    cap.release()
    out.release()


def video_inference_with_backgroundremoval(model_name: str, input_video_path: str, intermediate_output_path: str, final_output_path: str, device=None):
    device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    model, transforms = load_model(model_name) #load model function
    model.to(device)

    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return

    squared_frame = crop_to_square(frame)
    height, width = squared_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Utilizing the width and height from the test squre frame to have dimensions match, multyplying by 3 since the circualr aranegeent will create a 3x3 grid

    output_video = cv2.VideoWriter(intermediate_output_path, fourcc, fps, (width * 3, height * 3), isColor=True) #creates a new video file in the specified format at the specified path [intermediate_output_path]
    #output_video will be saved in intermediate_output_path , which is a string representing the file location where video data will be written

    #process each frame
    while True:
        frame = crop_to_square(frame) ##CROPPED, needs to go before funcitons so that bianry mask and everything is square too
        #process frame to remove background
        img_raw = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_t = transforms(img_raw)
        img_t = torch.unsqueeze(img_t, dim=0).to(device)

        with torch.no_grad():
            output = model(img_t)["out"].cpu()
        binary_mask = create_binary_mask(output)
        binary_mask = cv2.resize(binary_mask, (width, height), cv2.INTER_LINEAR) #rezisement

        result_image = remove_background(img_raw, binary_mask) #img_raw and bianry mask need same dimensions
        display_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
       
        arranged_frame = arrange_frames_circular(display_image)

        output_video.write(arranged_frame) #frames written into output_video object, Arranged frame should be wx3, hx3 size

        ret, frame = cap.read()
        if not ret:
            break

#checking video duration for fade-in effect

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    output_video.release()  #finalizes the video file writing process, ensuring file is closed properly and data is written properly

    if duration > 7:
        secondary_path = intermediate_output_path + ".tmp" #temporary for fade-in processing
        fade_in_effect(intermediate_output_path, secondary_path, 4)
        shutil.move(secondary_path, intermediate_output_path)

    # Add original audio to the processed video and re-encode
    processed_clip = VideoFileClip(intermediate_output_path) #creates 'VideoFileClip' object from the file located at intermediate_output_path.

    original_audio = VideoFileClip(input_video_path).audio
    
    final_clip = processed_clip.set_audio(original_audio)
    
    final_clip.write_videofile(final_output_path, codec='libx264', audio_codec='aac')

    # Cleanup: Remove the intermediate video file
    os.remove(intermediate_output_path)
























# import os
# import cv2
# import PIL
# import torch
# import numpy as np
# from moviepy.editor import VideoFileClip

# # Trying to reduce computational cost
# def resize_frame(frame, scale=0.5):
#     width = int(frame.shape[1] * scale)
#     height = int(frame.shape[0] * scale)
#     dimensions = (width, height)

#     return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
# from torchvision.models.segmentation import (
#                                              DeepLabV3_MobileNet_V3_Large_Weights
#                                              )


# def load_model(model_name: str):
#     if model_name.lower() not in ("mobilenet", "smallmobilenet"):
#         raise ValueError("'model_name' should be one of ('mobilenet', 'resnet_50', 'resnet_101')")
#     #transforms = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1.transforms()

#     # if model_name == "resnet_50":
#     #     model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
#     #     transforms = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1.transforms()

#     # elif model_name == "resnet_101":
#     #     model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
#     #     transforms = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1.transforms()

#     else:
#         model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
#         transforms = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1.transforms()

#     model.eval()

#     _ = model(torch.randn(1, 3, 520, 520))

#     return model, transforms

# def create_binary_mask(outputs, background_class=0):
#     labels = torch.argmax(outputs.squeeze(), dim=0).numpy()
#     mask = (labels == background_class).astype(np.uint8)
#     return mask

# def remove_background(original_image, binary_mask):
#     image_bgr = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

#     inverse_mask = abs(1 - binary_mask)

#     result = cv2.bitwise_and(image_bgr, image_bgr, mask=inverse_mask)

#     return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


# #flask app is only calling this functions, so this one has to include all the others. Treat the others as helper functions.
# def video_inference_with_backgroundremoval(model_name: str, input_video_path: str, intermediate_output_path: str, final_output_path: str, device=None):
#     device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
#     model, transforms = load_model(model_name) #load model function
#     model.to(device)

#     cap = cv2.VideoCapture(input_video_path)
#     if not cap.isOpened():
#         return

#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     output_video = cv2.VideoWriter(intermediate_output_path, fourcc, fps, (width, height), isColor=True)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
#         img_raw = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         img_t = transforms(img_raw)
#         img_t = torch.unsqueeze(img_t, dim=0).to(device)

#         with torch.no_grad():
#             output = model(img_t)["out"].cpu()

#         binary_mask = create_binary_mask(output)
#         binary_mask = cv2.resize(binary_mask, (width, height), cv2.INTER_LINEAR)
#         result_image = remove_background(img_raw, binary_mask)

#         display_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
#         output_video.write(display_image)

#         #cv2.imshow('Result', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     output_video.release()
#     cv2.destroyAllWindows()

#     # Add original audio to the processed video and re-encode
#     processed_clip = VideoFileClip(intermediate_output_path)
#     original_audio = VideoFileClip(input_video_path).audio
#     final_clip = processed_clip.set_audio(original_audio)
#     final_clip.write_videofile(final_output_path, codec='libx264', audio_codec='aac')

#     # Cleanup: Remove the intermediate video file
#     os.remove(intermediate_output_path)