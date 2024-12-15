# import cv2
#
# video = cv2.VideoCapture('236893_small.mp4')
#
# while True:
#     ret, frame = video.read()
#     if not ret:
#         break
#
#     yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
#
#     yuv_frame[:, :, 0] = cv2.equalizeHist(yuv_frame[:, :, 0])
#
#     equalized_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR)
#
#     cv2.imshow('Histogram Equalized Video', equalized_frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# video.release()
# cv2.destroyAllWindows()
#
# ##############################################################################################
#
#
#
# # import cv2
# #
# # image = cv2.imread("download.jpeg")
# # yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
# # yuv_image[:, :, 0] = cv2.subtract(yuv_image[:, :, 0], 50)
# # output_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
# # cv2.imshow("Original Image", image)
# # cv2.imshow("Modified Image", output_image)
# # cv2.imwrite("output_image.jpg", output_image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
#
#
#
# ##############################################################################################
#
#
#
# # import cv2
# # import numpy as np
# #
# # video = cv2.VideoCapture('236893_small.mp4')
# #
# # kernel = np.array([
# #     [-1, -1, -1],
# #     [-1,  9, -1],
# #     [-1, -1, -1]
# # ])
# #
# # while True:
# #     ret, frame = video.read()
# #     if not ret:
# #         break
# #
# #     enhanced_frame = cv2.filter2D(frame, -1, kernel)
# #
# #     cv2.imshow('Original', frame)
# #     cv2.imshow('Enhanced', enhanced_frame)
# #
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break
# #
# # video.release()
# # cv2.destroyAllWindows()
#
#
#
#
#
#
#
#
#
#
#
# ##############################################################################################
#
#
#
#
# # import cv2
# #
# # video = cv2.VideoCapture('236893_small.mp4')
# #
# # kernel_size = (5,5)
# #
# # def adjust_contrast_brightness(frame, contrast, brightness):
# #     return cv2.convertScaleAbs(frame, alpha=1 + contrast /127, beta=brightness)
# #
# # while True:
# #     res, frame = video.read()
# #     if not res:
# #         break
# #
# #     adjust_video = adjust_contrast_brightness(frame, contrast=120, brightness=4)
# #     cv2.imshow('Video', adjust_video)
# #     cv2.waitKey(4)
# #
#
#
#
#
#
# ##############################################################################################
# # import numpy as np
# # import scipy.io.wavfile as wavfile
# # from scipy.signal import wiener
# #
# # fs, noisy_signal = wavfile.read('noisy_audio_Final.wav')
# #
# # noisy_signal = noisy_signal / np.max(np.abs(noisy_signal))
# #
# # filtered_signal = wiener(noisy_signal, noise=1e-1)
# #
# # filtered_signal = (filtered_signal * 32767).astype(np.int16)
# #
# # wavfile.write('filtered_audio.wav', fs, filtered_signal)
#from email.mime import image

# ##############################################################################################













