# This is a sample Python script.
import pyautogui as pyautogui
import streamlit as st
import time
import cv2 as cv
import numpy as np
from PIL import Image
from utils.cv_filters import strel_line, imadjust, gaussian_kernel, wiener_filter, laplacianOfGaussian
import os

os.environ['DISPLAY'] = ':0'

def main():
    image_s=None
    image_h=None
    st.title("Dermoscopy Images Preprocessing")
    process = st.sidebar.radio('Type of process', ('Registration','Shaver'))
    with st.container():
        if process == 'Registration':
            st.title("Registration")
            st.sidebar.write('You selected registration')
            # Add a slider to the sidebar:
            austerity = st.sidebar.slider(
                'Austerity',
                1.0, 100.0, (70.0))
            minimum = st.sidebar.slider(
                'Minimum matches',
                0.0, 1000.0, (10.0))
            MIN_MATCH_COUNT = minimum
            sample = st.file_uploader("Choose an sample image...")
            if sample is not None:
                image_s = Image.open(sample)
                st.image(image_s, caption='Sample Image', width=300)
            history = st.file_uploader("Choose a history image...")
            if history is not None:
                image_h = Image.open(history)
                st.image(image_h, caption='History Image', width=300)
            if sample is not None and history is not None:
                if st.button('Transform image'):
                    with st.spinner('Registering image'):
                        image_s = np.array(image_s)
                        image_h = np.array(image_h)
                        # Initiate SURF detector
                        orb = cv.ORB_create(5000)
                        # find the keypoints and descriptors with SURF
                        kp1, des1 = orb.detectAndCompute(image_s, None)
                        kp2, des2 = orb.detectAndCompute(image_h, None)
                        FLANN_INDEX_KDTREE = 1
                        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                        search_params = dict(checks=50)
                        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
                        # Match the two sets of descriptors.
                        matches = matcher.match(des1, des2)
                        # Sort matches on the basis of their Hamming distance.
                        matches = sorted(matches,key=lambda x:x.distance)
                        # Take the top 70 % matches forward.
                        matches = matches[:int(len(matches) * austerity)]
                        no_of_matches = len(matches)
                        # flann = cv.FlannBasedMatcher(index_params, search_params)
                        # matches = flann.knnMatch(des1, des2, k=2)
                        # store all the good matches as per Lowe's ratio test.
                        # good = []
                        # for m, n in matches:
                        #    if m.distance < 0.7 * n.distance:
                        #        good.append(m)

                        if len(matches) > MIN_MATCH_COUNT:
                            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
                            im_out = cv.warpPerspective(image_s, M, (image_h.shape[1], image_h.shape[0]))
                        else:
                            st.write('Not enough matches found')
                    #output_image = cv.cvtColor(im_out, cv.COLOR_BGR2RGB)
                        output_image = Image.fromarray(im_out)
                        if output_image:
                            st.image(output_image, caption='Registered Image', width=300)
                            st.success('Done')
                            if st.button('Try Again'):
                                pyautogui.hotkey("ctrl", "F5")

        else:
            with st.container():
                st.sidebar.write("You selected shaver")
                st.title("Shaver")
                sample = st.file_uploader("Choose an image...")
                process = st.sidebar.radio('Type of process', ('Bothat','Laplacian','Log','Logsobel','Lls'))
                st.sidebar.write('You selected '+process)
                if sample:
                    st.image(sample, caption='Hairy Image', width=300)
                    if process=='Bothat':
                        if st.button('Shave image'):
                            with st.spinner('Shaving image'):
                                imgOriginal = Image.open(sample)
                                imgOriginal = np.array(imgOriginal)
                                img = cv.cvtColor(imgOriginal, cv.COLOR_BGR2GRAY)
                                avg_img = cv.blur(img, (3, 3))
                                laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
                                laplacian_img = cv.filter2D(avg_img, -1, laplacian_kernel)
                                subtracted_img = avg_img - laplacian_img
                                kernel0 = strel_line(9, 0)
                                kernel45 = strel_line(9, 45)
                                kernel45 = kernel45.astype('uint8')
                                kernel90 = strel_line(9, 90)
                                bothat_img0 = cv.morphologyEx(subtracted_img, cv.MORPH_BLACKHAT, kernel0)
                                bothat_img45 = cv.morphologyEx(subtracted_img, cv.MORPH_BLACKHAT, kernel45)
                                bothat_img90 = cv.morphologyEx(subtracted_img, cv.MORPH_BLACKHAT, kernel90)
                                bothat_img_add = bothat_img0 + bothat_img45
                                bothat_img = bothat_img_add + bothat_img90
                                bothat_img = imadjust(bothat_img)
                                ret, thresh2 = cv.threshold(bothat_img, 10, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                                kernel = np.ones((8, 8), np.uint8)
                                dilated_img = cv.dilate(thresh2, kernel, iterations=1)
                                res_img = cv.inpaint(imgOriginal, dilated_img, 3, cv.INPAINT_TELEA)
                                output_image = res_img.astype("uint8")
                                output_image = Image.fromarray(output_image)
                                if output_image:
                                    st.image(output_image, caption='Shaved Image', width=300)
                                    st.success('Done')
                                    if st.button('Try Again'):
                                        pyautogui.hotkey("ctrl", "F5")
                                else:
                                    st.write('Operation failed')
                    if process=='Laplacian':
                        if st.button('Shave image'):
                            with st.spinner('Shaving image'):
                                imgOriginal = Image.open(sample)
                                imgOriginal = np.array(imgOriginal)
                                img = cv.cvtColor(imgOriginal, cv.COLOR_BGR2GRAY)
                                laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
                                laplacian_img = cv.filter2D(img, -1, laplacian_kernel)
                                subtracted_img = img - laplacian_img
                                gaus_kernel = gaussian_kernel(1)
                                filtered_img = wiener_filter(subtracted_img, gaus_kernel, K=700)
                                log_img = laplacianOfGaussian(filtered_img, 3.5)
                                bridge_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
                                bridged_img = cv.dilate(log_img, bridge_kernel, iterations=1)
                                clean_kernel = np.ones((2, 2), np.uint8)
                                clean_img = cv.morphologyEx(bridged_img, cv.MORPH_OPEN, clean_kernel)
                                kernel0 = strel_line(15, 0)
                                kernel45 = strel_line(15, 45)
                                kernel45 = kernel45.astype('uint8')
                                kernel90 = strel_line(15, 90)
                                closing0_img = cv.morphologyEx(clean_img, cv.MORPH_CLOSE, kernel0)
                                closing45_img = cv.morphologyEx(closing0_img, cv.MORPH_CLOSE, kernel45)
                                closing90_img = cv.morphologyEx(closing45_img, cv.MORPH_CLOSE, kernel90)
                                res_img = cv.inpaint(imgOriginal, closing90_img.astype("uint8"), 1, cv.INPAINT_TELEA)
                                output_image = Image.fromarray(res_img.astype("uint8"))
                                if output_image:
                                    st.image(output_image, caption='Shaved Image', width=300)
                                    st.success('Done')
                                    if st.button('Try Again'):
                                        pyautogui.hotkey("ctrl", "F5")
                                else:
                                    st.write('Operation failed')
                    if process=='Log':
                        if st.button('Shave image'):
                            with st.spinner('Shaving image'):
                                imgOriginal = Image.open(sample)
                                imgOriginal = np.array(imgOriginal)
                                red_img = imgOriginal.copy()
                                red_img[:, :, 0] = 0
                                red_img[:, :, 1] = 0
                                red_img = cv.cvtColor(red_img, cv.COLOR_BGR2GRAY)
                                log = laplacianOfGaussian(red_img.astype('float64'), 3.5)
                                dil_kernel = np.ones((15, 15), np.uint8)
                                dilation = cv.dilate(log, dil_kernel, iterations=1)
                                er_kernel = np.ones((8, 8), np.uint8)
                                erode = cv.erode(dilation, er_kernel, iterations=1)
                                output_image = cv.inpaint(imgOriginal, erode.astype('uint8'), 1, cv.INPAINT_TELEA)
                                output_image = Image.fromarray(output_image.astype("uint8"))
                                if output_image:
                                    st.image(output_image, caption='Shaved Image', width=300)
                                    st.success('Done')
                                    if st.button('Try Again'):
                                        pyautogui.hotkey("ctrl", "F5")
                                else:
                                    st.write('Operation failed')
                    if process=='Lls':
                        if st.button('Shave image'):
                            with st.spinner('Shaving image'):
                                imgOriginal = Image.open(sample)
                                imgOriginal = np.array(imgOriginal)
                                img = cv.cvtColor(imgOriginal, cv.COLOR_BGR2GRAY)
                                laplacian_img = cv.Laplacian(img, cv.CV_8U)
                                subtracted_img = img - laplacian_img
                                blur = cv.GaussianBlur(subtracted_img, (3, 3), 0)
                                log_edge_detection_img = cv.Laplacian(blur, cv.CV_8U)
                                grad_x = cv.Sobel(subtracted_img, cv.CV_8U, 1, 0, ksize=3, scale=1, delta=0,
                                                  borderType=cv.BORDER_DEFAULT)
                                grad_y = cv.Sobel(subtracted_img, cv.CV_8U, 0, 1, ksize=3, scale=1, delta=0,
                                                  borderType=cv.BORDER_DEFAULT)
                                abs_grad_x = cv.convertScaleAbs(grad_x)
                                abs_grad_y = cv.convertScaleAbs(grad_y)
                                sobel_edge_detection_img = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
                                add_binary_img = log_edge_detection_img + sobel_edge_detection_img
                                gaus_kernel = gaussian_kernel(1)
                                filtered_img = wiener_filter(add_binary_img, gaus_kernel, K=700)
                                filtered_img = cv.normalize(filtered_img, None, alpha=0, beta=255,
                                                            norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
                                ret, thresh2 = cv.threshold(filtered_img, 127, 255, cv.THRESH_BINARY)
                                clean_kernel = np.ones((1, 1), np.uint8)
                                clean_img = cv.morphologyEx(thresh2, cv.MORPH_OPEN, clean_kernel)
                                kernel_dil = np.ones((6, 6), np.uint8)
                                dilated_img = cv.dilate(clean_img, kernel_dil, iterations=1)
                                kernel_er = np.ones((1, 1), np.uint8)
                                eroded_img = cv.erode(dilated_img, kernel_er, iterations=1)
                                res_img = cv.inpaint(imgOriginal, eroded_img, 1, cv.INPAINT_TELEA)
                                output_image = Image.fromarray(res_img.astype("uint8"))
                                if output_image:
                                    st.image(output_image, caption='Shaved Image', width=300)
                                    st.success('Done')
                                    if st.button('Try Again'):
                                        pyautogui.hotkey("ctrl", "F5")
                                else:
                                    st.write('Operation failed')
                    if process=='Logsobel':
                        if st.button('Shave image'):
                            with st.spinner('Shaving image'):
                                imgOriginal = Image.open(sample)
                                imgOriginal = np.array(imgOriginal)
                                red_img = imgOriginal.copy()
                                red_img[:, :, 0] = 0
                                red_img[:, :, 1] = 0
                                red_img = cv.cvtColor(red_img, cv.COLOR_BGR2GRAY)
                                blur = cv.GaussianBlur(red_img, (3, 3), 0)
                                log_edge_detection_img = cv.Laplacian(blur, cv.CV_8U)
                                grad_x = cv.Sobel(red_img, cv.CV_8U, 1, 0, ksize=3, scale=1, delta=0,
                                                  borderType=cv.BORDER_DEFAULT)
                                grad_y = cv.Sobel(red_img, cv.CV_8U, 0, 1, ksize=3, scale=1, delta=0,
                                                  borderType=cv.BORDER_DEFAULT)
                                abs_grad_x = cv.convertScaleAbs(grad_x)
                                abs_grad_y = cv.convertScaleAbs(grad_y)
                                sobel_edge_detection_img = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
                                add_binary_img = log_edge_detection_img + sobel_edge_detection_img
                                ret, thresh2 = cv.threshold(add_binary_img, 10, 255, cv.THRESH_BINARY)
                                kernel0 = strel_line(6, 0)
                                kernel45 = strel_line(6, 45)
                                kernel45 = kernel45.astype('uint8')
                                kernel90 = strel_line(6, 90)
                                closing0_img = cv.morphologyEx(thresh2, cv.MORPH_CLOSE, kernel0)
                                closing45_img = cv.morphologyEx(closing0_img, cv.MORPH_CLOSE, kernel45)
                                closing90_img = cv.morphologyEx(closing45_img, cv.MORPH_CLOSE, kernel90)
                                gaus_kernel = gaussian_kernel(1)
                                filtered_img = wiener_filter(closing90_img, gaus_kernel, K=700)
                                filtered_img = cv.normalize(filtered_img, None, alpha=0, beta=255,
                                                            norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
                                ret, thresh2 = cv.threshold(filtered_img, 127, 255, cv.THRESH_BINARY)
                                kernel = np.ones((3, 3), np.uint8)
                                dilated_img = cv.dilate(thresh2, kernel, iterations=1)
                                res_img = cv.inpaint(imgOriginal, dilated_img, 1, cv.INPAINT_TELEA)
                                output_image = Image.fromarray(res_img.astype("uint8"))
                                if output_image:
                                    st.image(output_image, caption='Shaved Image', width=300)
                                    st.success('Done')
                                    if st.button('Try Again'):
                                        pyautogui.hotkey("ctrl", "F5")
                                else:
                                    st.write('Operation failed')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
