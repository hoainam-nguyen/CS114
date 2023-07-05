import os
import time
import torch
import streamlit as st
from PIL import Image
import numpy as np
import requests
import json
from PIL import Image
from src.utils.image_utils import image_to_base64, base64_to_image

def process(img0):
    METHOD = [
        "yolo_ratate_yolo_vietocr",
        "rmbg_rotate_yolo_vietocr",
        "rmbg_rotate_craft_vietocr_pick",
        "rmgb_rotate_craft_vietocr_gpt_lines",
        "rmgb_rotate_craft_vietocr_gpt_boxes",
    ]

    image_base64 = image_to_base64(Image.fromarray(img0).convert("RGB"))
    response = requests.post(
        url = "http://localhost:5012/api/v1/e2e",
        data = json.dumps({"image_base64": image_base64,  "method": METHOD[1]}),
        headers = {"Content-Type": "application/json"}
    ).json()     

    extracted = response["data"]["extracted"]
    ploted_img = base64_to_image(response["data"]["images"]["ploted_img"])

    return extracted, np.array(ploted_img)


def process_2(img0, method):

    image_base64 = image_to_base64(Image.fromarray(img0).convert("RGB"))
    start_time = time.time()
    response = requests.post(
        url = "http://localhost:5012/api/v1/e2e",
        data = json.dumps({"image_base64": image_base64,  "method": method}),
        headers = {"Content-Type": "application/json"}
    ).json()     

    extracted = response["data"]["extracted"]
    ploted_img = base64_to_image(response["data"]["images"]["ploted_img"])

    return time.time() - start_time, extracted, np.array(ploted_img)

def show_gpu_info():
    allocated_mem = "[GPU]Max Memory Allocated {} MB".format(
        torch.cuda.max_memory_allocated(device="cuda") / 1024.0 / 1024.0
    )
    cached_mem = "[GPU]Max Memory Cached {} MB".format(
        torch.cuda.max_memory_reserved(device="cuda") / 1024.0 / 1024.0
    )
    text_write = "{}\n{}".format(allocated_mem, cached_mem)
    st.text(text_write)

def component_result(extracted, img, ploted_img):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img)
    with col2:
        st.image(ploted_img)
    with col3:
        text_write = "".join(
            ["{}: {}\n".format(k, v) for k, v in extracted.items()]
        )
    st.text(text_write)
    st.markdown("---")
    show_gpu_info()
    st.markdown("---")
    total_runtime = 10
    total_runtime = round(float(total_runtime), 4)
    st.text("Total runtime: {}s".format(total_runtime))

def page_ui_1():
    option_column_1, option_column_2 = st.columns(2)
    col1, col2, col3 = st.columns(3)
    
    with option_column_1:
        with st.form("form1", clear_on_submit=True):
            content_file = st.file_uploader(
                "Upload your image here", type=["jpg", "jpeg", "png"]
            )
            submit = st.form_submit_button("Upload")
            if content_file is not None:
                pil_img = Image.open(content_file)
                img = np.array(pil_img)

                if submit:
                    print(">" * 100)
                    wait_text = st.text("Please wait...")
                    wait_text.empty()

                    extracted, ploted_img = process(img)
                    with col1:
                        st.image(img)
                    with col2:
                        st.image(ploted_img)
                    with col3:
                        text_write = "".join(
                            ["{}: {}\n".format(k, v) for k, v in extracted.items()]
                        )
                    st.text(text_write)
                    st.markdown("---")
                    show_gpu_info()
                    st.markdown("---")
                    total_runtime = 10
                    total_runtime = round(float(total_runtime), 4)
                    st.text("Total runtime: {}s".format(total_runtime))
    with option_column_2:
        pass

def page_ui_2():
    # st.title("Ứng dụng Streamlit với hai cột")
    # st.title("Thử nghiệm với nhiều phương pháp")
    flag = False
    # Chia layout thành hai cột
    left_column, right_column = st.columns(2)

    col1, col2 = st.columns(2)
    # Cột bên trái là cột tải lên ảnh
    with left_column:
        st.header("Upload image")
        uploaded_file = st.file_uploader("Select", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file)
            img = np.array(pil_img)
        # Kiểm tra xem có ảnh được tải lên hay không
        if uploaded_file is not None:
            with col1:
                st.image(uploaded_file, use_column_width=True)

    # Cột bên phải là cột tùy chọn lựa chọn nhiều giá trị
    with right_column:
        st.header("Optional Methods")
        options = [
            "yolo_rotate_yolo_vietocr",
            "rmbg_rotate_yolo_vietocr",
            "rmbg_rotate_craft_vietocr_pick",
            "rmgb_rotate_craft_vietocr_gpt_lines",
            "rmgb_rotate_craft_vietocr_gpt_boxes",
        ]
        selected_option = st.radio("Select", options)

        if st.button('Prediction'):

            print(">" * 100)
            wait_text = st.text("Please wait...")
            wait_text.empty()

            total_runtime, extracted, ploted_img = process_2(img, selected_option)

            with col2:
                st.image(ploted_img)
            
            flag = True
            # with col3:
    if flag:
        text_write = "".join(
            ["{}: {}\n".format(k, v) for k, v in extracted.items()]
        )
        st.text(text_write)
        st.markdown("---")
        show_gpu_info()
        st.markdown("---")
        total_runtime = round(float(total_runtime), 4)
        st.text("Total runtime: {}s".format(total_runtime))

        # Hiển thị các tùy chọn đã chọn
        # st.write("Các tùy chọn đã chọn:")
        # for option in selected_options:
            # st.write(option)
        # st.button("Submit: ", selected_option)

def main():
    st.title("MC_OCR KIE")

    page_ui_2()
    # menu = ["Test", "Draft"]
    # choice = st.sidebar.selectbox("Chọn tab", menu)
    
    # if choice == "Test":
    #     # page_ui_1()
    #     # pass
    #     page_ui_2()

    # if choice == "Tab 2":
    #     # page_ui_2()
    #     pass


if __name__=="__main__":
    main()