# Updated Streamlit Web App with Navigation & Advanced Vehicle Analytics
import streamlit as st
import cv2
import numpy as np
from traffic_utils import process_frame_advanced


st.set_page_config(page_title="Smart Traffic Signal Optimizer", layout="wide")

# Sidebar Navigation
page = st.sidebar.selectbox("Navigate", ["About Us", "Traffic Video Analysis"])

if page == "About Us":
    st.title("👨‍💻 About the Project")
    st.markdown("""
    ### 🚦 Intelligent Traffic Signal Optimization System
    This project is developed by **Basavaraj Malipatil**, MCA Final Year Student, VTU Kalburgi College,
    under the guidance of **Ms.Shilpa Joshi(Assistant Professor)**.

    #### 💡 Purpose:
    To create a smart traffic control system using **YOLOv8** and **Canny Edge Detection** to estimate vehicle density and
    optimize traffic signal timings dynamically.

    #### 🛠 Technologies Used:
    - Python 3.8.0
    - Streamlit
    - OpenCV
    - YOLOv8 (Ultralytics)
    - NumPy
    """)

elif page == "Traffic Video Analysis":
    st.title("📊 Real-Time Traffic Video Processing Dashboard")
    video_file = st.file_uploader("Upload a Traffic Video", type=["mp4", "avi", "mov"])

    if video_file is not None:
        tfile = open("temp_video.mp4", "wb")
        tfile.write(video_file.read())
        cap = cv2.VideoCapture("temp_video.mp4")

        total_vehicle_count = 0
        total_type_counts = {}
        frame_count = 0

        stframe1 = st.empty()
        stframe2 = st.empty()
        stframe3 = st.empty()
        stats_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            edges, annotated_frame, vehicle_count, type_counts = process_frame_advanced(frame)

            total_vehicle_count += vehicle_count
            frame_count += 1

            # Update type counts
            for vtype, count in type_counts.items():
                total_type_counts[vtype] = total_type_counts.get(vtype, 0) + count

            # Display frames
            stframe1.image(edges, channels="GRAY", caption="Canny Edge Detection")
            stframe2.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                           caption=f"Annotated Frame - Vehicles: {vehicle_count}")

            avg_count = total_vehicle_count // frame_count
            optimal_green = int(10 + avg_count * 1.5)

            # Format type count string
            type_count_str = ", ".join([f"{k}: {v}" for k, v in type_counts.items()])

            stframe3.info(f"Vehicle Types in Frame: {type_count_str}")
            stats_placeholder.success(f"\nAverage Vehicles Detected: {avg_count} | "
                                     f"Optimal Green Light Time: {optimal_green} seconds")

        cap.release()
        st.success("✅ Video processing completed.")
        st.write("### 🚗 Total Vehicle Types Detected:")
        st.json(total_type_counts)
    else:
        st.warning("📷 Please upload a traffic video to start processing.")
