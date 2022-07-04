import matplotlib.pyplot as plt
from makeupKit import eyesLandMarks
import streamlit as st

if __name__ == "__main__":
    
    imgPath = r"images\img.png"
    # imgPath = st.file_uploader("Upload your image")

    # identify landmarks
    img, detMarks = eyesLandMarks(imgPath)

    st.image([img,detMarks])
   
    