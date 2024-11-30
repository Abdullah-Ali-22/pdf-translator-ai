# app.py

import streamlit as st
import tempfile
import os
from backend import (
    ensure_folder_exists,
    extract_features,
    analyze_layout,
    extract_job_details,
    save_translated_word
)
from PIL import Image

def main():
    st.set_page_config(page_title="PDF Translator", layout="wide")
    st.title("PDF Translator and Word Document Generator")
    st.write("Upload a PDF file to translate its content and download the translated Word document.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Save the uploaded PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_pdf_path = tmp_file.name

        # Display the uploaded PDF (optional)
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded PDF', use_column_width=True)
        except:
            st.write("Uploaded file is a PDF.")

        # Processing button
        if st.button("Process and Translate"):
            with st.spinner("Processing the PDF..."):
                try:
                    # Define temporary output paths
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        images_output_folder = os.path.join(tmp_dir, "cropped_notlogo")
                        logo_output_folder = os.path.join(tmp_dir, "cropped_logo")
                        ensure_folder_exists(images_output_folder)
                        ensure_folder_exists(logo_output_folder)

                        # Path to the logo image (ensure this path is correct)
                        logo_image_path = "/Users/AbdullahMS/Desktop/Work/TCS/Int/db_Intern_project/Data/logo/logo.png"  # Update this path as needed

                        if not os.path.exists(logo_image_path):
                            st.error(f"Logo image not found at {logo_image_path}. Please check the path.")
                            return

                        # Extract features of the logo image
                        with Image.open(logo_image_path) as logo_img:
                            logo_image_features = extract_features(logo_img)

                        # Analyze the PDF and save results
                        md_content, logo_fig_indices = analyze_layout(
                            tmp_pdf_path,
                            images_output_folder,
                            logo_output_folder,
                            logo_image_features,
                        )

                        # Extract job details from the extracted text
                        table_filled = extract_job_details(md_content)  # Assuming extract_job_details returns JobDetailsSchema

                        # Define paths for the output Word documents
                        translated_word_output_path = os.path.join(tmp_dir, "translated_output.docx")

                        # Save translated Word document, including job details table
                        save_translated_word(
                            md_content,
                            images_output_folder,
                            translated_word_output_path,
                            logo_output_folder,
                            logo_fig_indices,
                            table_filled  # Pass the extracted job details
                        )

                        # Read the translated Word document
                        with open(translated_word_output_path, "rb") as f:
                            translated_doc_bytes = f.read()

                        # Provide download button
                        st.success("Translation and document generation completed!")
                        st.download_button(
                            label="Download Translated Word Document",
                            data=translated_doc_bytes,
                            file_name="translated_output.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")

        # Cleanup the temporary PDF file
        os.unlink(tmp_pdf_path)

if __name__ == "__main__":
    main()