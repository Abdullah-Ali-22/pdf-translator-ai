# app.py

import streamlit as st
import tempfile
import os
from backend import (
    ensure_folder_exists,
    extract_features,
    analyze_layout,
    save_translated_word
)
from table_generate_agent import extract_pdf_summary
from PIL import Image

def main():
    st.set_page_config(page_title="PDF to Word Translator - Document Conversion & Translation Tool", layout="wide")
    st.title("PDF to Word Translator - Extract, Filter & Translate Documents")
    st.write("Upload a PDF file to extract content, filter out logos, and translate to a formatted Word document")

    # File uploader for PDF
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    # File uploader for logo image
    uploaded_logo = st.file_uploader("Upload a logo image to exclude from the document", type=["png", "jpg", "jpeg"])
    
    # Default logo path as fallback
    default_logo_path = "Data/logo/logo.png"
    
    if uploaded_file is not None:
        # Get original filename and prepare output filename
        original_filename = uploaded_file.name
        output_filename = os.path.splitext(original_filename)[0] + ".docx"
        
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
            
        # Display uploaded logo if provided
        logo_image_path = None
        if uploaded_logo is not None:
            # Save the uploaded logo to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_logo.name)[1]) as logo_tmp:
                logo_tmp.write(uploaded_logo.read())
                logo_image_path = logo_tmp.name
            
            # Display the logo
            try:
                logo_image = Image.open(logo_image_path)
                st.image(logo_image, caption='Uploaded Logo', width=200)
            except Exception as e:
                st.error(f"Error displaying logo: {e}")
                if os.path.exists(logo_image_path):
                    os.unlink(logo_image_path)
                logo_image_path = None

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

                        # Use uploaded logo or default logo
                        if not logo_image_path:
                            if os.path.exists(default_logo_path):
                                logo_image_path = default_logo_path
                                st.info(f"Using default logo image.")
                            else:
                                st.warning("No logo image provided and default logo not found. Processing without logo filtering.")
                                # Create a blank 1x1 image as a fallback
                                blank_logo = Image.new('RGB', (1, 1), color='white')
                                blank_logo_path = os.path.join(tmp_dir, "blank_logo.png")
                                blank_logo.save(blank_logo_path)
                                logo_image_path = blank_logo_path

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

                        # Extract title and summary from the extracted text
                        pdf_summary = extract_pdf_summary(md_content)

                        # Define paths for the output Word documents
                        translated_word_output_path = os.path.join(tmp_dir, output_filename)

                        # Save translated Word document, including title and summary
                        save_translated_word(
                            md_content,
                            images_output_folder,
                            translated_word_output_path,
                            logo_output_folder,
                            logo_fig_indices,
                            pdf_summary  # Pass the extracted title and summary
                        )

                        # Read the translated Word document
                        with open(translated_word_output_path, "rb") as f:
                            translated_doc_bytes = f.read()

                        # Provide download button
                        st.success("Translation and document generation completed!")
                        st.download_button(
                            label="Download Translated Word Document",
                            data=translated_doc_bytes,
                            file_name=output_filename,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                    
                    # Clean up the temporary logo file if it was created
                    if uploaded_logo is not None and os.path.exists(logo_image_path):
                        os.unlink(logo_image_path)

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")

        # Cleanup the temporary PDF file
        os.unlink(tmp_pdf_path)

if __name__ == "__main__":
    main()