import pdfplumber

# Replace this with the path to your PDF file
pdf_path = "raw_pdfs/SE124144.C1.pdf"

# Open the PDF
with pdfplumber.open(pdf_path) as pdf:
    # Loop through each page
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        print(f"--- Page {i+1} ---")
        print(text)
        print("\n")