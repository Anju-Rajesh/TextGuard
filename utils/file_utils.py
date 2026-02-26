import io
from pypdf import PdfReader

def extract_text_from_file(file):
    """
    Extracts text from a file object. Supports .txt and .pdf formats.
    """
    filename = file.filename.lower()
    
    if filename.endswith('.txt'):
        try:
            # Try utf-8 first
            return file.read().decode('utf-8')
        except UnicodeDecodeError:
            # Fallback to latin-1
            file.seek(0)
            return file.read().decode('latin-1')
            
    elif filename.endswith('.pdf'):
        try:
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return None
            
    return None
