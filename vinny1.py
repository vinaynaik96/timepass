from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def list_to_pdf(strings, output_file):
    pdf = canvas.Canvas(output_file, pagesize=letter)
    y_coordinate = 750  # Starting y-coordinate for text

    for string in strings:
        pdf.drawString(100, y_coordinate, string)
        y_coordinate -= 20  # Move to the next line

    pdf.save()

# Example list of strings
my_strings = [
    "Hello,",
    "This is a sample text.",
    "Creating a PDF from a list of strings.",
    "Regards,",
    "Me"
]

# Convert the list of strings to a PDF file
list_to_pdf(my_strings, "output.pdf")
