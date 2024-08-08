# Anx-Check

Anx-Check is a stress detection web application designed to detect stress in textual input. Utilizing natural language processing techniques, this app analyzes the text to determine if it indicates stress. The project is built using Python, Flask, and HTML.

## Features

- Analyze text for stress indicators
- Display results indicating whether the text is stressed or not
- Simple and user-friendly web interface

## Technologies Used

- **Python**: For the main logic and stress detection model
- **Flask**: To create and run the web application
- **HTML**: For the frontend part of the web application

## Installation

 **Clone the repository**:
    ```bash
    git clone https://github.com/harsh-kakaiya/Anx-Check.git
    cd Anx-Check
    ```

 **Install the required packages**:

**Run the application**:
    ```bash
    python app.py
    ```

## Usage

1. Open your web browser and go to `_displayed link of localhost_`.
2. Enter the text you want to analyze in the provided text box.
3. Click on the "Predict" button.
4. The result will indicate whether the input text is stressed or not.

## Project Structure

```plaintext
Anx-Check/
│
├── app.py                  # Main application file
├── templates/
│   └── index.html          # HTML template for the web interface
└── README.md               # This readme file
