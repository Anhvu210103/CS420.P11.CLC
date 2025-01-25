# Image Captioning with CLIP and ClipCap

## Overview
This Streamlit application uses CLIP (Contrastive Language-Image Pre-training) and ClipCap to generate descriptive captions for uploaded images.

## Features
- Upload images in PNG, JPG, or JPEG formats
- Generate captions using state-of-the-art AI models
- Simple and intuitive user interface

## Prerequisites
- Python 3.8+
- GPU recommended for optimal performance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/image-captioning-app.git
cd image-captioning-app
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the pre-trained model:
- Place `pytorch_model.pt` in the project root directory

## Usage
Run the Streamlit app:
```bash
streamlit run app.py
```

## Model Details
- CLIP Model: ViT-B/32
- Prefix Length: 10
- Caption Generation: Beam Search

## Technologies
- Streamlit
- PyTorch
- CLIP
- GPT-2
- ClipCap

## License
[Specify your license here]

## Acknowledgements
- CLIP by OpenAI
- ClipCap Research Project