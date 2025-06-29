# Attention Graph Visualizer

This project is a web-based tool for visualizing token-level attention patterns in transformer-based language models like GPT-2. It allows users to input custom text and explore how attention flows between tokens across all layers and heads.

![Screenshot](./screenshots/demo.png) <!-- Add screenshot if available -->

## 🔍 Features
- Visualizes token-to-token attention using interactive D3.js graphs
- Layer-wise and head-wise attention inspection
- Threshold slider for filtering low-weight edges
- Built with Flask (backend), D3.js (frontend), and Hugging Face Transformers (model)

---

## 🚀 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/attention-graph-visualizer.git
cd attention-graph-visualizer
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** If `requirements.txt` is missing, install manually:
```bash
pip install flask torch transformers
```

### 4. Run the application
```bash
python app.py
```

The app will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## 📁 Project Structure
```
├── app.py                     # Main Flask app backend
├── app/
│   ├── static/
│   │   ├── style.css         # UI styles
│   │   └── visualizer.js     # D3-based visualization logic
│   └── templates/
│       └── index.html        # Frontend UI
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```
