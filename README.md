# 🗑️ Garbage Classifier UI

A deep learning-powered image classification app that detects types of garbage (e.g., Plastic, Organic, Metal) using a trained CNN model. This project demonstrates real-world AI deployment using FastAPI and Streamlit.

---

## 🚀 Live Demo

- 🌐 **Web App (Streamlit):** [https://huggingface.co/spaces/sykamraju/garbage-classifier-ui](https://huggingface.co/spaces/sykamraju/garbage-classifier-ui)

---

## 🧠 Model Details

- Trained a CNN model on a garbage classification dataset with multiple classes:
  - `plastic`, `paper`, `metal`, `glass`, `organic`, etc.
- Final model saved at: `model/garbage_model.h5`

---

## 🗂️ Project Structure

```
.
├── app.py               # Streamlit UI
├── utils.py             # Preprocessing and prediction
├── model/
│   └── garbage_model.h5 # Trained model
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 📸 Screenshot

![Streamlit App Screenshot](https://raw.githubusercontent.com/SykamRaju/Garbage-Classifier-UI/main/screenshot.png)

---

## ⚙️ Tech Stack

- Python 3.9+
- TensorFlow / Keras
- OpenCV / PIL
- Streamlit
- Hugging Face Spaces

---

## 🧪 Local Setup (Optional)

```bash
git clone https://github.com/SykamRaju/Garbage-Classifier-UI.git
cd Garbage-Classifier-UI
pip install -r requirements.txt
streamlit run app.py
```

---

## 👨‍💻 Author

**Raju Sykam**  
🔗 Website: [https://raju.net.in](https://raju.net.in)  
🔗 GitHub: [https://github.com/SykamRaju](https://github.com/SykamRaju)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
