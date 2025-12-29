# CardioVision AI 🫀

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![ML](https://img.shields.io/badge/AI-Ensemble%20Learning-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Enterprise Early Detection System using Ensemble Learning & Computer Vision**

**CardioVision AI** is a full-stack Clinical Decision Support System (CDSS) designed to assist medical professionals in the early diagnosis of heart disease. It bridges the gap between raw medical data and actionable insights by automating the digitization of physical lab reports and providing instant, AI-driven risk stratification.

---

## 📋 Table of Contents
- [The Problem](#-the-problem)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Screenshots](#-screenshots)
- [Future Scope](#-future-scope)
- [License](#-license)

---

## 🏥 The Problem
Cardiovascular diseases are the #1 cause of death globally. A major challenge in critical care is **"Time to Diagnosis."**
- **Manual Errors:** Typographical errors during manual data entry into EHR systems can lead to incorrect triage.
- **Silent Ischemia:** Asymptomatic patients often go undiagnosed until a critical event occurs.
- **Fragmented Data:** Physical paper reports make it difficult to visualize long-term health trends.

**CardioVision AI** solves this by acting as a "Second Opinion" tool that automates data entry and provides objective, mathematical risk assessment.

---

## 🌟 Key Features

### 1. Smart Vision OCR 👁️
- **Automated Digitization:** Uses **Tesseract OCR** and OpenCV to extract vitals (BP, Cholesterol) from images of lab reports in <2 seconds.
- **Synonym Detection:** Custom Regex logic identifies terms like "Resting BP," "B.P.," and "Blood Pressure" automatically.

### 2. AI Prediction Engine 🧠
- **Ensemble Learning:** Combines **Random Forest**, **HistGradientBoosting**, and **Logistic Regression** via a Soft-Voting Classifier.
- **High Accuracy:** Achieved **86.23% accuracy** with high sensitivity for critical cases.
- **Athero_Score:** A custom engineered feature `(BP * Cholesterol) / 100` to capture compounded risk.

### 3. Enterprise Dashboard 💻
- **Dr. AI Chatbot:** Context-aware assistant that provides specific diet and exercise advice based on the patient's calculated risk level.
- **Live Health Trends:** Interactive **Chart.js** line graphs that plot patient vitals over multiple visits.
- **PDF Reporting:** Generates and downloads official, stamped medical reports for patient records.
- **Dark Mode:** One-click toggle for low-light hospital environments.

---

## 🏗 System Architecture

The project follows a standard **MVC (Model-View-Controller)** pattern:

1.  **Data Acquisition:** User uploads an image -> OCR Engine extracts text.
2.  **Processing:** Data is cleaned -> "Athero_Score" is calculated -> Ensemble Model predicts risk.
3.  **Presentation:** Flask renders the dashboard -> User views results/trends.

---

## 🛠 Tech Stack

| Component | Technology |
| :--- | :--- |
| **Backend** | Python, Flask, Jinja2 |
| **Frontend** | Bootstrap 5, Chart.js, Glassmorphism UI |
| **Database** | SQLite (with Concurrency Locking Fixes) |
| **Machine Learning** | Scikit-Learn (Voting Classifier), Pandas, NumPy |
| **Computer Vision** | Tesseract OCR, PIL (Python Imaging Library), Regex |
| **Reporting** | FPDF (for PDF generation) |

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system.

### Steps

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/yourusername/cardiovision-ai.git](https://github.com/yourusername/cardiovision-ai.git)
   cd cardiovision-ai

```

2. **Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. **Install Dependencies**
```bash
pip install -r requirements.txt

```


4. **Configure Tesseract Path**
Open `milestone1_ocr.py` and update the path if necessary:
```python
# Windows Example
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

```


5. **Run the Application**
```bash
python app.py

```


Visit `http://127.0.0.1:5000` in your browser.

---

## 🕹 Usage

1. **Register/Login:** Create a secure account (Role: Doctor or Patient).
2. **Upload Report:** Click the "Scan" area to upload a JPG/PNG lab report. Watch the **Green Laser Animation** as it scans.
3. **Review Data:** The system will auto-fill the clinical form. Verify the numbers.
4. **Get Diagnosis:** Click "Run Diagnosis."
* **Red Alert:** High Risk (Download PDF immediately).
* **Green Alert:** Low Risk.


5. **Consult Dr. AI:** Click the robot icon to ask: *"What diet should I follow for high cholesterol?"*

---

## 📸 Screenshots

| Dashboard | Health Trends |
| --- | --- |
|  |  |
| *AI Risk Analysis* | *Live Patient History* |

---

## 🔮 Future Scope

* **Cloud Deployment:** Deploying to AWS/Render for global access.
* **Wearable Integration:** Fetching live pulse data from Apple Watch/Fitbit APIs.
* **Mobile App:** React Native version for on-the-go patient monitoring.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

---

## 🙏 Acknowledgements

Developed during the **Infosys Springboard Internship 6.0**.
Special thanks to my mentors for their guidance on Enterprise Architecture and AI Ethics.

```

```
