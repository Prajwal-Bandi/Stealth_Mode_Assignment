# 🏃‍♂️ Player Re-Identification using YOLOv8 and DeepSORT

This project is a **real-time sports analytics system** that performs player re-identification from a single video feed. It ensures that players who leave and re-enter the frame retain consistent IDs, simulating real-world tracking scenarios.

---

## 🎯 Objective

To identify and track players in a 15-second video using:
- A custom-trained YOLOv8 model for player detection
- DeepSORT for re-identification and ID consistency

---

## 📁 Project Structure


---

## ⚙️ Installation & Setup

### 🧪 Create a Virtual Environment (Optional)
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

pip install ultralytics==8.0.50
pip install opencv-python
pip install deep_sort_realtime


model_path = r'C:\path\to\best.pt'
video_path = r'C:\path\to\15sec_input_720p.mp4'
output_path = r'C:\path\to\output.mp4'

python main.py
