
<img width="2862" height="1153" alt="Screenshot 2026-01-03 055357" src="https://github.com/user-attachments/assets/c16c766a-5a74-40ee-9e6f-7208310e945d" />

<img width="2879" height="1188" alt="Screenshot 2026-01-02 132700" src="https://github.com/user-attachments/assets/8aed393e-ecce-4d5e-b978-7d0410a1cd4e" />
<img width="2150" height="1210" alt="Screenshot 2025-12-22 213906" src="https://github.com/user-attachments/assets/53864124-b10a-42f7-9219-39361b3ae5a9" />
<img width="2859" height="375" alt="Screenshot 2025-12-22 215804" src="https://github.com/user-attachments/assets/c4eafb30-fc5d-47d7-82b6-776f6c855aa9" />

<img width="2337" height="1543" alt="Screenshot 2025-12-22 221233" src="https://github.com/user-attachments/assets/c6f7658a-d917-48e1-88e0-a3869d8f6d6e" />
<img width="2879" height="1319" alt="Screenshot 2025-12-22 221540" src="https://github.com/user-attachments/assets/2c333637-2ac9-425e-8c53-16c2850d9a6d" />
<img width="2631" height="1027" alt="Screenshot 2025-12-22 222742" src="https://github.com/user-attachments/assets/81399ec5-fb25-4b54-9e33-2774616ba5a7" />

The Smart Attendance System is an AI-driven automated solution designed to mark attendance using Artificial Intelligence and Computer Vision.
The system recognizes students’ faces from live classroom video feeds and automatically records attendance without requiring students to stand in front of a camera or perform any manual action.

This project aims to replace traditional attendance methods such as roll calls, RFID cards, and fingerprint scanners with a contactless, efficient, and intelligent system.

🎯 Objectives

Automate the attendance process using AI

Reduce time wasted during lectures

Prevent proxy attendance

Improve accuracy and reliability

Enable real-time, contactless attendance marking

🛠️ Technologies Used

Programming Language: Python

Computer Vision: OpenCV

Machine Learning / Deep Learning: Face Recognition Models

Libraries: NumPy, Pandas, OpenCV

Database: SQLite / CSV

Interface: Web or Desktop (optional)

⚙️ System Implementation (Working Flow)
Step 1: Dataset Creation

Student face images are collected during registration.

Multiple images are captured under different angles and lighting conditions.

Images are stored with unique student IDs.

Step 2: Face Encoding

Facial features are extracted using AI models.

Encodings are stored for future matching.

Step 3: Live Face Detection

Classroom camera captures real-time video.

Multiple faces are detected simultaneously.

Step 4: Face Recognition

Detected faces are compared with stored encodings.

If a match is found, the student is identified.

Step 5: Attendance Marking

Attendance is marked only if the student is present for a defined duration.

Duplicate entries are avoided.

Step 6: Data Storage

Attendance data is stored with:

Student ID

Date

Time

Session details

🗂️ Project Folder Structure
Smart-Attendance-System/
│
├── dataset/
│   ├── student_1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   ├── student_2/
│
├── encodings/
│   └── face_encodings.pkl
│
├── src/
│   ├── capture_faces.py
│   ├── train_model.py
│   ├── face_recognition.py
│   ├── attendance_marker.py
│
├── database/
│   └── attendance.db
│
├── output/
│   └── attendance_report.csv
│
├── requirements.txt
├── README.md
└── main.py

📐 Data Flow Diagrams (DFDs)
🔹 DFD Level 0 (Context Diagram)

Entities:

Student

Camera

Smart Attendance System

Database

Flow:

Student → Camera → AI System → Attendance Database

🔹 DFD Level 1

Camera
  │
  ▼
[1] Capture Video Feed
  │
  ▼
[2] Detect Faces
  │
  ▼
[3] Recognize Faces ◄──── Student Face Dataset
  │
  ▼
[4] Mark Attendance
  │
  ▼
[5] Store Attendance ────► Attendance Database



🔹 DFD Level 2 (Detailed)

Live Video
   │
   ▼
Frame Extraction
   │
   ▼
Face Detection
   │
   ▼
Feature Extraction
   │
   ▼
Face Matching ◄──── Face Encodings Dataset
   │
   ▼
Attendance Validation
   │
   ▼
Attendance Database



🚀 How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/your-username/smart-attendance-system.git
cd smart-attendance-system

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Register Students
python src/capture_faces.py

4️⃣ Train the Model
python src/train_model.py

5️⃣ Start Attendance System
python main.py

📊 Output

Attendance is saved in:

Database (SQLite)

CSV report for analysis

Each entry contains:

Student ID

Date

Time

Status (Present/Absent)

🔐 Security Considerations

No biometric data is shared externally

Facial data is stored locally

Duplicate attendance entries are prevented

🔮 Future Enhancements

Emotion & engagement detection

Mask-aware recognition

Mobile app integration

Cloud-based analytics dashboard

Late entry & early exit tracking

📌 Conclusion

This AI-Based Smart Attendance System provides an efficient, scalable, and secure alternative to traditional attendance methods. By leveraging Artificial Intelligence and Computer Vision, the system ensures accuracy, saves time, and enhances classroom productivity.
