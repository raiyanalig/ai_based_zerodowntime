# smart-attendance-face-recognition
<img width="2862" height="1153" alt="Screenshot 2026-01-03 055357" src="https://github.com/user-attachments/assets/c16c766a-5a74-40ee-9e6f-7208310e945d" />

<img width="2879" height="1188" alt="Screenshot 2026-01-02 132700" src="https://github.com/user-attachments/assets/8aed393e-ecce-4d5e-b978-7d0410a1cd4e" />

<img width="2337" height="1543" alt="Screenshot 2025-12-22 221233" src="https://github.com/user-attachments/assets/c6f7658a-d917-48e1-88e0-a3869d8f6d6e" />
<img width="2879" height="1319" alt="Screenshot 2025-12-22 221540" src="https://github.com/user-attachments/assets/2c333637-2ac9-425e-8c53-16c2850d9a6d" />
<img width="2631" height="1027" alt="Screenshot 2025-12-22 222742" src="https://github.com/user-attachments/assets/81399ec5-fb25-4b54-9e33-2774616ba5a7" />
The Smart Attendance System Using Face Recognition is an automated solution designed to mark attendance by identifying individuals through their facial features. Traditional attendance methods such as manual registers, RFID cards, and fingerprint-based systems are time-consuming, prone to proxy attendance, unhygienic, and inefficient for large groups.

This project leverages Computer Vision and Machine Learning to recognize faces in real time and automatically record attendance into a digital database. Along with application development, this project strongly focuses on Git Bash and GitHub version control practices, following an industry-standard workflow.

🎯 Objectives of the Project

To automate the attendance process using face recognition technology

To eliminate proxy attendance and ensure accurate identification

To store attendance records digitally in a database

To handle real-world scenarios like early exit or temporary absence

To gain hands-on experience with Git Bash and GitHub workflows

To implement branching, merging, and conflict resolution

🛠 Tools & Technologies Used

Python – Core programming language

OpenCV – Face detection and recognition

LBPH Algorithm – Face recognition model

NumPy & Pandas – Data processing

SQLite / MySQL – Attendance database

Git Bash – Local version control

GitHub – Remote repository management

⚙️ Project Methodology

Dataset Collection
Face images are captured using a webcam and stored in a dataset directory.

Image Preprocessing
Images are converted to grayscale and face regions are extracted.

Model Training
The LBPH algorithm is trained using labeled face images.

Real-Time Face Recognition
Faces are detected and matched with trained data during live camera input.

Attendance Marking
Attendance is recorded automatically with ID, date, and timestamp.

Presence Verification
Attendance is validated based on presence duration to handle early exit cases.

📁 Project Folder Structure
attendance/
├── dataset/                 # Captured face images
├── src/
│   ├── dataset_capture.py   # Face dataset collection
│   ├── train_model.py       # Model training
│   ├── recognize_face.py    # Face recognition & attendance
│   └── database.py          # Database logic
├── trainer/                 # Trained model files
├── .gitignore               # Ignored files & folders
├── README.md                # Project documentation
└── requirements.txt         # Dependencies

🌟 Unique Features of the Project

Confidence-based face recognition to prevent false detection

Duplicate attendance prevention per session

Presence verification window to handle washroom/early exit cases

Entry–exit based attendance logic

Secure and automated attendance workflow

Industry-level Git & GitHub usage

🔄 Git & GitHub Workflow

This project strictly follows professional Git and GitHub practices:

🔹 Repository Initialization
git init

🔹 Staging & Commit
git add .
git commit -m "Meaningful commit message"


Minimum 10 meaningful commits were created

Each commit represents a logical change

🌿 Branching Strategy

The following branches were created:

main – Stable production branch

feature – New feature development

test – Testing changes

bugfix – Bug fixing

experiment – Experimental changes

git branch feature
git branch test
git branch bugfix
git branch experiment


All branches were pushed to GitHub.

🔀 Merge & Merge Conflict Resolution

Branches were merged into the main branch

A merge conflict was intentionally created in README.md

Conflict was resolved manually by editing the file

Final changes were committed successfully

This demonstrates real-world Git conflict handling.

🌍 GitHub Remote Operations

The local repository was connected to GitHub using SSH.

🔹 Remote Connection
git remote add origin <repository-url>
git remote -v

🔹 Push Operation
git push -u origin main

🔹 Pull Operation
git pull origin main

🔹 Clone Operation
git clone <repository-url>
