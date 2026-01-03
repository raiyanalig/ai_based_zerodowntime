Ai Based autoscaling feauture is an automated solution designed to mark attendance by identifying individuals through their facial features. Traditional attendance methods such as manual registers, RFID cards, and fingerprint machines are time-consuming, prone to proxy attendance, unhygienic, and inefficient for large organizations.

This project leverages Computer Vision and Machine Learning techniques to recognize faces in real time and automatically record attendance in a secure database. The system provides a fast, contactless, and accurate attendance solution, suitable for educational institutions and organizations.

🎯 Objectives

To automate the attendance process using face recognition technology

To eliminate proxy attendance and ensure accurate identification of individuals

To store and manage attendance records digitally in a database

To generate daily, weekly, and monthly attendance reports

To reduce manual effort and improve operational efficiency

⚙️ Methodology

The Smart Attendance System follows a structured workflow:

Dataset Collection
Multiple face images of each user are captured using a webcam and stored in a dataset directory.

Image Preprocessing
Captured images are converted to grayscale and face regions are extracted for better recognition accuracy.

Model Training
The LBPH (Local Binary Patterns Histogram) algorithm is used to train a face recognition model using labeled face images.

Real Time Face Detection & Recognition
The trained model detects faces in real time and matches them with stored identities.

Attendance Marking
On successful recognition, attendance is automatically recorded with user ID, date, and timestamp.

Data Storage & Report Generation
Attendance data is stored securely in a database and can be used for report generation.

🛠 Tools & Technologies Used

Python – Core programming language

OpenCV – Face detection and recognition

NumPy & Pandas – Data processing

MySQL / SQLite – Attendance database

LBPH Algorithm – Face recognition model

Git & GitHub – Version control and collaboration



🧾 Git Commands Used

The following Git commands were used during the project:



git init
git status
git add .
git commit -m "Commit message"
git branch
git branch feature
git branch test
git branch bugfix
git branch experiment
git checkout branch_name
git merge branch_name
git remote add origin <repository-url>
git push -u origin main
git pull origin main
git clone <repository-url>
git log --oneline



The Smart Attendance System using Face Recognition successfully demonstrates the use of Computer Vision and Machine Learning for automating attendance processes. The system eliminates manual errors, prevents proxy attendance, and improves efficiency by providing a secure and contactless solution.

Additionally, this project provided hands-on experience with Git Bash, GitHub, branching, merging, conflict resolution, and documentation best practices, making it a complete and industry-relevant implementation.
