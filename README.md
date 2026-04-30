# 🏀 Aggie Hoops Analytics Platform

## 📌 Project Overview
Aggie Hoops Analytics is a full-stack basketball analytics platform developed as part of the **COMP 496 – Senior Project II** course at **North Carolina Agricultural & Technical State University**.

This system is designed to transform raw basketball data into actionable insights that help **coaches and players prepare for games, analyze performance, and make smarter decisions**.

The platform integrates:
- Data collection  
- Feature engineering  
- Predictive modeling  
- Interactive visualization  

All into one unified system.

A key strength of this platform is its ability to support **both Men’s and Women’s basketball programs**, making it scalable, flexible, and applicable across multiple teams.

---

## 🎯 Purpose
The goal of Aggie Hoops Analytics is to bridge the gap between raw statistics and real basketball decisions by:

- Centralizing all team data into one system  
- Helping coaches prepare for opponents using analytics  
- Giving players insight into their performance and development  
- Providing predictive tools for game strategy  
- Improving overall team efficiency and decision-making  

---

## 🏀 Real-World Impact

### 🧠 For Coaches
Aggie Hoops Analytics provides coaches with tools to:
- Analyze opponent tendencies using historical data  
- Identify optimal lineups and rotations  
- Break down player efficiency and performance trends  
- Generate data-backed scouting insights  
- Prepare for games with predictive analytics  
- Make faster and more informed in-game decisions  

---

### 💪 For Players
Players can use the platform to:
- Track individual performance over time  
- Understand strengths and weaknesses  
- Improve shot selection and efficiency  
- Receive data-driven feedback for development  
- Prepare for matchups using performance trends  

---

## 🚀 Core Features

### 📊 Dual Data Pipelines (Men’s & Women’s Teams)
- Separate processing pipelines for:
  - Men’s basketball data  
  - Women’s basketball data  
- Ensures scalability and consistency across programs  
- Handles:
  - Team stats generation  
  - Data updates  
  - Feature extraction  

---

### 📈 Interactive Dashboard & Visualization
- Converts raw data into intuitive charts and graphs  
- Provides quick insights into team and player performance  
- Designed for non-technical users (coaches & players)

---

### 🤖 Predictive Analytics Engine
- Machine learning models predict player performance  
- Identifies high-impact players and trends  
- Supports pre-game strategy and planning  

---

### 🧠 Feature Engineering System
- Cleans and transforms raw basketball data into meaningful metrics  
- Includes:
  - Efficiency ratings  
  - Performance indicators  
  - Custom analytics  

---

### 🏀 Strategy Lab (Game Preparation Tool)
- Interactive frontend tool for analyzing game scenarios  
- Allows:
  - Lineup experimentation  
  - Matchup analysis  
  - Tactical decision-making  
- Supports both pre-game and post-game analysis  

---

### 🔔 Notification System (Planned)
- Alerts for:
  - Game updates  
  - Data uploads  
  - Performance insights  

---

## 🛠️ Tech Stack

### Frontend
- HTML5  
- CSS3  
- JavaScript  
- (Optional: Chart.js for visualization)

### Backend / Data Processing
- Python  

### Data Storage
- JSON / Flat-file system  

### Tools & Environment
- GitHub  
- WebStorm / VS Code  
- Node.js (for future backend expansion)

---

## 📂 Project Structure

AggieHoopsAnalyticsDemo/

├── 📁 Frontend
│   ├── home.html
│   ├── welcome.html
│   └── strategy-lab.html

├── 📁 Data Processing (Men’s & Women’s)
│   ├── build_team_stats_json.py
│   ├── build_team_stats_json_womens.py
│   ├── feature_engineering.py
│   └── feature_engineering_womens.py

├── 📁 Machine Learning
│   ├── model_player_performance.py
│   └── model_player_performance_womens.py

├── 📁 Data Collection & Updates
│   ├── scrape_pbp_features.py
│   ├── scrape_pbp_features_womens.py
│   ├── update_boxscores.py
│   ├── update_boxscores_womens.py
│   └── update_stats.py

├── 📁 Assets
│   └── assets/

├── 📁 Config
│   └── package.json

└── 📁 Misc
    ├── .vscode/
    └── .idea/

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
git clone https://github.com/AJThomasNCAT/AggieHoopsAnalyticsDemo.git  
cd AggieHoopsAnalyticsDemo  

---

### 2. Install Dependencies (Optional)
npm install  

---

### 3. Run the Frontend
- Open `home.html` in your browser  
OR  
- Use Live Server (recommended)

---

## ▶️ Running the Analytics System (Python)

Make sure Python is installed, then run:

python build_team_stats_json.py  
python feature_engineering.py  
python model_player_performance.py  

### 🔍 What Happens:
- Data is collected and processed  
- Features are engineered  
- Team stats are generated  
- Predictive models run  

---

## 👤 User Roles

### Coaches
- View team analytics  
- Analyze opponent data  
- Prepare game strategies  

### Players
- Track performance  
- Analyze trends  
- Improve gameplay  

---

## 🔮 Future Enhancements
- Full backend API (Node.js / Express)
- Real-time data updates  
- Secure login system (Coach/Player roles)  
- Mobile application (React Native)  
- Advanced AI-driven insights  
- Integration with live game feeds  

---

## 👥 Development Team

- **Ahmad Thomas** – Frontend Development, Data Processing, System Design  
- **Emmanuel Lemi** – Data Pipeline & Analytics  
- **Miles Johnson** – Visualization & UI  
- **Bryan Carbajal Albarran** – Data Engineering  
- **Kenton Moore** – Data Processing, System Design, System Integration & Testing  

---

## 📅 Project Information

- **Course:** COMP 496 – Senior Project II  
- **Semester:** Spring 2026  
- **Institution:** North Carolina Agricultural & Technical State University  


---

## 📄 License
This project is developed for academic purposes and may be expanded for real-world deployment in the future.

---

## ⭐ Final Note
Aggie Hoops Analytics is more than just a class project — it is a scalable analytics platform designed to bring **data-driven decision-making into basketball programs**, helping teams gain a competitive edge through technology.
