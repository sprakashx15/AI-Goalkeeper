# AI-Goalkeeper 🧤⚽

An interactive reinforcement learning dashboard built with Streamlit. Play against an AI Goalkeeper that learns to save your penalty shots using either Q-Learning (Tabular RL) or a Deep Q-Network (Neural Net).

## 🚀 Features

### 🎮 Interactive Gameplay
- **Play vs AI:** Take penalty shots on a 5x3 grid
- Real-time AI predictions and saves
- Visual feedback and animations

### 🤖 AI-Powered Goalkeeper
- Trains using Q-Learning or Deep Q-Network (DQN)
- **Auto-Train:** Train the AI agent against simulated striker profiles (corners, center, random)
- Dynamic learning based on user shot patterns

### 📊 Analytics & Tuning
- **Hyperparameter Tuning:** Experiment with Learning Rate, Gamma, and Epsilon Decay
- Heatmaps of user shots and the goalkeeper's dives
- Visual learning curve of the agent over time

## 🛠️ Tech Stack

**Frontend & Dashboard**
- Streamlit
- Pandas
- Matplotlib

**Machine Learning & AI**
- PyTorch (Deep Q-Network)
- NumPy (Tabular Q-Learning)

## 🧠 How It Works

1. **User enters a shot**
   - Example: Selects Top Right Corner on the 5x3 grid.
2. **AI Goalkeeper processes the state**
   - The RL agent (Q-Learning or DQN) predicts the optimal cell to dive to based on past training.
3. **Environment resolution**
   - The AI dives, and the app visually renders the outcome (Goal or Save).
4. **Learning step**
   - The environment updates the AI's reward, helping it learn and adapt over time.

## 📂 Project Structure
```text
AI-Goalkeeper/
│── agents/          # Q-Learning and DQN agent implementations
│── env/             # Penalty shootout environment logic
│── models/          # Saved neural network weights
│── utils/           # Helper functions and UI animations
│── app.py           # Main Streamlit dashboard application
│── requirements.txt # Python dependencies
│── README.md
│── .gitignore
```

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/sprakashx15/AI-Goalkeeper.git
cd AI-Goalkeeper
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App
```bash
streamlit run app.py
```
Open:
`http://localhost:8501`

## 📸 Screenshots
*Add your project screenshots here.*

**Gameplay**  
*(Insert Gameplay Screenshot)*

**Analytics Dashboard**  
*(Insert Analytics Screenshot)*

## 🔥 Future Enhancements
- Multiplayer mode (User vs User)
- More complex shot physics and mechanics
- Different goalkeeper playing styles (Aggressive, Defensive)
- Cloud deployment with persistent model weights
- Leaderboard system

## 🤝 Contributing
Contributions are welcome!

1. Fork the repository
2. Create a new branch
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to your branch
   ```bash
   git push origin feature-name
   ```
5. Open a Pull Request

## 📄 License
This project is licensed under the MIT License.

## 👨💻 Author
**sprakashx15**
- GitHub: [sprakashx15](https://github.com/sprakashx15)

## 🌟 Acknowledgements
Special thanks to:
- Streamlit
- PyTorch
- NumPy

## ⭐ Support
If you like this project, give it a ⭐ on GitHub!
