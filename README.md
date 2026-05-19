# RL Goalkeeper Dashboard 🧤⚽

An interactive reinforcement learning dashboard built with Streamlit. Play against an AI Goalkeeper that learns to save your penalty shots using either Q-Learning (Tabular RL) or a Deep Q-Network (Neural Net).

## Features
- **Play vs AI**: Take penalty shots on a 5x3 grid and see if the AI can predict your moves!
- **Auto-Train**: Train the AI agent against simulated striker profiles (corners, center, random).
- **Hyperparameter Tuning**: Experiment with Learning Rate, Gamma (discount factor), and Epsilon Decay.
- **Analytics**: View heatmaps of your shots and the goalkeeper's dives, as well as the agent's learning curve over time.

## Installation & Local Usage

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd RL_PROJECT
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

## Deployment on Streamlit Community Cloud

This app is ready to be deployed for free on [Streamlit Community Cloud](https://share.streamlit.io/).

1. Push this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New app"**.
4. Select your repository, branch (`main`), and main file path (`app.py`).
5. Click **"Deploy"**.

*Note: Since the file system on Streamlit Cloud is ephemeral, any models trained online will reset when the app sleeps or redeploys.*
