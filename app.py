import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from env.penalty_env import PenaltyEnv, CELL_NAMES
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent, TORCH_AVAILABLE
from utils.streamlit_anim import get_animation_html
import os

# --- Page Config ---
st.set_page_config(page_title="RL Goalkeeper Dashboard", layout="wide", page_icon="🧤")

# --- Sidebar Settings ---
with st.sidebar:
    st.header("Agent Settings")
    agent_type = st.selectbox("Select Model", ["Q-Learning (Tabular)", "DQN (Neural Net)"])
    if agent_type == "DQN (Neural Net)" and not TORCH_AVAILABLE:
        st.error("PyTorch not installed. DQN will not work. Please install torch.")
        st.stop()

# --- Session State Initialization ---
if 'env' not in st.session_state:
    st.session_state.env = PenaltyEnv(striker_strategy="human")

def init_agent(agent_type_str):
    if agent_type_str == "DQN (Neural Net)":
        agent = DQNAgent(n_states=16, n_actions=15, epsilon=0.1)
        model_path = "models/dqn.pt"
    else:
        agent = QLearningAgent(n_states=16, n_actions=15, epsilon=0.1)
        model_path = "models/q_table.pkl"
    
    if os.path.exists(model_path):
        try:
            agent.load(model_path)
            if agent_type_str == "Q-Learning (Tabular)" and agent.Q.shape != (16, 15):
                agent.Q = np.zeros((16, 15))
                raise ValueError("Old model shape detected.")
            st.success(f"Loaded pre-trained {agent_type_str}!")
        except Exception:
            pass # Start fresh
    return agent

# Re-init agent if type changes
if 'agent_type' not in st.session_state or st.session_state.agent_type != agent_type:
    st.session_state.agent_type = agent_type
    st.session_state.agent = init_agent(agent_type)

if 'state' not in st.session_state:
    st.session_state.state = st.session_state.env.reset()

if 'history' not in st.session_state:
    st.session_state.history = {"goals": 0, "saves": 0, "misses": 0}
    st.session_state.shot_log = []
    st.session_state.keeper_log = []
    st.session_state.last_anim = None

# --- Custom CSS ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e1e2f;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
    }
    .goal-text { color: #2ecc71; }
    .save-text { color: #e74c3c; }
    .miss-text { color: #95a5a6; }
    </style>
""", unsafe_allow_html=True)

st.title("⚽ Goalkeeper RL Agent Dashboard")
st.markdown("Play against an AI Goalkeeper that learns to save your penalties!")

tab_play, tab_analytics, tab_train = st.tabs(["🎮 Play vs AI", "📊 Analytics", "⚙️ Train & Settings"])

# --- TAB: PLAY ---
with tab_play:
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("Penalty Area (You are the Striker)")
        st.write("Click a cell to shoot!")
        
        # Grid of buttons (3 rows x 5 columns)
        for r in range(3):
            cols = st.columns(5)
            for c in range(5):
                cell_idx = r * 5 + c
                with cols[c]:
                    if st.button(f"{CELL_NAMES[cell_idx]}\n(Cell {cell_idx})", key=f"btn_{cell_idx}", use_container_width=True):
                        # Play turn
                        state = st.session_state.state
                        agent = st.session_state.agent
                        env = st.session_state.env
                        
                        # Keeper decides
                        action = agent.select_action(state, greedy=False) # allow slight exploration
                        
                        # Step environment
                        next_state, reward, done, info = env.step(action, human_shot=cell_idx)
                        
                        # Update Agent
                        if isinstance(agent, QLearningAgent):
                            agent.update(state, action, reward, next_state, done)
                        else:
                            agent.store(state, action, reward, next_state, done)
                            agent.update()
                            agent.end_episode()
                        
                        # Save logs
                        outcome = info["outcome"]
                        st.session_state.history[outcome + "s" if outcome != "miss" else "misses"] += 1
                        st.session_state.shot_log.append(cell_idx)
                        st.session_state.keeper_log.append(action)
                        st.session_state.state = env.reset() # Reset for next
                        
                        # Generate animation
                        st.session_state.last_anim = get_animation_html(cell_idx, action, outcome)
        
        if st.session_state.last_anim:
            st.components.v1.html(st.session_state.last_anim, height=350)
            
    with col_right:
        st.subheader("Scoreboard")
        h = st.session_state.history
        total = sum(h.values())
        
        st.markdown(f"""
            <div class="metric-card">
                <h3>Saves (AI)</h3>
                <div class="metric-value save-text">{h['saves']}</div>
            </div>
            <br>
            <div class="metric-card">
                <h3>Goals (You)</h3>
                <div class="metric-value goal-text">{h['goals']}</div>
            </div>
            <br>
            <div class="metric-card">
                <h3>Misses</h3>
                <div class="metric-value miss-text">{h['misses']}</div>
            </div>
        """, unsafe_allow_html=True)
        
        if total > 0:
            st.progress(h['saves'] / total, text=f"AI Save Rate: {h['saves']/total*100:.1f}%")

# --- TAB: ANALYTICS ---
with tab_analytics:
    st.subheader("Session Analytics")
    
    if 'train_rates' in st.session_state:
        st.markdown("#### Latest Auto-Training Performance")
        fig3, ax3 = plt.subplots(figsize=(10, 3))
        ax3.plot(np.arange(1, len(st.session_state.train_rates)+1)*100, st.session_state.train_rates, marker='o', color='#3498db')
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Save Rate")
        ax3.set_title("Agent Learning Curve (Per 100 Episodes)")
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)
        st.divider()

    if len(st.session_state.shot_log) == 0:
        st.info("Play some rounds to see analytics!")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Your Shot Heatmap")
            grid = np.zeros((3, 5))
            for s in st.session_state.shot_log:
                grid[s // 5, s % 5] += 1
            fig1, ax1 = plt.subplots()
            cax1 = ax1.imshow(grid, cmap="Greens")
            ax1.set_xticks(range(5))
            ax1.set_yticks(range(3))
            ax1.set_xticklabels(["L1", "L2", "C", "R2", "R1"])
            ax1.set_yticklabels(["Top", "Mid", "Bot"])
            for i in range(3):
                for j in range(5):
                    ax1.text(j, i, int(grid[i, j]), ha="center", va="center", color="black")
            st.pyplot(fig1)
            
        with c2:
            st.markdown("#### Keeper Dive Heatmap")
            grid2 = np.zeros((3, 5))
            for k in st.session_state.keeper_log:
                grid2[k // 5, k % 5] += 1
            fig2, ax2 = plt.subplots()
            cax2 = ax2.imshow(grid2, cmap="Reds")
            ax2.set_xticks(range(5))
            ax2.set_yticks(range(3))
            ax2.set_xticklabels(["L1", "L2", "C", "R2", "R1"])
            ax2.set_yticklabels(["Top", "Mid", "Bot"])
            for i in range(3):
                for j in range(5):
                    ax2.text(j, i, int(grid2[i, j]), ha="center", va="center", color="white")
            st.pyplot(fig2)

# --- TAB: TRAIN ---
with tab_train:
    st.subheader("Auto-Train the AI Goalkeeper")
    st.write("You can fast-track the Goalkeeper's learning by training it against an automated Striker profile.")
    
    c1, c2 = st.columns(2)
    with c1:
        striker_profile = st.selectbox("Striker Profile", ["corners", "center", "random"])
        train_eps = st.number_input("Training Episodes", min_value=100, max_value=10000, value=1000, step=100)
    with c2:
        st.markdown("#### Hyperparameters")
        learning_rate = st.slider("Learning Rate", 0.001, 0.5, 0.1 if agent_type=="Q-Learning (Tabular)" else 0.001, format="%.3f")
        gamma = st.slider("Gamma (Discount)", 0.5, 0.99, 0.9)
        epsilon_decay = st.slider("Epsilon Decay", 0.9, 0.999, 0.995, format="%.3f")
    
    if st.button("Start Training"):
        with st.spinner(f"Training {agent_type} against '{striker_profile}' striker for {train_eps} episodes..."):
            train_env = PenaltyEnv(striker_strategy=striker_profile, miss_prob=0.05)
            agent = st.session_state.agent
            
            # Apply hyperparams
            if isinstance(agent, QLearningAgent):
                agent.lr = learning_rate
                agent.gamma = gamma
            else:
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = learning_rate
                agent.gamma = gamma
                
            agent.epsilon = 1.0 # boost exploration for training
            agent.epsilon_decay = epsilon_decay
            
            save_rate_history = []
            saves = 0
            
            for ep in range(train_eps):
                s = train_env.reset()
                a = agent.select_action(s)
                ns, r, d, i = train_env.step(a)
                
                if isinstance(agent, QLearningAgent):
                    agent.update(s, a, r, ns, d)
                else:
                    agent.store(s, a, r, ns, d)
                    agent.update()
                    agent.end_episode()
                    
                agent.decay_epsilon()
                
                if r > 0: saves += 1
                if (ep + 1) % 100 == 0:
                    save_rate_history.append(saves / 100.0)
                    saves = 0
                
            if isinstance(agent, QLearningAgent):
                agent.save("models/q_table.pkl")
            else:
                agent.save("models/dqn.pt")
                
            st.session_state.train_rates = save_rate_history
            
        st.success("Training Complete! The Goalkeeper is now smarter. Check the Analytics tab for the learning curve!")
