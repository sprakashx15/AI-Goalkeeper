import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from env.penalty_env import PenaltyEnv, CELL_NAMES
from agents.q_learning import QLearningAgent
from utils.streamlit_anim import get_animation_html
import os

# --- Page Config ---
st.set_page_config(page_title="RL Goalkeeper Dashboard", layout="wide", page_icon="🧤")

# --- Session State Initialization ---
if 'env' not in st.session_state:
    st.session_state.env = PenaltyEnv(striker_strategy="human")
    
if 'agent' not in st.session_state:
    st.session_state.agent = QLearningAgent(n_states=16, n_actions=15, epsilon=0.1) # Small epsilon for slight exploration
    # Load if exists
    if os.path.exists("models/q_table.pkl"):
        try:
            st.session_state.agent.load("models/q_table.pkl")
            if st.session_state.agent.Q.shape != (16, 15):
                # Reset if old model
                st.session_state.agent.Q = np.zeros((16, 15))
                raise ValueError("Old model shape detected, starting fresh.")
            st.success("Loaded pre-trained Goalkeeper agent!")
        except Exception as e:
            st.warning("Failed to load pre-trained agent or shape mismatch. Starting fresh.")
            
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

st.title("⚽ Goalkeeper  Agent Dashboard")
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
                        agent.update(state, action, reward, next_state, done)
                        
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
    
    striker_profile = st.selectbox("Striker Profile", ["corners", "center", "random"])
    train_eps = st.number_input("Training Episodes", min_value=100, max_value=10000, value=1000, step=100)
    
    if st.button("Start Training"):
        with st.spinner(f"Training AI Goalkeeper against '{striker_profile}' striker for {train_eps} episodes..."):
            train_env = PenaltyEnv(striker_strategy=striker_profile, miss_prob=0.05)
            agent = st.session_state.agent
            agent.epsilon = 1.0 # boost exploration for training
            
            for _ in range(train_eps):
                s = train_env.reset()
                a = agent.select_action(s)
                ns, r, d, i = train_env.step(a)
                agent.update(s, a, r, ns, d)
                agent.decay_epsilon()
                
            agent.save("models/q_table.pkl")
        st.success("Training Complete! The Goalkeeper is now smarter. Go play against it!")
