def get_animation_html(shot_cell: int, keeper_cell: int, outcome: str) -> str:
    """
    Generates an HTML snippet that animates a penalty kick.
    Grid is 5 columns x 3 rows.
    """
    # Grid dimensions (relative)
    cols, rows = 5, 3
    
    # Calculate positions
    def get_pos(cell):
        col = cell % cols
        row = cell // cols
        # Return left and top percentages
        left = 10 + (col * 20)  # 10% to 90%
        top = 20 + (row * 30)   # 20% to 80%
        return left, top

    shot_left, shot_top = get_pos(shot_cell)
    keeper_left, keeper_top = get_pos(keeper_cell)

    ball_color = "#2ecc71" if outcome == "goal" else ("#e74c3c" if outcome == "save" else "#95a5a6")
    msg = "GOAL!" if outcome == "goal" else ("SAVE!" if outcome == "save" else "MISS!")

    html = f"""
    <style>
    .field {{
        position: relative;
        width: 100%;
        height: 300px;
        background-color: #27ae60;
        border-radius: 10px;
        border: 4px solid #fff;
        overflow: hidden;
        font-family: sans-serif;
    }}
    .goal-post {{
        position: absolute;
        top: 10%;
        left: 5%;
        width: 90%;
        height: 85%;
        border: 6px solid #ecf0f1;
        border-bottom: none;
        box-sizing: border-box;
    }}
    .ball {{
        position: absolute;
        width: 30px;
        height: 30px;
        background-color: #fff;
        border-radius: 50%;
        bottom: 5%;
        left: 50%;
        transform: translateX(-50%);
        animation: shoot 1s forwards cubic-bezier(0.25, 1, 0.5, 1);
        z-index: 10;
        box-shadow: inset -5px -5px 10px rgba(0,0,0,0.3);
    }}
    .keeper {{
        position: absolute;
        width: 60px;
        height: 60px;
        background-color: #f1c40f;
        border-radius: 10px;
        bottom: 15%;
        left: 50%;
        transform: translateX(-50%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 30px;
        animation: dive 1s forwards cubic-bezier(0.25, 1, 0.5, 1);
        z-index: 5;
    }}
    .message {{
        position: absolute;
        top: 40%;
        width: 100%;
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: white;
        text-shadow: 2px 2px 4px #000;
        opacity: 0;
        animation: fadein 0.5s forwards 1s;
        z-index: 20;
    }}
    @keyframes shoot {{
        100% {{
            left: {shot_left}%;
            top: {shot_top}%;
            background-color: {ball_color};
            transform: translate(-50%, -50%) scale(0.7);
        }}
    }}
    @keyframes dive {{
        100% {{
            left: {keeper_left}%;
            top: {keeper_top}%;
            transform: translate(-50%, -50%) scale(1.5);
            background-color: #e67e22;
        }}
    }}
    @keyframes fadein {{
        100% {{ opacity: 1; }}
    }}
    </style>
    
    <div class="field">
        <div class="goal-post"></div>
        <div class="keeper">🧤</div>
        <div class="ball"></div>
        <div class="message">{msg}</div>
    </div>
    """
    return html
