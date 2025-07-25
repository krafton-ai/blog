---
layout: distill
title: Can LLM Learn Chess with RL?
description: In recent years, LLMs trained with RL have made remarkable progress in logical reasoning. They have learned to think step by step, reflect on their own errors, and solve challenging tasks like Olympiad-level math problems or debugging complex C++ programs. However, logic alone is not enough for many real-world problems. These problems require strategic reasoning. The ability to plan ahead, anticipate consequences, and make decisions that unfold over time. In this blog post, we explore a simple but important question. Can LLMs learn strategic reasoning through RL? To find out, we look at chess, a game of perfect information, deep tactics, and rich strategic complexity. If a model can learn to play chess well, it may be developing real strategic thinking.
date: 2025-05-12
future: true
htmlwidgets: true
hidden: false


# Anonymize when submitting
authors:
  - name: Dongyoon Hwang*
    url: "https://godnpeter.github.io/"
    affiliations:
      name: Krafton, KAIST
  - name: Hojoon Lee*
    url: "https://joonleesky.github.io/"
    affiliations:
      name: Krafton, KAIST
  - name: Jongho Park
    url: "https://jerryjonghopark.github.io/"
    affiliations:
      name: Krafton, UC Berkely
  - name: Dongmin Park
    url: "https://dongmean.github.io/"
    affiliations:
      name: Krafton
  - name: Jaegul Choo
    url: "https://sites.google.com/site/jaegulchoo/"
    affiliations:
      name: KAIST


# must be the exact same name as your blogpost
bibliography: 2025-05-11-chess-llm.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Why Chess?
  - name: Representing Chess as Text
    subsections:
      - name: FEN (Describing the Board)
      - name: SAN (Describing the Move)
      - name: Do LLMs Understand Chess?
  - name: Zero Shot Evaluation
    subsections:
      - name: Evaluation Tasks
      - name: Let's Try it Out
  - name: Supervised Fine-Tuning
    subsections:
      - name: Experimental Setup
      - name: Let's Try it Out
  - name: RL Fine-Tuning
    subsections:
      - name: Experimental Setup
      - name: Let's Try it Out
      - name: Distillation from Expert
  - name: Lessons and Thought

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
  blockquote {
    border-left: 4px solid gold;
    padding-left: 1em;
    margin-top: 0;
    margin-bottom: 0;
  }
---

In recent years, we’ve seen that **training LLMs with RL can unlock logical reasoning abilities** <d-cite key="guo2025deepseekR1"></d-cite>. When fine-tuned on math and programming tasks, LLMs began to reason step by step, reflect on their own mistakes, and solve complex problems—all without being explicitly programmed to do so.

<div style="margin: 0;">
  This progress raises a deeper question:
  <blockquote style="border-left: 4px solid gold; padding-left: 1em; margin-top: 12px; margin-bottom: 0;">
     Is logical reasoning enough for solving real-world problems?
  </blockquote>
</div>

## Why Chess?

I first got into chess when I was around 10. I’d play after school, sometimes for hours. But my mom didn’t really liked it. She worried it was a distraction from “real” learning. *You should be solving math problems,* she’d say.

But over time, I realized chess was teaching me something valuable. I started approaching problems more systematically. In class debates, I’d think about counterarguments before speaking. When I ran for student council, I planned my campaign like a game; anticipating moves and preparing responses.

Chess wasn’t just about logic. It required planning, foresight, and thinking from the opponent’s perspective. It taught a different kind of reasoning: **strategic reasoning.**

<div style="margin: 0;">
  That brings us to a new kind of question for today’s AI:
  <blockquote style="border-left: 4px solid gold; padding-left: 1em; margin-top: 12px; margin-bottom: 0;">
     If we train an LLM to play chess with RL, can it develop strategic reasoning?
  </blockquote>
</div>

If so, the implications go far beyond the chessboard. From writing long-form code to negotiating, debating, or making complex decisions, many real-world tasks require more than logical accuracy. They require strategy.

Chess might be the proving ground.

So, we gave it a shot.


## Representing Chess as Text

Before we could train anything, we faced a basic but critical challenge:
<blockquote style="border-left: 4px solid gold; padding-left: 1em;">
How do you represent a game of chess in a format an LLM can understand?
</blockquote>
Fortunately, we didn’t have to invent anything new. The chess community has been encoding games in text for decades. We used two standard formats:
- Forsyth–Edwards Notation (FEN): describes the current state of the board.
- Standard Algebraic Notation (SAN): describes the move the model should make.

Together, they give us a clean text interface: FEN as input, SAN as output.

Let’s break these down.


### FEN: Describing the Board

FEN captures the entire state of a chess game in a single line of text. It includes:
- The piece layout
- Whose turn it is
- Castling rights
- En passant target squares
- Move number

The most important part of FEN is the board layout, recorded row by row from Black’s side (top) to White’s side (bottom). Each piece is represented by a letter (K for king, Q for queen, N for knight, R for rook, B for bishop, and P for pawn) with uppercase letters for White and lowercase for Black. Numbers are used to indicate consecutive empty squares, making the representation compact and efficient.

Here’s an example:

<figure style="text-align: center;">
  <img src="{{ 'assets/img/2025-05-15-llm-chess/chess_board.png' | relative_url }}" style="display: inline-block; width: 60%; height: auto;">
  <figcaption style="font-size: 1em;">Figure 1: A midgame board state represented in FEN.</figcaption>
</figure>

can be represented as:
```
r1bk3r/p2pBpNp/n4n2/1p1NP2P/6P1/3P4/P1P1K3/q5b1
```

This compact string tells the model everything it needs to know about the game at this moment.

If you’re curious, the full FEN format is described here: [Wikipedia - FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation).


### SAN: Describing the Move

Once the board is described, we need to tell the model what move to make. That’s where SAN comes in.

Standard Algebraic Notation (SAN) is the compact, human-readable way chess players write moves. It usually includes:
- The piece being moved (omit for pawns),
- The destination square,
- Extra symbols if needed (for captures, checks, etc.).

A few examples:
```
Nf3: knight to f3.
Bxf6: bishop captures on f6.
Qh7#: queen checkmates on h7.
```
It’s short and concise, perfect to tokenize for LLM.


### Do LLMs Understand Chess?

Before we started training, we ran a sanity check. We asked Qwen2.5-7B:

```
Do you know Forsyth–Edwards Notation and Standard Algebraic Notation? 
Please describe in detail.
```

The response?
```
Yes, I'm familiar with both Forsyth–Edwards Notation and Standard Algebraic Notation. 
These are two common systems used in chess to represent positions and moves.

Let me explain each in detail:
They are blah blah...
```

<details markdown="1"><summary>Full Output of explanining FEN</summary>
```
Yes, I'm familiar with both Forsyth–Edwards Notation and Standard Algebraic Notation. 
These are two common systems used in chess to represent positions and moves.
Let me explain each in detail:

## 1. Forsyth–Edwards Notation (FEN)
FEN is a standard notation for describing a particular board position in chess.
It is commonly used in chess software, databases, and engines to store or transmit a specific arrangement of pieces on the board.

### Structure of FEN

A FEN string has six components, separated by spaces:

[Piece placement] [Side to move] [Castling ability] [En passant target square] [Halfmove clock] [Fullmove number]

Let's break down each part:

### 1. Piece Placement
This describes where all the pieces are on the board.

- It starts from **rank 8 (top row)** and goes down to **rank 1**.
- Each rank is described from **left to right (file a to h)**.
- Letters represent pieces:
  - `K` = King
  - `Q` = Queen
  - `R` = Rook
  - `B` = Bishop
  - `N` = Knight
  - `P` = Pawn
- Uppercase = White pieces, lowercase = Black pieces.
- Numbers represent empty squares. For example:
  - `4` means four empty squares.
  - `2` means two empty squares

### Example of Piece Placement

Let’s take the starting position:

rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR
- Each `/` separates the ranks.
- The first rank (8th row for White) is `rnbqkbnr` (Black pieces).
- The third and fourth ranks are empty (`8` each).
- The last rank is `RNBQKBNR` (White pieces).

### 2. Side to Move
This is a single character:
- `w` = White to move
- `b` = Black to move

Example: `w` means it's White's turn.

### 3. Castling Ability
Indicates which castling rights are still available.

- `K` = White can castle kingside
- `Q` = White can castle queenside
- `k` = Black can castle kingside
- `q` = Black can castle queenside
- If no castling is possible, use `-`.

Examples:
- `KQkq` → All castling rights are intact.
- `Kq` → White can only castle kingside; Black can only castle queenside.
- `-` → No castling rights remain.

### 4. En Passant Target Square
This indicates if an en passant capture is possible after a double pawn push.

- It shows the square where the en passant capture can be made.
- If no en passant is possible, use `-`.

Example:
- `e3` means a pawn just moved from e5 to e4, and a pawn on d4 or f4 could capture en passant at e3.
- `-` means no en passant is available.

### 5. Halfmove Clock
Counts the number of halfmoves (plies) since the last capture or pawn advance. This is used for the **50-move rule** in chess.

- Resets to `0` when a pawn moves or a piece is captured.
- Increments by 1 for every other move.

Example: `0` means no captures or pawn moves have occurred in the last 50 moves.

### 6. Fullmove Number
This is the number of full moves that have been played. Starts at `1`, and increases by 1 after each pair of moves (one for White, one for Black).

Example: `1` means the game is just beginning. After White and Black each make a move, it becomes `2`.

### Complete FEN Example

For the **starting position**, the FEN string is:
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

Meaning:
- Board setup as above.
- White to move.
- Castling rights for both sides.
- No en passant.
- 0 halfmoves since last capture/pawn move.
- First full move.
```
</details>

<details markdown="1"><summary>Full Output of explanining SAN</summary>
```
## 2. Standard Algebraic Notation (SAN)

SAN is the standard way to record individual chess moves in a game. It's human-readable and widely used in chess literature, software, and databases.

Each move is written with:
- A piece symbol (optional),
- The destination square,
- Optional modifiers like `x` (capture), `+` (check), `#` (checkmate), `O-O` (castling), etc.

### 📌 Rules of SAN

#### 1. Piece Symbols
- `K` = King
- `Q` = Queen
- `R` = Rook
- `B` = Bishop
- `N` = Knight
- `P` = Pawn (usually omitted)

So, `Ke4` means the king moves to e4.

#### 2. Captures
A capture is denoted with an `x`. For example:
- `Bxf7` = Bishop captures on f7.
- `exd5` = Pawn captures on d5 (from c-file or e-file — more on disambiguation below).

#### 3. Check and Checkmate
- `+` = Check
- `#` = Checkmate

Example:
- `Qh5+` = Queen moves to h5 and gives check.
- `Rf1#` = Rook delivers checkmate on f1.

#### 4. Castling
- `O-O` = Kingside castling
- `O-O-O` = Queenside castling
Note: These are often written using the letter "O" (not zero).

#### 5. Pawn Promotion
When a pawn reaches the 8th rank, it is promoted. The notation includes the destination square followed by the new piece:
- `e8=Q` = Pawn promotes to queen on e8.
- Sometimes written as `e8Q` in shorthand.

#### 6. Disambiguation
If two identical pieces can move to the same square, you need to specify which one:
- Use the file (letter) or rank (number) to distinguish.

Examples:
- `Nbd2` = Knight on b-file moves to d2.
- `Rad1` = Rook on a-file moves to d1.
- If both file and rank are needed: `Rada1` = Rook on a-file and a-rank moves to a1.
```
</details>

\
Fortunately, the model already understands chess notation. It recognizes FEN, parses SAN, and can even discuss the game in abstract terms. 

A promising start.


## Zero-Shot Evaluation

Now that we’ve represented chess in a format language models can understand, the next question is:
<blockquote style="border-left: 4px solid gold; padding-left: 1em;">
How do we actually evaluate whether an LLM can play chess?
</blockquote>

Your first instinct might be to let the model play full games. Match LLM against other engines or humans, and estimate its ELO based on wins and losses.

But full games come with headaches. They’re long (40+ moves), high-variance (a single mistake can swing the result), and slow to evaluate. Worse, the feedback is hard to interpret. Did the model win because it understands strategy, or because the opponent made a mistake?

So instead, we broke down chess into **four targeted evaluation tasks**—each designed to isolate and measure a key chess skill: understanding the rules, updating the board, choosing strong moves, and solving tactical puzzles.

Let’s see what today’s pretrained LLMs can do before we teach them anything new.


### Evaluation Tasks

**1. Legal Move Similarity**
<blockquote style="border-left: 4px solid gold; padding-left: 1em;">
Can the model list all legal moves from a position?
</blockquote>

This is the model’s version of passing a “rules test.” Given a board state in FEN, can it output all valid next moves in SAN format?

Prompt
```
You are a professional chess-playing assistant.
Given a FEN string, list all legal moves in SAN format.
```
Input
```
FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
```
Expected Output
```
f5, e5, d5, Nf6, Nc6, ... (total ~30 legal moves)
```
To evaluate this, we use Jaccard Similarity between the model's predicted move set and the ground truth:
$\text{Jaccard}(A, B) = \frac{|A \cap B|}{|A \cup B|}$

We created a dataset of 100,000 real positions from Lichess games, and used a chess engine to generate the full legal move set for each.


**2. Board State Prediction Accuracy**
<blockquote style="border-left: 4px solid gold; padding-left: 1em;">
Can the model simulate how the board changes after a sequence of moves?
</blockquote>

To generate ground-truth data, we sampled 100,000 real positions from Lichess and used a chess engine to enumerate legal moves.

Prompt
```
You are a professional chess-playing assistant.
Given a FEN string and a series of SAN moves, predict the FEN string after moves.
```
Input
```
FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
SAN moves: f5 Bc5 nh6
```
Expected Output
```
rnbqkbnr/ppppp1pp/7n/2b2p2/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 2 3
```

We generated 100,000 of these examples using real game positions, simulating 1–5 moves per sequence. The model passes if it outputs the exact final FEN.


**3. MATE Puzzle Accuracy**
<blockquote style="border-left: 4px solid gold; padding-left: 1em;">
Can the model choose the better of two candidate moves?
</blockquote>

To assess whether the model has actual chess skill, we use MATE (Move on strAtegy and Tactics datasEt)[1]. MATE is a curated dataset where each puzzle presents two plausible moves. One is clearly superior, as judged by Stockfish with a significant evaluation difference.

This setup enables a clean binary classification task: pick the better move.

Prompt
```
You are a professional chess-playing assistant.
You are given a chess board in FEN format.
Two candidate moves are provided, along with reasoning for each.
Your goal is to select the better move based on strategy and tactics.
```
Input
```
FEN: rnbqkbnr/ppppp1pp/7n/2b2p2/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 2 3

Candidate 1: a5. 
It aims to sacrifice  a piece to clear a file or diagonal adjacent to the enemy king.
Candidate 2: Bc5. 
It offers a piece to unblock a diagonal, potentially enabling threats like Bf7 or Qd3.

Which is better?
```
Expected Output
```
Bc5
```

We evaluate accuracy as the proportion of cases where the model selects the correct move. We used the MATE's validation set consists of 100,000 examples.

**4. LiChess Puzzle Acc.**
<blockquote style="border-left: 4px solid gold; padding-left: 1em;">
Can the model solve real-world tactical puzzles?
</blockquote>

For our final and most realistic benchmark, we turn to the [Lichess Puzzle Database](https://database.lichess.org/). This massive resource includes millions of real-world positions, each ending in a beautiful tactic or checkmate.

These puzzles are ideal for evaluation because:
- Puzzles are grounded in real play.
- They involve non-trivial tactics: Sicilian Defense, Ruy Lopez, Queen's Gambit.
- Human ELO correlates well with performance.

We selected 50,000 puzzles and decomposed them into individual steps. To count as correct, the model must get every move in a puzzle right.

Prompt
```
You are a professional chess-playing assistant.
You are given a chess board with FEN format.
Your task is to analyze the position select the best move. 

Use <think>...</think> tags to explain your reasoning. Including:
  - A strategic evaluation of the position.
  - A comparison of key candidate moves.
  - For each candidate, consider the opponent's likely response and outcome.

Use <answer>...</answer> to give the best move in SAN format.
Only include the move.
```
Input
```
FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
Please think step by step.
```
Expected Output
```
<think>
White has just played e4. 
By responding with f5, Black fights for central control and activates the kingside. 
This also opens lines for future development.
...
</think>
<answer>f5</answer>
```

We report the percentage of puzzles the model solves completely.

### Let's Try it Out

We evaluated two popular open-source model families:
- **LLaMA3** (3B, 8B, 70B).
- **Qwen2.5** (3B, 7B, 70B). 

For baselines, we included:
- A random policy (as a sanity check)
- AlphaZero’s policy network (without search).

So... How did they do?

| Model      | Legal Move Sim. ↑ | Board State Acc. ↑ | MATE Acc. ↑ | Lichess Puzzle Acc. ↑ |
| ---------- | ----------------- | ------------------ | ----------- | --------------------- |
| Random     | 3.0               | 0.0                | 50.0        | 1.0                   |
| AlphaZero (policy-net) | -     | -                  | -           | 56.1%                 |
| LLaMA3-3B  | -                 | -                  | -           | -                     |
| LLaMA3-8B  | -                 | -                  | -           | -                     |
| LLaMA3-70B | -                 | -                  | -           | -                     |
| Qwen2.5-3B   | -                 | -                  | -           | -                     |
| Qwen2.5-7B   | -                 | -                  | -           | -                     |
| Qwen2.5-70B  | -                 | -                  | -           | -                     |


In short, **terrible.**

Across all tasks, pretrained language models struggled badly. Here’s what we observed:
- They often missed legal moves, or worse, invented illegal ones.
- They hallucinated board updates, sometimes placing two kings on the board.
- In tactical puzzles, they consistently picked the worse of two moves—even ones that led directly to checkmate.

Even the largest models like Qwen2.5-70B and LLaMA3-70B struggled with basic mechanics: how a knight moves, or what happens after a capture.

We tried everything: rewording prompts, adding reasoning steps, providing examples, using ASCII boards. Nothing really helped.

The takeaway is clear: **Pretrained LLMs don’t really understand how to play chess.**

Before we can teach them to plan or strategize with RL, we need to teach them the rules. That’s where supervised fine-tuning comes in.

## Supervised Fine-Tuning

Before we jump into RL, there’s a more immediate goal:
**Make the model chess-literate.**

That means teaching LLM to:
- Understand the board,
- Follow the rules,
- And reason through moves like a real player.

We’re not aiming for grandmaster play (yet). 
We just want the model to stop making up nonsense and start thinking like someone who knows how bishops move.


### Experimental Setup

**Dataset**

We built a custom dataset of **1 million training examples** designed to teach the fundamentals of chess. It covers four core skills:
- Legal Move Prediction (200k examples)
- Board State Prediction (200k examples)
- MATE puzzle (300k examples)
- Chess Commentary (300k examples)

Let’s take a look at each.

**1. Legal Move Prediction** 

This task grounds the model in the basic rules. Given a FEN board position, the model must list all legal moves in SAN format.

We sampled real game positions from Lichess and used a chess engine to get the correct move list.

Example:
```
You are a professional chess-playing assistant.
Given a FEN string, list all legal moves in SAN format.
FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
Answer: f5, e5, d5, Nf6, Nc6, ... (total ~30 legal moves)
```

**2. Board state prediction**

This teaches the model to simulate how the game progresses. Given a starting FEN and a short sequence of SAN moves, it must output the final board state in FEN.

Example:
```
You are a professional chess-playing assistant.
Given a FEN string, list all legal moves in SAN format.
FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
SAN moves: f5 Bc5 Nh6
Answer: rnbqkbnr/ppppp1pp/7n/2b2p2/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 2 3
```

**3. MATE Puzzle**

This task is designed to build tactical intuition. Each puzzle presents two reasonable candidate moves, along with explanations for why each one might be good. However, only one move is clearly superior. The model's job is to read the explanations and choose the better move based on strategic and tactical reasoning.

In the MATE dataset, all move explanations were generated by GPT-4 using prompts carefully designed in collaboration with former Women’s World Chess Champion Hou Yifan. This setup ensures that the explanations reflect deep, high-quality chess insight rather than superficial heuristics.

Example:
```
You are a professional chess-playing assistant.
You are given a chess board in FEN format.
Two candidate moves are provided, along with reasoning for each.
Your goal is to select the better move based on strategy and tactics.

FEN: rnbqkbnr/ppppp1pp/7n/2b2p2/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 2 3

<think>
Candidate 1: a5. It aims to sacrifice a piece to clear a file near the enemy king.
Candidate 2: Bc5. It opens a diagonal that could enable threats like Bf7 or Qd3.
</think>
<answer>
Bc5
</answer>
```

**4. Chess Commentary**

This task teaches the model to reason like a human player. That means going beyond labeling a move as good or bad. We want the model to explain why a move works, what might happen if it's played, what alternatives exist, and how the position could unfold.

To make this possible, we turned to GameKnot, an online chess platform where players upload their games and annotate each move with detailed commentary. It is a rich source of strategic thinking, offering insight into how people evaluate positions, anticipate threats, and make decisions.

We collected thousands of annotated games and filtered for comments with at least 50 words. This ensured the examples contained in-depth analysis rather than surface-level explanations.

Example:
```
You are a professional chess-playing assistant.
You are given a FEN string and a SAN move.
Your goal is to explain the move from a strategic perspective.

FEN: rnbqkbnr/ppppp1pp/7n/2b2p2/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 2 3
SAN: Qe7

<think>
The game-losing blunder, but one could hardly blame Thomas for missing this one in blitz! 
10...Bxe5, giving up the Bishop Pair to remove White's monster Knight on e5, is pretty much forced. 
Now in this position, after the text move (10...Qe7??) and the supposed 11.Nxf6 gxf6 (not 11...Rxf6?? 12.Qxh7 Kf8 13.Qh8#) the Black Queen covers h7. 
I actually had the rare privilege of playing this same game against a computer on the Internet Chess Club. 
The crazy thing was the computer, rated around 2100, fell victim to the "horizon effect", 
a well-known phenomenon where a chess engine cannot "see" the final mate position because it simply cannot calculate far enough into the position (beyond its horizon). 
In this position, the engine probably calculated 6 moves ahead and assessed that Black is "safe" after 10...Qe7, since h7 is guarded and there is no way to force mate in 6. 
But there IS a way to force mate in 7, if you look beyond the horizon! Drum roll...
</think>
```

**Training**

We fine-tuned Qwen2.5-7B using LLaMA Factory, with full parameter tuning (no adapters). Here are the details:
- Model: Qwen2.5-7B.
- Data size: 1M.
- Compute: 8× A100 80GB GPUs.

Hyperparameters:
- dype: bf16
- learning rate: 5e-5
- epochs: 3
- batch size: 256
- optimizer: AdamW


### Let's Try it Out

Let’s see how much the model improved after supervised fine-tuning:

| Model      | Legal Move Sim. ↑ | Board State Acc. ↑ | MATE Acc. ↑ | Lichess Puzzle Acc. ↑ |
| ---------- | ----------------- | ------------------ | ----------- | --------------------- |
| Random     | 3.0               | 0.0                | 50.0        | 1.0                   |
| AlphaZero (policy-net) | -     | -                  | -           | 56.1%                 |
| Qwen2.5-7B   | -                 | -                  | -           | -                     |
| Qwen2.5-70B  | -                 | -                  | -           | -                     |
| Qwen2.5-7B-chess-sft| -              | -                  | -           | -                     |


Fine-tuning worked—really well.
- Legal Move Prediction jumped from 11% to 85%.
- Board State Accuracy soared past 90%, showing the model learned game mechanics.
- MATE Accuracy rose to 76%, a huge leap in tactical reasoning.
- Lichess Puzzle Accuracy climbed from <4% to 47%. It gets closer to beginner-level humans.

After fine-tuning, the model no longer invents moves or forgets the rules. It thinks like a chess player—at least for a few moves ahead.

## RL Fine-Tuning

So now, the next question is:
<blockquote style="border-left: 4px solid gold; padding-left: 1em;">
Can RL teach LLMs to think strategically—like a real chess player?
</blockquote>

Time to find out.

### Experimental Setup

To fine-tune our model through RL, we used lichess puzzles as our learning environment. 

**Dataset: Lichess Puzzle**

We used a collection of 300,000 puzzles from the Lichess Puzzle Database, which expands to around 1,000,000 unique board positions. Each puzzle is presented as a reasoning task: the model must analyze the board, explore possible moves, and choose the best one.

prompt
```
You are a professional chess-playing assistant.
You are given a chess board with FEN format.
Your task is to analyze the position select the best move. 

Use <think>...</think> tags to explain your reasoning. Including:
  - A strategic evaluation of the position.
  - A comparison of key candidate moves.
  - For each candidate, consider the opponent's likely response and outcome.

Use <answer>...</answer> to give the best move in SAN format.
Only include the move.

FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
Please think step by step.
```

And the expected output looks like:
```
<think>
Black has several options here, but the most promising is to contest the center immediately...
</think>
<answer>
f7
</answer>
```

**Models: Qwen2.5-7B**

We tested two versions of the model:
- Qwen2.5-7B-base: a pretrained LLM with no chess-specific knowledge.
- Qwen2.5-7B-chess-sft: the same model, but with supervised fine-tuning on our chess tasks.

**Algorithm: GRPO**

To fine-tune our model with RL, we used a GRPO (Group Relative Policy Optimization) algorithm.

GRPO is a policy optimization algorithm designed specifically for language models. Instead of relying on a complex value function to estimate how good each output is, GRPO compares groups of responses and rewards the ones that perform better than their peers.

It’s a bit like running a mini tournament every training step, and picking the winners.

GRPO optimizes the following objective:

$$
J(\theta) = \mathbb{E} \left[ \min \left( \frac{\pi_\theta(a)}{\pi_{\text{old}}(a)} \hat{A}, \text{clip} \left( \frac{\pi_\theta(a)}{\pi_{\text{old}}(a)}, 1 - \epsilon, 1 + \epsilon \right) \hat{A} \right) - \beta \cdot \text{KL}[\pi_\theta || \pi_{\text{ref}}] \right]
$$

where:
* $\pi_\theta$ is the current model’s policy.
* $\pi_{\text{old}}$ is the policy from the last iteration.
* $\hat{A}$ is the relative advantage. How much better a response is compared to others.
* The clipping term prevents overly large updates, ensuring stability.
* The KL penalty keeps the new policy close to a reference policy (e.g., the supervised model), so the model doesn’t drift too far during training.

**Reward**

To guide learning, we defined a composite reward:

$r = \lambda_1 \cdot r_{\text{answer}} + \lambda_2 \cdot r_{\text{format}} + \lambda_3 \cdot r_{\text{language}}$

* Answer Correctness ($r_{\text{answer}}$): 1 if the move is optimal, 0 otherwise.
* Format Validity ($(r_{\text{format}}$): 1 if the response includes both <think> and <answer> tags in the correct format, 0 otherwise.
* Language Consistency ($r_{\text{language}}$): Following DeepSeek, we observed that models sometimes output mixed-language responses. We used spaCy’s `en_core_web_lg` to detect language consistency. If the entire output is in English, the reward is 1; otherwise, 0.

This combination rewards not just correctness but also clear structure and coherent reasoning.


**Training**

We used the VERL framework for fine-tuning, with full parameter updates.
- Model: Qwen2.5-7B or Qwen2.5-7B-chess-sft.
- Data size: 1M.
- Compute: 8× A100 80GB GPUs.

Hyperparameters:
- Max response length: 2048
- Prompts per rollout: 128
- Responses per prompt: 8
- Training steps: 500
- Batch size: 32
- Learning rate: 1e-6
- Optimizer: AdamW
- kl coefficient: 0.001

We explored various hyperparameter settings and report the best-performing configuration above.

### Let's Try it Out

Here’s what happened during training.

[Tensorboard Learning Curve]

Learning curves from TensorBoard show a clear upward trend. As training progresses, the model starts generating longer, more structured reasoning traces, and its accuracy improves from around 10% to 30%.

Let’s check how the model reasons at different stages of training.

**0 steps:** Not good reasoning.

```
<think>
1. The position is in the middle game, with black to move.
2. Black's king is in the center, and the rook on h8 is a potential threat.
3. The black queen is on h7, which is a good position to control the center and the d5 square.
4. The black pawn on g6 is a potential target for the white knight on f3.
5. The white king is on e1, and the white rook on a1 is a potential threat to the black king.
6. The best move for black should be to protect the g6 pawn and control the center.
</think>
<answer> 
g6h7
</answer>
```

**100 steps:** Start to do the breath first search.

```
<think>
Ba5: Defends h2 and supports the pawn on f6.
Bb4: Also supports pawns and opens up Black's king-side.
Ke2: Protects f2 and looks to push with g4 or h4.
Kc2: Protects f2 and looks to push with g4 or h4.
Ke1: Helps with the defense of h2.

Ba5 and Bb4 are strong choices as they both threaten to open up lines and potentially improve the position.
Both moves defend crucial squares (h2 for Ba5 and f2 for Bb4) and open up the bishop for future play. 
The best move here is to defend key squares and prepare for offense with a developed piece.
</think>
<answer>
Ba5
</answer>
```

**500 steps:** More in-depth breath first search. evlauating much more candidates

```
<think>
Candidate Moves:
Rh8: Attacks the white pawn on h2 and supports the rook on f7.
Rg8: Similar to Rh8 but more focused on the center.
Re8: Attacking the white pawn on e2.
Rd8: Supports the rook on f7 and pressures the white pawn on d2.
Rc8: Centers the rook and supports the pawn on c7.
Rb8: Centers the rook and supports the pawn on b7.
Ra8: Centers the rook and supports the pawn on a7.
Rf7: A strong move, pinning the white queen to the king.
Rf6: Centers the rook and supports the pawn on f7.
Rf5: Centers the rook and supports the pawn on f7.
Rf4+: Pins the white queen to the king.
Rf3: Centers the rook and supports the pawn on f7.
Rff2: Supports the rook on f7 and adds pressure.
Rf1: Centers the rook and adds pressure.
Kh8: Just moves the king to safety.
Kg8: Moves the king to safety.
Rg5: Centers the rook and adds pressure.
Rg4+: Attacks the white king and pins the queen.
Rg3: Centers the rook and adds pressure.
Rh2: Supports the rook on f7 and adds pressure.
Rgf2: Supports the rook on f7 and adds pressure.
Re2+: Attacks the white king and adds pressure.
Rd2: Supports the rook on f7 and adds pressure.
Rc2: Centers the rook and adds pressure.

Evaluation of Candidate Moves:
Rf7 and Rf4+ are strong as they pin the white queen to the king, putting heavy pressure.
Rg4+ is particularly strong as it pins the white queen and aims at the king.
h5 and g5 are strong as they weaken key pawns for the white side.
The other moves either center the rook or add pressure to pawns but do not offer the same strategic benefit as pinning the queen.
Final Choice:
Rg4+ is the best move as it pins the white queen to the king, offering an immediate threat. 
</think>
<answer>
Rg4+
</answer>
```



But that was it, even after much longer training, **performance plateaus at 30%.** 
Not so great.



However, even we trained much more still bfs. not dfs!!! not anticipating future moves.

What's the problem?

maybe chess puzzle is too difficult?
It may not recieve the good signal at all? 
since average move has around 30 possible candidate per move maybe model struggle to get the feedback for not selecting the **optimal move** while it still provides quite reasonable sub-optimal move.

Thus, we desinged a better way to design reward model so that language model can recieve signal for suboptimal move.

### Recieving Reward from an Expert

Goal was to provide reward of chosing better move than previously even if it does not 



[Tensorboard Learning Curve]




## Lessons and Thoughts







# Acknowledgments

Thank you for Taehwan Kwon and Jungkwan Lee for helpful discussion of this work. 




