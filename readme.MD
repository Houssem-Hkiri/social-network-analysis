# 🧠 SOCIAL NETWORK ANALYSIS USING GRAPH THEORY  
### _Identifying Influencers, Modeling Diffusion, and Analyzing Collective Behavior_

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/Python-3.9+-informational)](https://www.python.org/)  
[![Streamlit App](https://img.shields.io/badge/Visualization-Dashboard-brightgreen)](#dashboard)

---

## 📖 Overview

This project explores how **influence and information spread** across social networks using **graph theory** and **diffusion models**.  
By combining structural metrics with dynamic simulations, it identifies key influencers and analyzes the cascade behavior of information flow.

> 🏫 Project developed as part of a 2nd-year industrial engineering course at ENIT, under the supervision of Ms. Myriam Gharbi and Ms. Nabila Bitri.

---

## 🎯 Objectives

- Identify **key influencers** using multi-metric centrality analysis  
- Simulate **information diffusion** using Independent Cascade and Linear Threshold models  
- Build an **interactive dashboard** for live simulation  
- Compare static and dynamic influence scores  
- Provide **actionable insights** for marketing, public health, and information control

---

## 🧠 Methodology

### 📊 Centrality Analysis

- Degree, Betweenness, Closeness, PageRank  
- Composite Influencer Score = Normalized average of all centralities

### 🔁 Diffusion Models

- Independent Cascade (IC) → Probabilistic one-shot influence  
- Linear Threshold (LT) → Cumulative influence activation  
- Monte Carlo Simulations → Averaging influence across 20+ randomized diffusion runs

### 🧪 Datasets

- Facebook Circles (4039 nodes / 88234 edges)  
- Twitter Ego Networks (973 ego-centric graphs)

---

## 🛠️ Tech Stack

- Python 3.9+  
- NetworkX  
- NumPy  
- Plotly  
- Streamlit  
- Matplotlib

---

## 📊 Dashboard Features

### ▶️ Run the App

To launch the dashboard locally, run:

    streamlit run dashboard.py

### 🧩 Functionalities

- Dataset selection  
- Choose centrality metrics  
- Launch IC / LT diffusion models  
- Visualize cascades and node activations  
- View influencer rankings with average spread

---

## 📂 Project Structure

- `centrality.py` – Centrality measures  
- `data.py` – Dataset loading and cleaning  
- `diffusion.py` – IC and LT models + Monte Carlo  
- `metrics.py` – Composite influence metrics  
- `visualization.py` – Graph and cascade plots  
- `dashboard.py` – Streamlit app entry point  
- `requirements.txt` – All dependencies  
- `README.md` – This documentation  

---

## 📈 Key Results

| Metric         | Top Node | Composite Score | Avg Spread (IC) | Avg Spread (LT) |
|----------------|----------|-----------------|------------------|------------------|
| Composite      | Node 107 | 0.3015          | 2561             | 264              |
| Degree         | Node 351 | 0.2335          | 2931             | 277              |
| Diffusion      | Node 352 | -               | 2971             | -                |

✅ Statistical Test: Diffusion from top influencers outperforms random seeds significantly (**p < 0.001**)

---

## 📌 Takeaways

- Centrality metrics reliably identify key influencers  
- IC → Fast, wide cascades. LT → Local, threshold-based influence  
- Static and dynamic metrics complement each other  
- Simulation results validated by Monte Carlo analysis

---

## 🔭 Future Work

- Add temporal network support  
- Improve edge-weight estimation and threshold tuning  
- Explore GNNs for influence prediction  
- Conduct cross-platform comparisons (e.g., LinkedIn, Reddit)

---

## 🙏 Acknowledgments

Thanks to our supervisors **Ms. Myriam Gharbi** and **Ms. Nabila Bitri** for their guidance and support throughout this project.

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.
