🔍 Social Network Analysis using Graph Theory
Identifying Influencers, Modeling Diffusion, and Analyzing Collective Behavior


📖 Overview
This project investigates how information and influence propagate through social networks using graph theory and diffusion models. It combines static centrality measures (Degree, Betweenness, Closeness, PageRank) with dynamic simulations (Independent Cascade and Linear Threshold models) to identify key influencers and simulate cascade behaviors in large-scale real-world datasets.

Developed as part of a 2nd-year industrial engineering project at ENIT – supervised by Ms. Myriam Gharbi and Ms. Nabila Bitri.

🎯 Objectives
Identify key influencers in social networks using multi-metric centrality analysis.

Simulate information diffusion using Independent Cascade (IC) and Linear Threshold (LT) models.

Build an interactive dashboard for visual analytics.

Compare static influence measures with Monte Carlo-based dynamic simulations.

Provide actionable insights for marketing strategies, public health campaigns, and misinformation control.

🧠 Methodology
📊 Centrality Analysis
Degree Centrality

Betweenness Centrality

Closeness Centrality

PageRank

Composite Influencer Score: Aggregates the above into a unified score.

🔄 Diffusion Models
Independent Cascade (IC): Probabilistic, one-shot influence spread.

Linear Threshold (LT): Cumulative influence activation with thresholds.

Monte Carlo Simulations: Multiple runs per node to assess dynamic influence.

🧪 Datasets
Facebook Circles (4039 nodes / 88234 edges)

Twitter Ego Networks (973 ego-centric graphs)

💻 Tech Stack
Tool	Purpose
Python 3.9+	Core scripting and analysis logic
NetworkX	Graph construction and centrality metrics
NumPy	Efficient numerical computations
Plotly	Interactive graph visualizations
Streamlit	Dynamic, user-friendly dashboard
Matplotlib	Supplementary static plots

🖼️ Dashboard Features
🚀 Launch the app locally:

bash
Copy
Edit
streamlit run dashboard.py
✅ Functionality:
Dataset and model selection

Visual comparison of centrality metrics

Real-time IC / LT simulations

Interactive influencer ranking via Monte Carlo analysis

📂 Repository Structure
bash
Copy
Edit
├── centrality.py          # Centrality algorithms
├── data.py                # Data import & preprocessing
├── diffusion.py           # IC & LT diffusion models
├── metrics.py             # Composite influencer scoring
├── visualization.py       # Network & simulation visualizations
├── dashboard.py           # Streamlit dashboard
├── requirements.txt       # Required Python libraries
└── README.md              # Project overview and documentation
📈 Results Summary
Top Centrality Node (Facebook)	Composite Score	Avg Diffusion Spread (MC)
Node 107	0.3015	2561 (IC), 264 (LT)
Node 351	0.2335	2931 (IC), 277 (LT)
Node 352	0.1928	2971 (IC)

✅ Statistical Significance: Top influencers triggered significantly larger cascades (p < 0.001) than random seeds.

🧭 Key Takeaways
Centrality metrics can reliably detect influential nodes.

IC enables rapid viral spread, while LT captures social reinforcement.

The hybrid static-dynamic approach offers a robust framework for real-world diffusion planning.

🚀 Future Directions
Temporal and evolving networks (real-time graphs)

Edge-weight learning from behavioral data

Scaling simulations using GNN-based approximations

Cross-platform analysis (LinkedIn, Instagram)

📜 License
This project is licensed under the MIT License.

🙏 Acknowledgments
We thank our supervisors Ms. Myriam Gharbi and Ms. Nabila Bitri for their valuable guidance and support.
