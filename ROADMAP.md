# 🗺️ Project Roadmap & Team Assignments

This document outlines the current limitations of the TokenFlow system and the development roadmap assigned to the 3-member project team.

## 🚧 Current Limitations

1.  **Telemetry**: Limited to TCP RTT/Retransmission. Lacks application-level metrics (HTTP/gRPC) and hardware correlation (GPU/CPU).
2.  **Modeling**: Simple offline training (Isolation Forest + GBDT). No online/incremental learning or support for advanced sequence models (LSTM/Transformer).
3.  **System**: Simulation-only LLM integration. Single-node architecture.
4.  **Engineering**: Lack of containerization (Docker), automated testing, and comprehensive CI/CD.

---

## 👥 Team Assignments

### 🧑‍💻 Member A: System & Kernel Expert
**Focus: Advanced Telemetry & Real Integration**

*   **Task A1: Advanced eBPF & Hardware Telemetry**
    *   [ ] (Optional) HTTP/gRPC latency tracking via eBPF uprobes.

*   **Task A2: Real LLM Engine Integration**
    *   [ ] Deploy a real inference engine (e.g., **vLLM** or **Ollama**).
    *   [ ] Develop `real_llm_client.py`: A client that sends prompts to the engine while respecting the rate limits from the Hint Server.

### 🧑‍🔬 Member B: AI & Strategy Expert
**Focus: Model Optimization & Control Algorithms**

*   **Task B1: Model Benchmarking & Online Learning**
    *   [ ] **Model Comparison**: Train and compare LSTM/Transformer models against the current GBDT for RTT prediction.
    *   [ ] **Online Learning**: Implement incremental learning (using `River` or `Scikit-multiflow`) to adapt to network changes without retraining.

*   **Task B2: Advanced Control Algorithms**
    *   [ ] **PID Controller**: Replace the current Sigmoid mapping with a PID controller to stabilize RTT at a target setpoint (e.g., 50ms).
    *   [ ] (Advanced) **Reinforcement Learning**: Explore RL agents to dynamically optimize the Throughput-Latency trade-off.

### 🛠️ Member C: Fullstack & DevOps Expert
**Focus: Visualization, Engineering & QA**

*   **Task C1: Dashboard 2.0**
    *   [ ] **Multi-Metric View**: Visualize GPU metrics and socket backlog alongside RTT.
    *   [ ] **A/B Testing View**: Create a mode to compare "Pacing ON" vs "Pacing OFF" performance side-by-side.
    *   [ ] **Historical Data**: Add date pickers to view historical telemetry logs.

*   **Task C2: Engineering & Quality Assurance**
    *   [ ] **Dockerization**: Create `Dockerfile` and `docker-compose.yml` for one-click deployment (Agent, Redis, Server, Dashboard).
    *   [ ] **Storage Upgrade**: Migrate data storage from CSV to **Redis** for production-grade performance.
    *   [ ] **Automated Testing**: Write `pytest` cases to verify Hint Server logic and Pacer response.

---

## 📅 Collaboration Plan

*   **Data Interface**: Members A & B to define the extended feature set (e.g., adding `gpu_util` column).
*   **API Contract**: Members B & C to finalize the Hint Server JSON response format.

### Phase 1 (Week 1)
*   A: GPU Monitor.
*   B: PID Controller.
*   C: Docker Environment & Redis Integration.

### Phase 2 (Week 2)
*   A: vLLM Integration.
*   B: LSTM Model.
*   C: Dashboard 2.0 (A/B View).

