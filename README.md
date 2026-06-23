🚀 Live Demo: https://gulxfxv5cq6o24pek6finh.streamlit.app/

# AI-Driven Parking Intelligence System for Bengaluru

## Overview

This project addresses the challenge of **parking-induced traffic congestion** in Bengaluru by transforming parking violation records into actionable enforcement intelligence.

The system identifies illegal parking hotspots, quantifies their congestion impact, forecasts future risk, and generates patrol-ready enforcement recommendations through a multi-phase analytics pipeline.

---

# Problem Statement

### Operational Challenge
Illegal parking near:
- Commercial zones
- Metro stations
- Busy intersections
- Main roads
- Event locations

often reduces road capacity and creates congestion.

### Key Questions
- Where are the most critical parking hotspots?
- Which locations create the highest traffic impact?
- Where should enforcement teams be deployed first?
- Which hotspots are likely to worsen tomorrow?

---

# Project Architecture

## Phase 1 — Data Engineering & Spatial Risk Mapping

### Objectives
- Clean and enrich raw parking violation data
- Generate congestion risk scores
- Build spatial intelligence layers

### Key Tasks
- Timestamp standardization (UTC → IST)
- Violation type extraction and cleaning
- Vehicle impact weighting
- Risk score calculation
- H3 hexagonal spatial indexing
- Reverse geocoding of missing junctions
- Exploratory Data Analysis
- Interactive hotspot heatmap generation

### Outputs
- `violations_geocoded.csv`
- `h3_hexagons.csv`
- EDA reports
- Interactive Folium maps

---

## Phase 2 — Hotspot Detection & Congestion Impact Index (CII)

### Objectives
Identify natural parking hotspots and rank them by severity.

### Techniques Used

#### DBSCAN Clustering
Groups violations based on geographic proximity.

Parameters:
- Radius: 150 meters
- Minimum samples: 25

#### Congestion Impact Index (CII)

Combines:

1. Frequency of violations
2. Severity of violations
3. Spatial persistence

Result:
- Normalized score (0–100)
- Enforcement priority ranking

### Outputs
- Hotspot clusters
- Congestion Impact Index
- Ranked enforcement zones
- Enforcement priority table

---

## Phase 3 — Commander Dashboard & Patrol Route Planning

### Objectives
Convert analytics into decision-support tools.

### Features

#### Interactive Dashboard
- Priority hotspot map
- Risk analytics
- Historical trends
- Hotspot rankings

#### Patrol Route Optimizer
Nearest-neighbour route planning for enforcement teams.

Outputs:
- Patrol sequence
- Route distance estimates
- High-priority coverage plan

### Deliverable
Standalone interactive HTML dashboard.

---

## Phase 4 — Traffic Delay Quantification

### Objectives
Estimate how parking violations affect traffic movement.

### Key Components
- H3-level risk aggregation
- Junction mapping
- Priority zone analysis
- Congestion impact estimation

### Outputs
- Delay estimates
- Junction-level congestion intelligence
- Risk impact summaries

---

## Phase 5 — Predictive Enforcement Intelligence

### Objectives
Forecast future congestion risk.

### Machine Learning Pipeline

#### Features
- Historical risk lags
- Neighboring hotspot risk
- Rolling statistics
- Day-of-week patterns

#### Model
CatBoost-based forecasting pipeline

### Predictions
- Future hotspot risk
- Trend detection
- Alert generation

### Alert Levels
- CRITICAL
- WATCH
- INFO

---

# Technologies Used

## Data Processing
- Python
- Pandas
- NumPy

## Geospatial Analytics
- H3
- Folium

## Machine Learning
- CatBoost
- XGBoost
- Scikit-Learn

## Clustering
- DBSCAN

## Visualization
- Matplotlib
- Seaborn
- Leaflet
- Chart.js

---

# Workflow

Raw Violation Data
↓
Data Cleaning & Risk Scoring
↓
H3 Spatial Aggregation
↓
Hotspot Detection (DBSCAN)
↓
Congestion Impact Index
↓
Priority Ranking
↓
Dashboard Generation
↓
Traffic Delay Quantification
↓
Risk Forecasting
↓
Enforcement Alerts

---

# Key Innovations

- Spatial hotspot detection using H3 + DBSCAN
- Congestion Impact Index (CII)
- Risk-based enforcement prioritization
- Patrol route optimization
- Future hotspot forecasting
- Interactive commander dashboard

---

# Expected Impact

The system enables traffic authorities to move from:

**Reactive Enforcement**
→ Responding after congestion occurs

to

**Predictive Enforcement**
→ Deploying resources before congestion becomes severe.

---

# Repository Structure

```text
Phase1.ipynb   -> Data Cleaning & Spatial Mapping
Phase 2.ipynb  -> Hotspot Detection & CII
Phase 3.ipynb  -> Dashboard & Route Planning
Phase 4.ipynb  -> Traffic Delay Quantification
Phase5.ipynb   -> Forecasting & Alert Generation
```

---

## Author

Neeraj Sharma  
IIT Guwahati

