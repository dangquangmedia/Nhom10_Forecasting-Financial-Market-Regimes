import React from "react";
import { CheckCircle, AlertTriangle, ShieldCheck } from "lucide-react";

interface ExplainabilityProps {
  explainabilityData: any;
}

export default function ModelExplainabilityPage({ explainabilityData }: ExplainabilityProps) {
  if (!explainabilityData) {
    return (
      <div style={{ color: "#94A3B8", textAlign: "center", padding: "40px" }}>
        Loading model performance details...
      </div>
    );
  }

  const { metrics, confusion_matrix, drift_monitoring, model_version_history } = explainabilityData;

  const ringItems = [
    { label: "Accuracy", value: metrics.accuracy, target: "85%", color: "#3B82F6" },
    { label: "F1-Score", value: metrics.f1_score, target: "80%", color: "#10B981" },
    { label: "Precision", value: metrics.precision, target: "85%", color: "#F97316" },
    { label: "Recall", value: metrics.recall, target: "75%", color: "#06B6D4" },
  ];

  return (
    <div className="page-container">
      {/* Circle metrics row */}
      <div className="panel">
        <div className="panel-header" style={{ marginBottom: "24px" }}>
          <h3 className="panel-title">Model Performance Metrics</h3>
        </div>
        <div className="rings-grid">
          {ringItems.map((item) => {
            const radius = 35;
            const circumference = 2 * Math.PI * radius;
            const offset = circumference - item.value * circumference;

            return (
              <div className="ring-card" key={item.label}>
                <div className="radial-container">
                  <svg className="radial-svg">
                    <circle className="radial-bg" cx="45" cy="45" r={radius} />
                    <circle
                      className="radial-indicator"
                      cx="45"
                      cy="45"
                      r={radius}
                      stroke={item.color}
                      strokeDasharray={circumference}
                      strokeDashoffset={offset}
                    />
                  </svg>
                  <span className="radial-text" style={{ color: item.color }}>
                    {(item.value * 100).toFixed(1)}%
                  </span>
                </div>
                <span className="ring-lbl">{item.label}</span>
                <span className="ring-target">Target: {item.target}</span>
              </div>
            );
          })}
        </div>
        
        {/* Metadata row */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            gap: "24px",
            marginTop: "32px",
            paddingTop: "24px",
            borderTop: "1px solid var(--border-color)",
            textAlign: "center",
          }}
        >
          <div>
            <span style={{ fontSize: "11px", color: "#64748B", textTransform: "uppercase" }}>Last Training Date</span>
            <div style={{ fontSize: "16px", fontWeight: 700, marginTop: "4px" }}>{metrics.last_training_date}</div>
          </div>
          <div>
            <span style={{ fontSize: "11px", color: "#64748B", textTransform: "uppercase" }}>Training Samples</span>
            <div style={{ fontSize: "16px", fontWeight: 700, marginTop: "4px" }}>{metrics.training_samples.toLocaleString()}</div>
          </div>
          <div>
            <span style={{ fontSize: "11px", color: "#64748B", textTransform: "uppercase" }}>Validation Split</span>
            <div style={{ fontSize: "16px", fontWeight: 700, marginTop: "4px" }}>{(metrics.validation_split * 100).toFixed(0)}%</div>
          </div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: "24px" }}>
        {/* Confusion Matrix Table */}
        <div className="panel">
          <div className="panel-header">
            <h3 className="panel-title">Confusion Matrix</h3>
          </div>
          <div className="table-container">
            <table className="matrix-table">
              <thead>
                <tr>
                  <th style={{ border: "none" }}>Actual / Predicted</th>
                  {confusion_matrix.labels.map((lbl: string) => (
                    <th key={lbl}>{lbl}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {confusion_matrix.labels.map((rowLabel: string, rIdx: number) => (
                  <tr key={rowLabel}>
                    <td style={{ fontWeight: 700, color: "#94A3B8", border: "1px solid var(--border-color)", padding: "10px" }}>
                      {rowLabel}
                    </td>
                    {confusion_matrix.labels.map((colLabel: string, cIdx: number) => {
                      const val = confusion_matrix.values[rIdx]?.[cIdx] || 0;
                      const isDiagonal = rIdx === cIdx;
                      return (
                        <td
                          key={colLabel}
                          className={isDiagonal ? "matrix-diagonal" : "matrix-error"}
                        >
                          {val}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Drift & Versions */}
        <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          {/* Drift Status */}
          <div className="panel">
            <div className="panel-header">
              <h3 className="panel-title">Drift Monitoring</h3>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: "13px", color: "#94A3B8" }}>Data Drift</span>
                <span style={{ display: "flex", alignItems: "center", gap: "4px", color: "#10B981", fontSize: "12px", fontWeight: 600 }}>
                  <ShieldCheck size={14} /> Normal
                </span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: "13px", color: "#94A3B8" }}>Prediction Drift</span>
                <span style={{ display: "flex", alignItems: "center", gap: "4px", color: "#10B981", fontSize: "12px", fontWeight: 600 }}>
                  <ShieldCheck size={14} /> Normal
                </span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: "13px", color: "#94A3B8" }}>Concept Drift</span>
                <span style={{ display: "flex", alignItems: "center", gap: "4px", color: "#F97316", fontSize: "12px", fontWeight: 600 }}>
                  <AlertTriangle size={14} /> Warning
                </span>
              </div>
            </div>
          </div>

          {/* Model Version History */}
          <div className="panel" style={{ flex: 1 }}>
            <div className="panel-header">
              <h3 className="panel-title">Model Version History</h3>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
              {model_version_history.map((ver: any, i: number) => (
                <div key={i} style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", paddingBottom: "8px", borderBottom: i < model_version_history.length - 1 ? "1px solid var(--border-color)" : "none" }}>
                  <div>
                    <div style={{ fontWeight: 600, color: "white" }}>{ver.version}</div>
                    <div style={{ color: "#64748B", fontSize: "10px", marginTop: "2px" }}>{ver.date}</div>
                  </div>
                  <span style={{ color: ver.status === "Active" ? "#10B981" : "#64748B" }}>
                    {ver.status}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
