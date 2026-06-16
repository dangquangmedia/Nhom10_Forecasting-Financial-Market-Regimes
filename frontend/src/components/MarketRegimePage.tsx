import React from "react";

const REGIME_LEGEND = [
  {
    name: "Bull Market",
    color: "#10B981",
    description: "Strong upward trend with low volatility and high buyer support.",
  },
  {
    name: "Bear Market",
    color: "#EF4444",
    description: "Strong downward trend characterized by fear, heavy selling, and distribution.",
  },
  {
    name: "Sideways",
    color: "#64748B",
    description: "Range-bound consolidation, lacks clear directional momentum.",
  },
  {
    name: "High Volatility",
    color: "#F97316",
    description: "Large, rapid price swings in both directions. Heightened uncertainty.",
  },
  {
    name: "Crisis / Risk-off",
    color: "#8B5CF6",
    description: "Severe liquidity stress or extreme macroeconomic instability.",
  },
  {
    name: "Recovery / Risk-on",
    color: "#06B6D4",
    description: "Initial rebound from a bearish phase, increasing positive structural changes.",
  },
];

const TRANSITION_PROBS = [
  { from: "Bull", toBull: "65%", toBear: "5%", toSide: "20%", toVol: "8%", toCrisis: "1%", toRecov: "1%" },
  { from: "Bear", toBull: "5%", toBear: "60%", toSide: "15%", toVol: "12%", toCrisis: "6%", toRecov: "2%" },
  { from: "Sideways", toBull: "25%", toBear: "20%", toSide: "35%", toVol: "15%", toCrisis: "2%", toRecov: "3%" },
  { from: "High Vol", toBull: "10%", toBear: "15%", toSide: "20%", toVol: "40%", toCrisis: "10%", toRecov: "5%" },
  { from: "Crisis", toBull: "2%", toBear: "25%", toSide: "10%", toVol: "20%", toCrisis: "35%", toRecov: "8%" },
  { from: "Recovery", toBull: "35%", toBear: "8%", toSide: "15%", toVol: "12%", toCrisis: "2%", toRecov: "28%" },
];

export default function MarketRegimePage() {
  return (
    <div className="page-container">
      {/* Regime Timeline panel */}
      <div className="panel">
        <div className="panel-header">
          <h3 className="panel-title">Regime Timeline</h3>
        </div>
        <div className="timeline-bar">
          <div className="timeline-segment" style={{ width: "20%", backgroundColor: "#10B981" }}>Bull</div>
          <div className="timeline-segment" style={{ width: "25%", backgroundColor: "#64748B" }}>Sideways</div>
          <div className="timeline-segment" style={{ width: "20%", backgroundColor: "#EF4444" }}>Bear</div>
          <div className="timeline-segment" style={{ width: "20%", backgroundColor: "#06B6D4" }}>Recovery</div>
          <div className="timeline-segment" style={{ width: "15%", backgroundColor: "#10B981" }}>Bull</div>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", padding: "0 10px", fontSize: "12px", color: "#64748B", fontWeight: 600 }}>
          <span>Jan</span>
          <span>Mar</span>
          <span>May</span>
          <span>Jul</span>
          <span>Aug</span>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: "24px" }}>
        {/* Transition Matrix */}
        <div className="panel">
          <div className="panel-header">
            <h3 className="panel-title">Regime Transition Matrix</h3>
          </div>
          <div className="table-container">
            <table className="custom-table" style={{ width: "100%" }}>
              <thead>
                <tr>
                  <th>From / To</th>
                  <th style={{ color: "#10B981", textAlign: "center" }}>Bull</th>
                  <th style={{ color: "#EF4444", textAlign: "center" }}>Bear</th>
                  <th style={{ color: "#64748B", textAlign: "center" }}>Side</th>
                  <th style={{ color: "#F97316", textAlign: "center" }}>Vol</th>
                  <th style={{ color: "#8B5CF6", textAlign: "center" }}>Crisis</th>
                  <th style={{ color: "#06B6D4", textAlign: "center" }}>Recov</th>
                </tr>
              </thead>
              <tbody>
                {TRANSITION_PROBS.map((row, index) => (
                  <tr key={index}>
                    <td style={{ fontWeight: 600, color: "#94A3B8" }}>{row.from}</td>
                    <td style={{ textAlign: "center", color: "#10B981", fontWeight: 600 }}>{row.toBull}</td>
                    <td style={{ textAlign: "center", color: "#EF4444", fontWeight: 600 }}>{row.toBear}</td>
                    <td style={{ textAlign: "center", color: "#94A3B8" }}>{row.toSide}</td>
                    <td style={{ textAlign: "center", color: "#F97316", fontWeight: 600 }}>{row.toVol}</td>
                    <td style={{ textAlign: "center", color: "#8B5CF6", fontWeight: 600 }}>{row.toCrisis}</td>
                    <td style={{ textAlign: "center", color: "#06B6D4", fontWeight: 600 }}>{row.toRecov}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Legend Panel */}
        <div className="panel">
          <div className="panel-header">
            <h3 className="panel-title">Regime Color Legend</h3>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
            {REGIME_LEGEND.map((item) => (
              <div
                key={item.name}
                style={{
                  display: "flex",
                  alignItems: "flex-start",
                  gap: "12px",
                  padding: "10px",
                  borderRadius: "8px",
                  backgroundColor: "rgba(30, 37, 51, 0.3)",
                }}
              >
                <div
                  style={{
                    width: "14px",
                    height: "14px",
                    backgroundColor: item.color,
                    borderRadius: "3px",
                    flexShrink: 0,
                    marginTop: "3px",
                  }}
                ></div>
                <div style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
                  <span style={{ fontSize: "13px", fontWeight: 600, color: "white" }}>{item.name}</span>
                  <span style={{ fontSize: "11px", color: "#94A3B8", lineHeight: "1.4" }}>
                    {item.description}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
