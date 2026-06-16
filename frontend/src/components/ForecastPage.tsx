import React from "react";
import { PieChart, Pie, Cell, ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, Legend } from "recharts";
import { ShieldCheck, AlertTriangle, ShieldAlert } from "lucide-react";

interface ForecastProps {
  selectedAssetDetails: any;
  selectedAsset: string;
}

const REGIME_COLORS: { [key: string]: string } = {
  "Bull Market": "#10B981",
  "Bear Market": "#EF4444",
  "Sideways": "#64748B",
  "High Volatility": "#F97316",
  "Recovery": "#06B6D4",
  "Crisis": "#8B5CF6",
};

export default function ForecastPage({ selectedAssetDetails, selectedAsset }: ForecastProps) {
  if (!selectedAssetDetails) {
    return (
      <div style={{ color: "#94A3B8", textAlign: "center", padding: "40px" }}>
        Loading forecast data...
      </div>
    );
  }

  const { probabilities, scenarios } = selectedAssetDetails;
  
  // Get last probability
  const lastProbs = probabilities[probabilities.length - 1] || {};
  
  // Form data for pie chart
  const pieData = Object.keys(lastProbs).map((name) => ({
    name,
    value: lastProbs[name],
  }));

  // Form data for line chart
  const lineData = scenarios.dates.map((date: string, i: number) => ({
    date,
    Best: scenarios.best[i],
    Base: scenarios.base[i],
    Worst: scenarios.worst[i],
  }));

  return (
    <div className="page-container">
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px" }}>
        {/* Left Side: Pie Chart */}
        <div className="panel" style={{ display: "flex", flexDirection: "column", justifySelf: "stretch" }}>
          <div className="panel-header">
            <h3 className="panel-title">Forecast Probability Distribution</h3>
          </div>
          <div style={{ display: "flex", flex: 1, alignItems: "center", gap: "10px" }}>
            <div style={{ height: "200px", width: "50%" }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    paddingAngle={3}
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={REGIME_COLORS[entry.name] || "#64748B"} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value: any) => `${(parseFloat(value) * 100).toFixed(0)}%`} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            
            {/* List with color blocks */}
            <div style={{ display: "flex", flexDirection: "column", gap: "8px", width: "50%" }}>
              {pieData.map((entry) => (
                <div key={entry.name} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", fontSize: "12px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                    <div style={{ width: "10px", height: "10px", borderRadius: "2px", backgroundColor: REGIME_COLORS[entry.name] }}></div>
                    <span style={{ color: "#94A3B8" }}>{entry.name}</span>
                  </div>
                  <span style={{ fontWeight: 700 }}>{(entry.value * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Side: Scenario returns cards */}
        <div className="panel">
          <div className="panel-header">
            <h3 className="panel-title">Scenario Analysis Returns</h3>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            <div className="scenario-card best">
              <span className="scenario-lbl">Best Case Scenario (Bull Regime shift)</span>
              <span className="scenario-ret positive">+{(scenarios.best_return * 100).toFixed(1)}%</span>
            </div>
            <div className="scenario-card base">
              <span className="scenario-lbl">Base Case Scenario (Sideways consensus)</span>
              <span className="scenario-ret" style={{ color: "#3B82F6" }}>
                {scenarios.base_return >= 0 ? "+" : ""}{(scenarios.base_return * 100).toFixed(1)}%
              </span>
            </div>
            <div className="scenario-card worst">
              <span className="scenario-lbl">Worst Case Scenario (Bear Regime crash)</span>
              <span className="scenario-ret negative">{(scenarios.worst_return * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Path Projection Line Chart */}
      <div className="panel">
        <div className="panel-header">
          <h3 className="panel-title">30-Day Scenario Path Projections</h3>
        </div>
        <div style={{ height: "300px", width: "100%" }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={lineData}>
              <XAxis dataKey="date" stroke="#64748B" fontSize={11} tickLine={false} />
              <YAxis stroke="#64748B" fontSize={11} domain={["auto", "auto"]} tickLine={false} />
              <Tooltip
                contentStyle={{ backgroundColor: "#141923", borderColor: "#242D3D" }}
                labelStyle={{ color: "#94A3B8" }}
              />
              <Legend verticalAlign="top" height={36} iconType="circle" />
              <Line type="monotone" dataKey="Best" stroke="#10B981" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="Base" stroke="#3B82F6" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="Worst" stroke="#EF4444" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Scenario Qualitative summary */}
      <div className="panel" style={{ border: "1px solid rgba(234, 179, 8, 0.2)", background: "linear-gradient(135deg, rgba(234, 179, 8, 0.03) 0%, rgba(20, 25, 35, 1) 100%)" }}>
        <div className="panel-header" style={{ marginBottom: "12px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "#EAB308" }}>
            <ShieldAlert size={16} />
            <h3 className="panel-title" style={{ color: "#EAB308" }}>Forecast Explanation</h3>
          </div>
        </div>
        <p style={{ fontSize: "14px", lineHeight: "1.6", color: "#94A3B8" }}>
          Statistical projections over a 30-day window indicate high regime instability. The Base Case expects the asset to continue consolidating within a range-bound structure, yielding expected returns near {(scenarios.base_return * 100).toFixed(1)}%. Under severe selling pressure (Worst Case), volatility is projected to spike to 35%, causing a potential correction of {(scenarios.worst_return * 100).toFixed(1)}%. Maintain defensive allocations or cash-like instruments to buffer downside risks.
        </p>
      </div>
    </div>
  );
}
