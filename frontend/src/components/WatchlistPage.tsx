import React, { useState } from "react";
import { Bell, BellOff, X, Activity, AlertCircle, Sparkles } from "lucide-react";
import { LineChart, Line, ResponsiveContainer } from "recharts";

interface WatchlistProps {
  assetsList: any[];
  onAssetSelect: (ticker: string) => void;
  selectedAssetDetails: any;
}

const REGIME_COLORS: { [key: string]: string } = {
  "Bull Market": "#10B981",
  "Bear Market": "#EF4444",
  "Sideways": "#64748B",
  "High Volatility": "#F97316",
  "Recovery": "#06B6D4",
  "Crisis": "#8B5CF6",
};

export default function WatchlistPage({ assetsList, onAssetSelect, selectedAssetDetails }: WatchlistProps) {
  const [alertsEnabled, setAlertsEnabled] = useState<{ [key: string]: boolean }>({
    "^GSPC": true,
    "BTC-USD": true,
    "TSLA": true,
    "CL=F": true,
  });

  const [panelOpen, setPanelOpen] = useState(false);
  const [selectedRowAsset, setSelectedRowAsset] = useState<any>(null);

  const toggleAlert = (ticker: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setAlertsEnabled((prev) => ({
      ...prev,
      [ticker]: !prev[ticker],
    }));
  };

  const handleRowClick = (asset: any) => {
    setSelectedRowAsset(asset);
    onAssetSelect(asset.ticker);
    setPanelOpen(true);
  };

  return (
    <div className="page-container" style={{ position: "relative" }}>
      <div className="panel">
        <div className="panel-header">
          <h3 className="panel-title">Asset Watchlist</h3>
          <span style={{ fontSize: "12px", color: "#64748B" }}>Click any row to inspect details</span>
        </div>

        <div className="table-container">
          <table className="custom-table">
            <thead>
              <tr>
                <th>Asset Name</th>
                <th>Last Price</th>
                <th>Daily Change</th>
                <th>Current Regime</th>
                <th>Forecast Regime</th>
                <th>Confidence</th>
                <th>Volatility</th>
                <th>7-Day Trend</th>
                <th>Alerts</th>
              </tr>
            </thead>
            <tbody>
              {assetsList.map((asset) => {
                const isAlertOn = alertsEnabled[asset.ticker];
                const changePct = asset.change_pct;
                const sparkData = asset.sparkline.map((val: number, i: number) => ({ val, index: i }));

                return (
                  <tr key={asset.ticker} onClick={() => handleRowClick(asset)}>
                    <td>
                      <div style={{ display: "flex", flexDirection: "column" }}>
                        <span style={{ fontWeight: 600 }}>{asset.name}</span>
                        <span style={{ fontSize: "11px", color: "#64748B" }}>{asset.ticker}</span>
                      </div>
                    </td>
                    <td style={{ fontWeight: 700 }}>
                      ${asset.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </td>
                    <td style={{ color: changePct >= 0 ? "#10B981" : "#EF4444", fontWeight: 600 }}>
                      <div style={{ display: "flex", flexDirection: "column" }}>
                        <span>{changePct >= 0 ? "+" : ""}{asset.change.toFixed(2)}</span>
                        <span style={{ fontSize: "11px" }}>{changePct >= 0 ? "+" : ""}{(changePct * 100).toFixed(2)}%</span>
                      </div>
                    </td>
                    <td>
                      <span className={`badge ${asset.regime.replace(" ", "-")}`}>
                        {asset.regime}
                      </span>
                    </td>
                    <td>
                      <span className={`badge ${asset.forecast_regime.replace(" ", "-")}`}>
                        {asset.forecast_regime}
                      </span>
                    </td>
                    <td>
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <div className="progress-track" style={{ width: "60px", height: "5px" }}>
                          <div
                            className="progress-fill"
                            style={{
                              width: `${asset.confidence * 100}%`,
                              backgroundColor: "#3B82F6",
                            }}
                          ></div>
                        </div>
                        <span style={{ fontSize: "12px", fontWeight: 600 }}>
                          {(asset.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </td>
                    <td style={{ color: asset.volatility > 0.3 ? "#F97316" : "#94A3B8" }}>
                      {(asset.volatility * 100).toFixed(1)}%
                    </td>
                    <td style={{ width: "100px" }}>
                      <div style={{ height: "30px", width: "90px" }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={sparkData}>
                            <Line
                              type="monotone"
                              dataKey="val"
                              stroke={changePct >= 0 ? "#10B981" : "#EF4444"}
                              strokeWidth={1.5}
                              dot={false}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </td>
                    <td>
                      <button
                        className={`bell-icon-btn ${isAlertOn ? "active" : ""}`}
                        onClick={(e) => toggleAlert(asset.ticker, e)}
                      >
                        {isAlertOn ? <Bell size={16} /> : <BellOff size={16} />}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Slide-out details drawer panel */}
      {selectedRowAsset && (
        <div className={`details-panel ${panelOpen ? "open" : ""}`}>
          <div className="details-panel-header">
            <div>
              <h3 style={{ fontSize: "20px", fontWeight: 700 }}>{selectedRowAsset.name} Details</h3>
              <span style={{ fontSize: "12px", color: "#64748B" }}>{selectedRowAsset.ticker} • {selectedRowAsset.class}</span>
            </div>
            <button className="close-btn" onClick={() => setPanelOpen(false)}>
              <X size={20} />
            </button>
          </div>

          <div className="panel" style={{ padding: "16px", backgroundColor: "var(--bg-main)" }}>
            <span style={{ fontSize: "11px", color: "#64748B", textTransform: "uppercase", fontWeight: 600 }}>Last Price</span>
            <div style={{ fontSize: "28px", fontWeight: 800, color: "white", marginTop: "4px" }}>
              ${selectedRowAsset.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
            <span style={{ fontSize: "13px", color: selectedRowAsset.change_pct >= 0 ? "#10B981" : "#EF4444", fontWeight: 600 }}>
              {selectedRowAsset.change_pct >= 0 ? "+" : ""}{(selectedRowAsset.change_pct * 100).toFixed(2)}% today
            </span>
          </div>

          {selectedAssetDetails ? (
            <>
              {/* Spark Trend Line */}
              <div className="panel" style={{ padding: "16px", backgroundColor: "var(--bg-main)" }}>
                <h4 style={{ fontSize: "13px", marginBottom: "12px", fontWeight: 600, color: "#94A3B8" }}>Recent Price Path</h4>
                <div style={{ height: "120px", width: "100%" }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={selectedAssetDetails.prices.map((p: number, idx: number) => ({ p, idx }))}>
                      <Line type="monotone" dataKey="p" stroke="#3B82F6" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Technical Indicator Metrics */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
                <div className="panel" style={{ padding: "12px", backgroundColor: "var(--bg-main)" }}>
                  <span style={{ fontSize: "10px", color: "#64748B", textTransform: "uppercase" }}>RSI (14)</span>
                  <div style={{ fontSize: "16px", fontWeight: 700, marginTop: "4px" }}>
                    {selectedAssetDetails.technical_indicators?.rsi?.toFixed(1) || "N/A"}
                  </div>
                </div>
                <div className="panel" style={{ padding: "12px", backgroundColor: "var(--bg-main)" }}>
                  <span style={{ fontSize: "10px", color: "#64748B", textTransform: "uppercase" }}>MACD Signal</span>
                  <div style={{ fontSize: "16px", fontWeight: 700, marginTop: "4px" }}>
                    {selectedAssetDetails.technical_indicators?.macd || "N/A"}
                  </div>
                </div>
              </div>

              {/* Regime Details */}
              <div className="panel" style={{ padding: "16px", backgroundColor: "var(--bg-main)" }}>
                <h4 style={{ fontSize: "13px", marginBottom: "8px", fontWeight: 600, color: "#94A3B8" }}>Regime Prediction Details</h4>
                <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ fontSize: "12px", color: "#64748B" }}>Current State</span>
                    <span className={`badge ${selectedRowAsset.regime.replace(" ", "-")}`}>{selectedRowAsset.regime}</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ fontSize: "12px", color: "#64748B" }}>T+1 Forecast</span>
                    <span className={`badge ${selectedRowAsset.forecast_regime.replace(" ", "-")}`}>{selectedRowAsset.forecast_regime}</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ fontSize: "12px", color: "#64748B" }}>Model Confidence</span>
                    <span style={{ fontSize: "12px", fontWeight: 700 }}>{(selectedRowAsset.confidence * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </div>

              {/* AI diagnostic warnings */}
              <div
                className="panel"
                style={{
                  border: "1px solid rgba(249, 115, 22, 0.25)",
                  backgroundColor: "rgba(249, 115, 22, 0.05)",
                  padding: "16px",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "#F97316", marginBottom: "8px" }}>
                  <AlertCircle size={16} />
                  <h4 style={{ fontSize: "13px", fontWeight: 600 }}>AI Diagnostic Alert</h4>
                </div>
                <p style={{ fontSize: "12px", color: "#94A3B8", lineHeight: "1.5" }}>
                  Asset is displaying heightened realized volatility at {(selectedRowAsset.volatility * 100).toFixed(1)}%. Regime transition probabilities lean towards a {selectedRowAsset.forecast_regime} regime in the upcoming week. Hedging or risk adjustment is advised.
                </p>
              </div>
            </>
          ) : (
            <div style={{ color: "#64748B", fontSize: "12px", textAlign: "center" }}>
              Loading additional indicators...
            </div>
          )}
        </div>
      )}
    </div>
  );
}
